import json
from app.llm import interpretar_mensaje, generar_respuesta_informacion, generar_respuesta_stock
from app.index_vector import buscar_similares
from app.information import obtener_info, resolver_ubicacion
from app.memory import get_context, set_context, add_message
from app.domain_config import LIMITS_DOMAIN
from app.logger import get_logger

logger = get_logger("AGENT")

# =========================
# HANDLERS POR DOMINIO
# =========================

def handle_out_of_scope():
    return {
        "respuesta": (
            "Gracias por tu mensaje 😊\n\n"
            "Voy a derivar tu consulta a uno de nuestros asesores "
            "para ayudarte correctamente."
        )
    }

def handle_informacion(slots, scope, texto_usuario):
    """
    Dominio: informacion
    Slots esperados:
      - ubicacion
      - tipo_info
    """

    ubicacion_slot = slots.get("ubicacion")
    tipo_info = slots.get("tipo_info")

    logger.info(f"Slot ubicación recibido: {ubicacion_slot}")
    logger.info(f"Tipo info: {tipo_info}")
    logger.info(f"Scope: {scope}")
    # Si no hay informacion relevante 
    if scope == "all":
    # 1️⃣ Cargar base completa
        with open("data/information.json", "r", encoding="utf-8") as f:
            data_all = json.load(f)

        if not data_all:
            return {
                "respuesta": "No hay información de delegaciones disponible en este momento."
            }

        # 2️⃣ Obtener límite por dominio (seguro)
        limite = LIMITS_DOMAIN.get("informacion", 3)
        total = len(data_all)

        respuestas = []

        # 3️⃣ Iterar con límite
        for idx, info in enumerate(data_all.values()):
            if idx >= limite:
                break

            respuestas.append(
                generar_respuesta_informacion(
                    ubicacion=info.get("nombre"),
                    tipo_info=tipo_info,
                    datos=info
                )
            )

        # 4️⃣ Mensaje de control (UX)
        mensaje_control = (
            f"📌 Mostrando {len(respuestas)} de {total} delegaciones disponibles.\n\n"
            "Si deseas información de una sede específica, indícalo por favor.\n\n"
        )

        return {
            "respuesta": mensaje_control + "\n\n".join(respuestas)
        }
    # 1️⃣ Si NO hay ubicación → chat de aclaración
    if not ubicacion_slot:
        return {
            "respuesta": (
                "Claro 😊 ¿De qué delegación necesitas la información?\n"
                "Por ejemplo: Santiago, La Coruña, Ferrol, Culleredo…"
            )
        }
    # 🔑 Resolver ubicación REAL (ID canónico)
    ubicacion_id = resolver_ubicacion(ubicacion_slot, texto_usuario)

    # ❗ Si no se pudo resolver → pedir precisión
    if not ubicacion_id:
        return {
            "respuesta": (
                "¿De qué delegación necesitas la información?\n"
                "Por ejemplo: Santiago, La Coruña, Ferrol…"
            )
        }

    info = obtener_info(ubicacion_id)

    # 🧠 El LLM SOLO redacta
    respuesta = generar_respuesta_informacion(
        ubicacion=info["nombre"],
        tipo_info=tipo_info,
        datos=info
    )

    return {"respuesta": respuesta}


def handle_producto(slots):
    producto = slots.get("producto")

    if not producto:
        return {
            "respuesta": "¿Podrías indicarme qué producto estás buscando?"
        }

    return {
        "respuesta": (
            f"Puedo ayudarte con información del producto '{producto}'. "
            "Si deseas conocer disponibilidad, pregúntame por el stock."
        )
    }


def handle_stock(slots, scope, texto_usuario):

    producto = slots.get("producto")
    marca = slots.get("marca")
    ubicacion = slots.get("ubicacion")  # 👈 ya interpretado por el LLM

    logger.info(f"[STOCK] Producto: {producto}")
    logger.info(f"[STOCK] Marca: {marca}")
    logger.info(f"[STOCK] Scope: {scope}")
    logger.info(f"[STOCK] Ubicación: {ubicacion}")

    # -------------------------
    # VALIDACIÓN BASE
    # -------------------------
    if not producto and not marca:
        return {
            "respuesta": generar_respuesta_stock(
                ubicacion,
                {"error": "missing_product"}
            )
        }

    # -------------------------
    # TEXTO DE BÚSQUEDA FAISS
    # -------------------------
    logger.info(f"[STOCK] Formando texto para buscar en el vector el producto: {producto}")
    tokens = []
    if producto:
        tokens.append(producto)
    if marca:
        tokens.append(marca)

    texto_busqueda = " ".join(tokens).strip()
    limite = LIMITS_DOMAIN.get("stock", 5)

    # -------------------------
    # BÚSQUEDA FAISS
    # -------------------------
    logger.info(f"[STOCK] Proceso de busqueda vectorial: {producto}")
    resultados_faiss = buscar_similares(
        texto=texto_busqueda,
        index_name="stock",
        top_k=limite * 3
    )

    if not resultados_faiss:
        data_final = {
            "scope": scope,
            "producto": producto,
            "ubicacion": ubicacion,
            "limite": limite,
            "resultados": []
        }
        return {
            "respuesta": generar_respuesta_stock(ubicacion, data_final)
        }

    # -------------------------
    # FILTRADO SEGÚN SCOPE
    # -------------------------
    logger.info(f"[STOCK] Resultado de busqueda: {resultados_faiss}")
  
    resultados_finales = []

    for r in resultados_faiss:

        descripcion = r.get("descripcion")
        stock_total = int(r.get("stock_total", 0))
        ubicaciones = r.get("ubicaciones", {})

        # SINGLE → solo una tienda
        if scope == "single" and ubicacion:
            stock_tienda = int(ubicaciones.get(ubicacion, 0))
            if stock_tienda <= 0:
                continue

            resultados_finales.append({
                "descripcion": descripcion,
                "stock_total": stock_total,
                "ubicaciones": {ubicacion: stock_tienda}
            })

        # ALL → todas las tiendas
        elif scope == "all":
            if stock_total <= 0:
                continue

            resultados_finales.append({
                "descripcion": descripcion,
                "stock_total": stock_total,
                "ubicaciones": ubicaciones
            })

        if len(resultados_finales) >= limite:
            break

    # -------------------------
    # DATA FINAL
    # -------------------------
    data_final = {
        "scope": scope,
        "producto": producto,
        "ubicacion": ubicacion,
        "limite": limite,
        "resultados": resultados_finales
    }

    # -------------------------
    # GENERAR RESPUESTA UX
    # -------------------------
    respuesta_texto = generar_respuesta_stock(ubicacion, data_final)

    logger.info("Respuesta STOCK generada correctamente")

    return {"respuesta": respuesta_texto}
# =========================
# FUNCIÓN PRINCIPAL
# =========================

def procesar_mensaje(texto_usuario: str, session_id: str = "default"):
    add_message(session_id, "user", texto_usuario)
    logger.info(f"Procesando mensaje: {texto_usuario}")

    # -------------------------
    # CONTEXTO PREVIO
    # -------------------------
    contexto = get_context(session_id) or {}
    logger.debug(f"Contexto previo: {contexto}")

    # -------------------------
    # INTERPRETACIÓN LLM
    # -------------------------
    data = interpretar_mensaje(texto_usuario, contexto)
    domain = data.get("domain")
    scope = data.get("scope", "single")
    slots = data.get("slots", {})

    logger.info(f"Dominio detectado: {domain}")
    logger.debug(f"Slots detectados: {slots}")
    logger.debug(f"Scope detectado: {scope}")

    # -------------------------
    # HERENCIA DE CONTEXTO
    # -------------------------
    for clave in ["producto", "marca", "ubicacion"]:
        if not slots.get(clave) and contexto.get(clave):
            slots[clave] = contexto[clave]

    # -------------------------
    # RESET CORRECTO SOLO POR SCOPE
    # -------------------------
    if contexto.get("scope") == "single" and scope == "all":
        # el usuario pidió "todos" → la ubicación ya no aplica
        slots.pop("ubicacion", None)
        contexto.pop("ubicacion", None)

    # -------------------------
    # GUARDAR CONTEXTO
    # -------------------------
    contexto["domain"] = domain
    contexto["scope"] = scope

    for clave in ["producto", "marca", "ubicacion"]:
        if slots.get(clave):
            contexto[clave] = slots[clave]

    set_context(session_id, contexto)
    logger.debug(f"Contexto actualizado: {contexto}")
        # -------------------------
    # ROUTING
    # -------------------------
    if domain == "out_of_scope":
        respuesta = handle_out_of_scope()

    elif domain == "informacion":
        respuesta = handle_informacion(slots, scope, texto_usuario)

    elif domain == "producto":
        respuesta = handle_producto(slots)

    elif domain == "stock":
        respuesta = handle_stock(slots, scope, texto_usuario)

    else:
        respuesta = handle_out_of_scope()

    add_message(session_id, "assistant", respuesta["respuesta"])
    return respuesta