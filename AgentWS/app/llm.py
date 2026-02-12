import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from app.logger import get_logger
from pathlib import Path
from app.domain_config import DOMAINS_VALIDOS, LIMITS_DOMAIN

logger = get_logger("LLM")

# =========================
# SETUP
# =========================

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROMPT_PATH = Path("prompts/interpretation.txt")
BASE_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")


# =========================
# INTERPRETACIÓN (LLM)
# =========================

def interpretar_mensaje(texto: str, contexto: dict | None = None) -> dict:
    """
    El LLM:
    - Clasifica dominio
    - Extrae señales explícitas
    - Devuelve SIEMPRE la MISMA estructura plana
    """

    logger.info(f"Interpretando mensaje: {texto}")
    contexto = contexto or {}

    try:
        messages = [
            {"role": "system", "content": BASE_PROMPT}
        ]

        # 👉 Historial conversacional (si existe)
        for m in contexto.get("messages", []):
            messages.append({
                "role": m["role"],
                "content": m["content"]
            })

        messages.append({
            "role": "user",
            "content": texto
        })

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
            max_tokens=200
        )

        contenido = response.choices[0].message.content.strip()
        logger.debug(f"Respuesta cruda del LLM: {contenido}")

        try:
            data = json.loads(contenido)
        except json.JSONDecodeError:
            logger.error(
                "El LLM devolvió JSON inválido",
                extra={"respuesta": contenido}
            )
            return {
                "domain": "out_of_scope",
                "scope": "single",
                "slots": {}
            }
        # 🔒 Validación mínima de dominio
        domain = data.get("domain")
        if domain not in DOMAINS_VALIDOS:
            logger.warning(f"Dominio inválido detectado: {domain}")
            data["domain"] = "out_of_scope"

        # 🔒 Normalización de salida (CONTRATO FIJO)
        resultado = {
            "domain": data.get("domain", "out_of_scope"),
            "scope": data.get("scope", "single"),  # 👈 NUEVO
            "slots": data.get("slots", {}) or {}
        }

        logger.info(f"Resultado LLM normalizado: {resultado}")
        return resultado

    except Exception as e:
        logger.exception(f"Error en LLM: {e}")
        return {
            "domain": "out_of_scope",
            "scope": "single",
            "slots": {}
        }


# =========================
# GENERACIÓN DE RESPUESTA
# =========================

def generar_respuesta_informacion(ubicacion, tipo_info, datos):
    """
    El LLM SOLO renderiza la respuesta final
    con formato controlado y consistente.
    """

    prompt = f"""
Eres un asistente de atención al cliente de una empresa.

INSTRUCCIONES OBLIGATORIAS:
- Usa SOLO la información proporcionada
- NO inventes datos
- NO agregues información externa
- NO cambies el orden ni el formato
- NO personalices el saludo
- Usa SIEMPRE el mismo formato

FORMATO DE RESPUESTA (OBLIGATORIO):

Ubicación: {{nombre}}
Dirección: {{direccion}}
Email: {{email}}
Teléfono: {{telefono}}
Horario:
{{horario}}

REGLAS DE FORMATO:
- Si un campo no existe o es null, escribe: "No disponible"
- El horario debe listarse por líneas (Lunes a viernes, Sábado, etc.)
- No agregues texto antes ni después del bloque

DATOS DISPONIBLES (JSON):
{json.dumps(datos, ensure_ascii=False, indent=2)}

Genera la respuesta EXACTAMENTE con el formato indicado.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200
    )

    return response.choices[0].message.content.strip()

def generar_respuesta_stock(ubicacion_slot: str | None, data: dict) -> str:
    """
    Genera una respuesta de stock usando LLM SOLO para UX.
    La lógica ya viene resuelta desde backend.
    """

    resultados = data.get("resultados", [])
    scope = data.get("scope")
    producto = data.get("producto")
    limite = data.get("limite", 0)

    if not resultados:
        return f"No encontré stock disponible{f' de {producto}' if producto else ''}."

    # -------------------------
    # PAYLOAD DE PRESENTACIÓN (UX)
    # -------------------------
    payload = {
        "producto": producto,
        "scope": scope,
        "ubicacion": ubicacion_slot,
        "limite": limite,
        "items": []
    }

    for r in resultados:
        item = {
            "descripcion": r.get("descripcion"),
            "stock_total": r.get("stock_total"),
            "ubicaciones": r.get("ubicaciones", {})
        }
        payload["items"].append(item)

    # -------------------------
    # PROMPT UX (SOLO FORMATO)
    # -------------------------
    prompt = f"""
Eres un asistente de atención al cliente por chat (WhatsApp).

Tu tarea es MOSTRAR información de stock de forma clara, breve y amigable.

REGLAS:
- NO inventes datos
- NO hagas cálculos nuevos
- NO expliques el proceso
- Usa emojis con moderación
- Sé claro y orientado al cliente
- Muestra SOLO hasta el límite indicado
- Después de listar productos, pide al usuario que especifique si desea algo más

COMPORTAMIENTO SEGÚN SCOPE:

Si scope = "single":
- Muestra el stock SOLO de la tienda indicada
- También muestra el stock total de la empresa

Si scope = "all":
- Muestra el stock por tienda
- Muestra también el total de la empresa

FORMATO RECOMENDADO:
- Lista numerada
- Cada producto en un bloque corto
- Al final, una pregunta para continuar

DATOS (JSON):
{json.dumps(payload, ensure_ascii=False, indent=2)}

Genera la respuesta final para el cliente.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300
    )

    return response.choices[0].message.content.strip()