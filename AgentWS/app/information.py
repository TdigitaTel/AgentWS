import json
from pathlib import Path
from app.index_vector import buscar_similares
from app.logger import get_logger

logger = get_logger("INFORMACION")

DATA_PATH = Path("data/information.json")  # 👈 ojo al typo

def obtener_info(ubicacion: str):
    logger.info(f"Obteniendo información de ubicación: {ubicacion}")

    if not ubicacion or not DATA_PATH.exists():
        logger.warning("Ubicación no encontrada o archivo inexistente")
        return None

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get(ubicacion)

def resolver_ubicacion(ubicacion, texto_usuario):
    """
    Devuelve un ID válido de ubicación o None
    """
    logger.info(f"Obteniendo información de ubicación: {ubicacion}")
    logger.info(f"Texto usuario: {texto_usuario}")

    # 1️⃣ Si ya es un ID válido, úsalo
    if ubicacion and obtener_info(ubicacion):
        logger.info(f"Id de ubicación identificado: {ubicacion}")
        return ubicacion

    # 2️⃣ Resolver semánticamente con FAISS
    resultados = buscar_similares(
        texto=texto_usuario,
        index_name="informacion",
        top_k=1
    )
    logger.info(f"Resultados Probables: {resultados}")

    if not resultados:
        logger.info(f"Resultados No encontrados con la siguiente ubicación: {ubicacion}")
        return None

    candidato = resultados[0].get("ubicacion_id")
    logger.info(f"Candidato seleccionado: {candidato}")
  
    # 3️⃣ Validar contra BD
    if candidato and obtener_info(candidato):
        logger.info(f"Candidato/ informacion seleccionada: {candidato}")
        return candidato

    return None