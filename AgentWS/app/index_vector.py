import faiss
import numpy as np
import os
from openai import OpenAI
from dotenv import load_dotenv
from app.logger import get_logger

# =========================
# CONFIGURACIÓN INICIAL
# =========================

load_dotenv()
logger = get_logger("FAISS")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# CARGA DE ÍNDICES FAISS
# (SE CARGAN UNA SOLA VEZ)
# =========================

INDICES = {}

def cargar_indices():
    """
    Carga todos los índices FAISS disponibles.
    Se ejecuta una sola vez al iniciar la app.
    """
    global INDICES

    INDICES = {
        "stock": {
            "index": faiss.read_index("data/faiss_stock_single.index"),
            "meta": np.load("data/faiss_stock_single_meta.npy", allow_pickle=True)
        },
        "informacion": {
            "index": faiss.read_index("data/info_index.faiss"),
            "meta": np.load("data/info_meta.npy", allow_pickle=True)
        }
    }

    logger.info(f"Índices FAISS cargados: {list(INDICES.keys())}")


# Cargar automáticamente al importar el módulo
cargar_indices()

# =========================
# FUNCIÓN GENERAL DE BÚSQUEDA
# =========================

def buscar_similares(
    texto: str,
    index_name: str,
    top_k: int = 5
):
    """
    Búsqueda semántica FAISS controlada por índice.

    Parámetros:
    - texto: texto de búsqueda (string obligatorio)
    - index_name: nombre del índice FAISS ("stock", "informacion", etc.)
    - top_k: número máximo de resultados
    """

    if not texto or not texto.strip():
        logger.warning("FAISS → texto vacío")
        return []

    if index_name not in INDICES:
        logger.error(f"FAISS → índice no registrado: {index_name}")
        return []

    logger.info(f"FAISS → index={index_name}, texto='{texto}', top_k={top_k}")

    # 1️⃣ Generar embedding
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=texto
    ).data[0].embedding

    vector = np.array([embedding], dtype="float32")

    # 2️⃣ Buscar en índice correspondiente
    index = INDICES[index_name]["index"]
    metadata = INDICES[index_name]["meta"]

    _, indices = index.search(vector, top_k)

    # 3️⃣ Mapear resultados
    resultados = []
    for idx in indices[0]:
        if idx < 0 or idx >= len(metadata):
            continue
        resultados.append(metadata[idx])

    logger.info(f"FAISS → resultados devueltos ({index_name}): {len(resultados)}")
    return resultados