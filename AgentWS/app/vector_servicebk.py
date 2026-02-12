# app/vector_service.py

import faiss
import numpy as np
import os
from openai import OpenAI
from dotenv import load_dotenv
from app.logger import get_logger

logger = get_logger("FAISS")

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

index = faiss.read_index("data/faiss.index")
metadata = np.load("data/faiss_meta.npy", allow_pickle=True)

def buscar_similares(texto: str, top_k: int = 5):
    logger.info(f"Búsqueda semántica para: {texto}")

    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=texto
    ).data[0].embedding

    emb = np.array([emb]).astype("float32")
    distancias, indices = index.search(emb, top_k)

    resultados = []
    for i, idx in enumerate(indices[0]):
        meta = metadata[idx]
        resultados.append({
            "articulo": meta.get("articulo"),
            "descripcion": meta.get("descripcion"),
            "stock": meta.get("stock"),
            "ubicacion": meta.get("ubicacion"),
            "distancia": float(distancias[0][i])
        })

    logger.info(f"Resultados FAISS: {resultados}")
    return resultados