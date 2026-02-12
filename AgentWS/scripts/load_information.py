import faiss
import numpy as np
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

# =========================
# CONFIGURACIÓN
# =========================

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DATA_PATH = "data/information.json"
INDEX_PATH = "data/info_index.faiss"
META_PATH = "data/info_meta.npy"

# =========================
# GENERAR DOCUMENTOS
# =========================

def generar_documentos():
    """
    Genera documentos semánticos para FAISS.
    Cada documento incluye:
    - texto: lo que se indexa
    - ubicacion_id: ID real del sistema
    - tipo_info: horario | contacto | direccion
    - nombre: display humano
    """

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []

    for ubicacion_id, info in data.items():
        nombre = info["nombre"]

        # Horario
        docs.append({
            "texto": f"Horario de atención de la delegación de {nombre}.",
            "ubicacion_id": ubicacion_id,
            "nombre": nombre,
            "tipo_info": "horario"
        })

        # Contacto
        docs.append({
            "texto": f"Datos de contacto de la delegación de {nombre}.",
            "ubicacion_id": ubicacion_id,
            "nombre": nombre,
            "tipo_info": "contacto"
        })

        # Dirección
        if "direccion" in info:
            docs.append({
                "texto": f"Dirección de la delegación de {nombre}: {info['direccion']}.",
                "ubicacion_id": ubicacion_id,
                "nombre": nombre,
                "tipo_info": "direccion"
            })

    return docs

# =========================
# MAIN
# =========================

def main():
    docs = generar_documentos()

    textos = [d["texto"] for d in docs]

    # 1️⃣ Crear embeddings
    emb_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=textos
    )

    embeddings = np.array(
        [e.embedding for e in emb_response.data],
        dtype="float32"
    )

    # 2️⃣ Crear índice FAISS
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # 3️⃣ Guardar índice y metadatos
    faiss.write_index(index, INDEX_PATH)
    np.save(META_PATH, docs, allow_pickle=True)

    print(f"✅ Índice FAISS de información creado con {len(docs)} documentos")


if __name__ == "__main__":
    main()