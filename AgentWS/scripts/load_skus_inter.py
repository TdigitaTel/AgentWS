import os
import json
import numpy as np
import pandas as pd
import faiss
from dotenv import load_dotenv
from openai import OpenAI

# =========================
# CONFIG
# =========================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "❌ Falta OPENAI_API_KEY"

client = OpenAI(api_key=OPENAI_API_KEY)

EXCEL_PATH = "data/stocks_store.xlsx"

INDEX_PATH = "data/faiss_stock_single.index"
META_PATH  = "data/faiss_stock_single_meta.npy"

MODEL_EMBEDDING = "text-embedding-3-small"

UBICACIONES = {
    "almeiras": "Stock Almeiras",
    "santiago": "Stock Santiago",
    "ferrol": "Stock Ferrol",
    "sandiego": "Stock Sandiego",
    "sanxenxo": "Stock SanXenxo"
}

BATCH_SIZE = 100  # 👈 importante

# =========================
# CARGA EXCEL
# =========================
print("📥 Cargando Excel...")
df = pd.read_excel(EXCEL_PATH).fillna("")

texts = []
metadatas = []

for _, row in df.iterrows():
    descripcion = str(row.get("Descrip.Propia", "")).strip()
    if not descripcion:
        continue

    sku = str(row.get("Artículo", "")).strip()
    grupo = str(row.get("Grupo", "")).strip()
    subgrupo = str(row.get("Subgrupo", "")).strip()
    unidad = str(row.get("UE Stock", "")).strip()

    stocks = {}
    total = 0

    for uid, col in UBICACIONES.items():
        if col in row:
            try:
                qty = int(float(row[col]))
            except ValueError:
                qty = 0

            if qty > 0:
                stocks[uid] = qty
                total += qty

    if total == 0:
        continue  # ❌ no indexar productos sin stock

    texts.append(descripcion)

    metadatas.append({
        "sku": sku,
        "descripcion": descripcion,
        "producto": descripcion.lower(),
        "grupo": grupo,
        "subgrupo": subgrupo,
        "unidad": unidad,
        "stock_total": total,
        "ubicaciones": stocks
    })

print(f"🧠 Productos a vectorizar: {len(texts)}")

# =========================
# EMBEDDINGS (BATCH)
# =========================
vectors = []

print("🔄 Generando embeddings...")

for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i:i + BATCH_SIZE]

    emb_response = client.embeddings.create(
        model=MODEL_EMBEDDING,
        input=batch
    )

    for e in emb_response.data:
        vectors.append(e.embedding)

vectors = np.array(vectors, dtype="float32")

print(f"✅ Embeddings generados: {vectors.shape}")

# =========================
# FAISS
# =========================
dim = vectors.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(vectors)

faiss.write_index(index, INDEX_PATH)
np.save(META_PATH, metadatas)

print("✅ Índice FAISS y metadata guardados correctamente")