import os
import json
import numpy as np
import pandas as pd
import faiss
from dotenv import load_dotenv
from openai import OpenAI

# =========================
# CONFIGURACIÓN
# =========================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "❌ Falta OPENAI_API_KEY en el entorno"

client = OpenAI(api_key=OPENAI_API_KEY)

EXCEL_PATH = "data/stocks_store.xlsx"

FAISS_INDEX_PATH = "data/faiss_stock.index"
FAISS_META_PATH = "data/faiss_stock_meta.npy"

EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 200  # 👈 CRÍTICO (no subir sin motivo)

# =========================
# CARGA DE DATOS
# =========================

print("📥 Cargando Excel...")
df = pd.read_excel(EXCEL_PATH)

df.columns = [c.strip() for c in df.columns]

# =========================
# NORMALIZACIÓN / LIMPIEZA
# =========================

def safe_str(value):
    if pd.isna(value):
        return ""
    return str(value).strip()

texts = []
metadata = []

for _, row in df.iterrows():
    descripcion = safe_str(row.get("Descrip.Propia"))
    grupo = safe_str(row.get("Grupo"))
    subgrupo = safe_str(row.get("Subgrupo"))
    observacion = safe_str(row.get("observacion"))

    texto_embedding = " | ".join(
        t for t in [
            descripcion,
            grupo,
            subgrupo,
            observacion
        ] if t
    )

    if not texto_embedding:
        continue

    texts.append(texto_embedding)

    meta = {
        "sku": safe_str(row.get("Artículo")),
        "descripcion": descripcion,
        "referencia": safe_str(row.get("Ref.Fabricante")),
        "unidad": safe_str(row.get("UE Stock")),
        "grupo": grupo,
        "subgrupo": subgrupo,
        "stock_total": float(row.get("Stock Total", 0)),

        # Stock por tienda
        "stock_almeiras": float(row.get("Stock Almeiras", 0)),
        "stock_santiago": float(row.get("Stock Santiago", 0)),
        "stock_ferrol": float(row.get("Stock Ferrol", 0)),
        "stock_sandiego": float(row.get("Stock Sandiego", 0)),
        "stock_sanxenxo": float(row.get("Stock SanXenxo", 0)),
    }

    metadata.append(meta)

print(f"🧠 Textos válidos: {len(texts)}")
assert len(texts) == len(metadata), "❌ Desfase textos / metadata"

# =========================
# GENERACIÓN DE EMBEDDINGS (BATCHED)
# =========================

print("🔄 Generando embeddings por lotes...")

all_embeddings = []

for i in range(0, len(texts), BATCH_SIZE):
    batch_texts = texts[i:i + BATCH_SIZE]
    print(f"➡️ Embedding batch {i} – {i + len(batch_texts)}")

    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=batch_texts
    )

    batch_embeddings = [e.embedding for e in response.data]
    all_embeddings.extend(batch_embeddings)

embeddings = np.array(all_embeddings, dtype="float32")

print(f"✅ Embeddings generados: {embeddings.shape}")

assert embeddings.shape[0] == len(metadata), "❌ Embeddings ≠ metadata"

# =========================
# CREACIÓN ÍNDICE FAISS
# =========================

print("📦 Creando índice FAISS...")

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

faiss.write_index(index, FAISS_INDEX_PATH)
np.save(FAISS_META_PATH, np.array(metadata, dtype=object))

print("🎉 ÍNDICE FAISS GENERADO CORRECTAMENTE")
print(f"➡️ Index: {FAISS_INDEX_PATH}")
print(f"➡️ Metadata: {FAISS_META_PATH}")