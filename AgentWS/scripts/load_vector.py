import pandas as pd
import numpy as np
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import faiss
import os

# =========================================================
# CONFIGURACIÓN
# =========================================================

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EXCEL_PATH = Path("data/Stocks.xlsx")
DOMAIN_PATH = Path("domain")

OUTPUT_INDEX = Path("data/faiss.index")
OUTPUT_META = Path("data/faiss_meta.npy")

EMBEDDING_MODEL = "text-embedding-3-small"


# =========================================================
# CARGA DE DOMINIO
# =========================================================

with open(DOMAIN_PATH / "group_subgroup.json", encoding="utf-8") as f:
    GRUPOS = json.load(f)

with open(DOMAIN_PATH / "brands.json", encoding="utf-8") as f:
    MARCAS = [m.lower() for m in json.load(f)]


# =========================================================
# UTILIDADES DE DOMINIO
# =========================================================

def detectar_marca(texto: str):
    if not texto:
        return None

    texto = texto.lower()
    for marca in MARCAS:
        if marca in texto:
            return marca
    return None


def resolver_grupo_subgrupo(texto: str):
    if not texto:
        return None, None

    texto = texto.lower()

    for grupo, subgrupos in GRUPOS.items():
        for sub in subgrupos:
            if sub.lower() in texto:
                return grupo, sub

    return None, None


# =========================================================
# CONSTRUCCIÓN DE TEXTO DE EMBEDDING (CLAVE)
# =========================================================

def construir_texto_embedding(row: dict):
    partes = []

    # Producto base
    if row.get("producto"):
        partes.append(f"Producto: {row['producto']}")

    # Dominio
    if row.get("grupo"):
        partes.append(f"Grupo: {row['grupo']}")

    if row.get("subgrupo"):
        partes.append(f"Subgrupo: {row['subgrupo']}")

    # Atributos técnicos (si existen)
    if row.get("rosca"):
        partes.append(f"Rosca: {row['rosca']}")

    if row.get("diametro"):
        partes.append(f"Diametro: {row['diametro']}")

    if row.get("unidad"):
        partes.append(f"Unidad: {row['unidad']}")

    if row.get("material"):
        partes.append(f"Material: {row['material']}")

    # Marca
    if row.get("marca"):
        partes.append(f"Marca: {row['marca']}")

    return "\n".join(partes)


# =========================================================
# PROCESO PRINCIPAL
# =========================================================

def main():
    print("📦 Cargando Excel...")
    df = pd.read_excel(EXCEL_PATH)

    textos_embedding = []
    metadatos = []

    print("🧠 Procesando filas...")

    for _, row in df.iterrows():
        articulo = str(row.get("Artículo", "")).strip()
        descripcion = str(row.get("Descrip.Propia", "")).strip()

        texto_completo = f"{articulo} {descripcion}"

        # Señales básicas
        producto = articulo.lower() if articulo else None

        # Dominio
        grupo, subgrupo = resolver_grupo_subgrupo(texto_completo)
        marca = detectar_marca(texto_completo)

        # Construir estructura base
        data = {
            "producto": producto,
            "grupo": grupo,
            "subgrupo": subgrupo,
            "marca": marca,
            "diametro": None,
            "unidad": None,
            "rosca": None,
            "material": None
        }

        texto_embedding = construir_texto_embedding(data)

        # Si no hay texto útil, saltar
        if not texto_embedding.strip():
            continue

        textos_embedding.append(texto_embedding)

        metadatos.append({
            "articulo": articulo,
            "grupo": grupo,
            "subgrupo": subgrupo,
            "marca": marca,
            "stock": row.get("Stock Total"),
            "ubicacion": row.get("Ubicación")
        })

    print(f"🔢 Total textos vectorizados: {len(textos_embedding)}")

    # =========================================================
    # GENERAR EMBEDDINGS
    # =========================================================

    print("🚀 Generando embeddings con OpenAI...")

    embeddings = []
    batch_size = 100

    for i in range(0, len(textos_embedding), batch_size):
        batch = textos_embedding[i:i + batch_size]

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch
        )

        for e in response.data:
            embeddings.append(e.embedding)

    embeddings = np.array(embeddings).astype("float32")

    # =========================================================
    # CREAR ÍNDICE FAISS
    # =========================================================

    print("📐 Creando índice FAISS...")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, str(OUTPUT_INDEX))
    np.save(OUTPUT_META, np.array(metadatos, dtype=object))

    print("✅ Vectorización completada correctamente")
    print(f"📁 Índice FAISS: {OUTPUT_INDEX}")
    print(f"📁 Metadatos: {OUTPUT_META}")


# =========================================================
# EJECUCIÓN
# =========================================================

if __name__ == "__main__":
    main()