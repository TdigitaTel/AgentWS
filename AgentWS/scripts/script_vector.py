import os
import psycopg2
from psycopg2.extras import execute_batch
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "❌ Falta OPENAI_API_KEY"

# ==============================
# CONFIG
# ==============================

DB_CONFIG = {
    "dbname": "agentws",
    "user": "luisrojas",
    "password": "",
    "host": "localhost",
    "port": "5432"
}

EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dims
BATCH_SIZE = 50


client = OpenAI(api_key=OPENAI_API_KEY)
# ==============================
# CONEXIÓN DB
# ==============================

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

# ==============================
# GENERAR EMBEDDING
# ==============================

def generar_embeddings_batch(textos):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=textos
    )
    return [d.embedding for d in response.data]

# ==============================
# CONSTRUIR TEXTO RAG
# ==============================

def construir_texto(product):
    return f"""
Producto: {product['descripcion']}
Referencia fabricante: {product['ref_fabricante']}
Grupo: {product['grupo']}
Subgrupo: {product['subgrupo']}
Observación: {product['observacion']}
""".strip()

# ==============================
# OBTENER PRODUCTOS SIN EMBEDDING
# ==============================

def obtener_productos_sin_embedding(conn):
    query = """
    SELECT p.id,
           p.descripcion,
           p.ref_fabricante,
           p.grupo,
           p.subgrupo,
           p.observacion
    FROM products p
    LEFT JOIN product_embeddings pe
        ON p.id = pe.product_id
    WHERE pe.product_id IS NULL
    """
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()

    columnas = ["id", "descripcion", "ref_fabricante", "grupo", "subgrupo", "observacion"]
    return [dict(zip(columnas, row)) for row in rows]

# ==============================
# INSERTAR EMBEDDINGS
# ==============================

def insertar_embeddings(conn, data):
    query = """
    INSERT INTO product_embeddings (product_id, content, embedding)
    VALUES (%s, %s, %s)
    ON CONFLICT (product_id) DO NOTHING
    """
    with conn.cursor() as cur:
        execute_batch(cur, query, data)
    conn.commit()

# ==============================
# PROCESO PRINCIPAL
# ==============================

def main():
    conn = get_connection()
    productos = obtener_productos_sin_embedding(conn)

    if not productos:
        print("No hay productos nuevos para vectorizar.")
        return

    print(f"Productos a procesar: {len(productos)}")

    for i in tqdm(range(0, len(productos), BATCH_SIZE)):
        batch = productos[i:i+BATCH_SIZE]

        textos = [construir_texto(p) for p in batch]
        embeddings = generar_embeddings_batch(textos)

        data_insert = [
            (p["id"], texto, embedding)
            for p, texto, embedding in zip(batch, textos, embeddings)
        ]

        insertar_embeddings(conn, data_insert)

    conn.close()
    print("Embeddings generados correctamente.")

# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    main()