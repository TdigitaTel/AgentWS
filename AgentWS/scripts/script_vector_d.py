import os
import psycopg2
from psycopg2.extras import execute_batch
from openai import OpenAI
from dotenv import load_dotenv

# ==========================================
# CONFIG
# ==========================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "❌ Falta OPENAI_API_KEY"

DB_CONFIG = {
    "dbname": "agentws",
    "user": "luisrojas",
    "password": "",
    "host": "localhost",
    "port": "5432"
}

EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 10

client = OpenAI(api_key=OPENAI_API_KEY)

# ==========================================
# CONEXIÓN
# ==========================================

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

# ==========================================
# OBTENER TIENDAS
# ==========================================

def obtener_tiendas(conn):
    query = """
    SELECT id, name, address, city, phone, email, opening_hours
    FROM public.stores;
    """
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()

    columnas = ["id", "name", "address", "city", "phone", "email", "opening_hours"]
    return [dict(zip(columnas, row)) for row in rows]

# ==========================================
# CONSTRUIR TEXTO
# ==========================================

def construir_texto(t):
    return f"""
Tienda: {t['name']}
Dirección: {t['address']}
Ciudad: {t['city']}
Teléfono: {t['phone']}
Email: {t['email']}
Horario: {t['opening_hours']}
""".strip()

# ==========================================
# GENERAR EMBEDDINGS
# ==========================================

def generar_embeddings(textos):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=textos
    )
    return [item.embedding for item in response.data]

# ==========================================
# INSERTAR / ACTUALIZAR
# ==========================================

def insertar_embeddings(conn, data):
    query = """
    INSERT INTO public.store_embeddings (store_id, content, embedding)
    VALUES (%s, %s, %s)
    ON CONFLICT (store_id)
    DO UPDATE SET
        content = EXCLUDED.content,
        embedding = EXCLUDED.embedding;
    """
    with conn.cursor() as cur:
        execute_batch(cur, query, data)

    conn.commit()

# ==========================================
# MAIN
# ==========================================

def main():
    conn = get_connection()

    tiendas = obtener_tiendas(conn)

    if not tiendas:
        print("No hay tiendas para vectorizar.")
        conn.close()
        return

    print(f"Vectorizando {len(tiendas)} tiendas...")

    for i in range(0, len(tiendas), BATCH_SIZE):
        batch = tiendas[i:i+BATCH_SIZE]

        textos = [construir_texto(t) for t in batch]
        embeddings = generar_embeddings(textos)

        data_insert = [
            (t["id"], texto, embedding)
            for t, texto, embedding in zip(batch, textos, embeddings)
        ]

        insertar_embeddings(conn, data_insert)

        print(f"Batch {i//BATCH_SIZE + 1} procesado")

    conn.close()
    print("Embeddings actualizados correctamente 🚀")

if __name__ == "__main__":
    main()