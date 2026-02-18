import os
import psycopg2
from openai import OpenAI
from dotenv import load_dotenv
from app.logger import get_logger

# ==========================================
# CONFIGURACIÓN
# ==========================================

load_dotenv()
logger = get_logger("PGVECTOR")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DB_CONFIG = {
    "dbname": "agentws",
    "user": "luisrojas",
    "password": "",
    "host": "localhost",
    "port": "5432"
}


# ==========================================
# FUNCIÓN GENERAL DE BÚSQUEDA
# ==========================================

def buscar_similares(texto: str, index_name: str, top_k: int = 5):

    if not texto or not texto.strip():
        logger.warning("PGVECTOR → texto vacío")
        return []

    if index_name not in ["stock", "informacion"]:
        logger.error(f"PGVECTOR → índice no válido: {index_name}")
        return []

    logger.info(f"PGVECTOR → index={index_name}, texto='{texto}', top_k={top_k}")

    # 1️⃣ Generar embedding
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=texto
    ).data[0].embedding

    # 🔥 CAMBIO CLAVE: convertir a string vector
    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # ==========================================
    # PRODUCTOS (RANKING HÍBRIDO)
    # ==========================================

    if index_name == "stock":

        sql = """
        WITH resultados AS (
            SELECT 
                p.id,
                p.descripcion,
                1 - (pe.embedding <=> %s::vector) AS similarity,
                COALESCE(SUM(s.quantity), 0) AS total_stock
            FROM public.product_embeddings pe
            JOIN public.products p ON p.id = pe.product_id
            LEFT JOIN public.stock s ON s.product_id = p.id
            GROUP BY p.id, p.descripcion, pe.embedding
        )
        SELECT *
        FROM resultados
        ORDER BY
            (similarity * 0.7) +
            (LOG(1 + total_stock) * 0.3) DESC
        LIMIT %s;
        """

        cur.execute(sql, (embedding_str, top_k))
        rows = cur.fetchall()

        resultados = []
        for r in rows:
            resultados.append({
                "id": r[0],
                "descripcion": r[1],
                "similarity": float(r[2]),
                "stock_total": int(r[3]),
                "ubicaciones": {}  # puedes completarlo luego si quieres por tienda
            })

    # ==========================================
    # TIENDAS (SOLO SIMILARIDAD)
    # ==========================================

    else:

        sql = """
        SELECT 
            s.id,
            s.name,
            s.address,
            s.phone,
            1 - (se.embedding <=> %s::vector) AS similarity
        FROM public.store_embeddings se
        JOIN public.stores s ON s.id = se.store_id
        ORDER BY se.embedding <=> %s::vector
        LIMIT %s;
        """

        cur.execute(sql, (embedding_str, embedding_str, top_k))
        rows = cur.fetchall()

        resultados = []
        for r in rows:
            resultados.append({
                "id": r[0],
                "nombre": r[1],
                "direccion": r[2],
                "telefono": r[3],
                "similarity": float(r[4])
            })

    cur.close()
    conn.close()

    logger.info(f"PGVECTOR → resultados devueltos ({index_name}): {len(resultados)}")

    return resultados