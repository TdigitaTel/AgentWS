import pandas as pd
import psycopg2

# CONFIGURACIÓN DB
DB_CONFIG = {
    "dbname": "agentws",
    "user": "luisrojas",
    "password": "",
    "host": "localhost",
    "port": "5432"
}

# RUTA DEL EXCEL
EXCEL_PATH = "data/stocks_store.xlsx"  # cambia si es necesario

def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def main():
    # Leer Excel
    df = pd.read_excel(EXCEL_PATH)

    conn = get_connection()
    cur = conn.cursor()

    for _, row in df.iterrows():
        # 1️⃣ Insertar producto
        cur.execute("""
            INSERT INTO products (articulo, descripcion, ref_fabricante, grupo, subgrupo, observacion)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (articulo) DO UPDATE SET
                descripcion = EXCLUDED.descripcion,
                ref_fabricante = EXCLUDED.ref_fabricante,
                grupo = EXCLUDED.grupo,
                subgrupo = EXCLUDED.subgrupo,
                observacion = EXCLUDED.observacion
            RETURNING id;
        """, (
            row["Artículo"],
            row["Descrip.Propia"],
            row["Ref.Fabricante"],
            row["Grupo"],
            row["Subgrupo"],
            row["observacion"]
        ))

        product_id = cur.fetchone()[0]

        # 2️⃣ Insertar stock por tienda
        stores = {
            "Almeiras": row["Stock Almeiras"],
            "Santiago": row["Stock Santiago"],
            "Ferrol": row["Stock Ferrol"],
            "Sandiego": row["Stock Sandiego"],
            "SanXenxo": row["Stock SanXenxo"]
        }

        for store_name, quantity in stores.items():

            # obtener store_id
            cur.execute("SELECT id FROM stores WHERE name = %s", (store_name,))
            store_id = cur.fetchone()[0]

            cur.execute("""
                INSERT INTO stock (product_id, store_id, quantity)
                VALUES (%s, %s, %s)
                ON CONFLICT (product_id, store_id) DO UPDATE SET
                    quantity = EXCLUDED.quantity,
                    updated_at = NOW();
            """, (product_id, store_id, int(quantity)))

    conn.commit()
    cur.close()
    conn.close()

    print("Carga completada correctamente 🚀")


if __name__ == "__main__":
    main()
