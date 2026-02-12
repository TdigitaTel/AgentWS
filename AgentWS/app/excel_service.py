import pandas as pd
import os
from app.logger import get_logger

logger = get_logger("EXCEL")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCEL_PATH = os.path.join(BASE_DIR, "data", "Stocks.xlsx")

df = pd.read_excel(EXCEL_PATH)
df.columns = df.columns.str.strip()

def buscar_producto_por_articulo(articulo):
    logger.info(f"Buscando producto en Excel: {articulo}")

    fila = df[df["Artículo"] == articulo]

    if fila.empty:
        logger.warning("Producto no encontrado en Excel")
        return None

    f = fila.iloc[0]

    producto = {
        "articulo": f["Artículo"],
        "descripcion": f["Descrip.Propia"],
        "stock": int(f["Stock Total"]),
        "ubicacion": f["Ubicación"]
    }

    logger.info(f"Producto encontrado: {producto}")
    return producto