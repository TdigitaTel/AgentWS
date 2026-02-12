import numpy as np
from pprint import pprint

# =========================
# CONFIGURACIÓN
# =========================

NPY_PATH = "data/faiss_meta.npy"   # cambia la ruta si aplica


# =========================
# CARGA DEL ARCHIVO
# =========================

try:
    data = np.load(NPY_PATH, allow_pickle=True)
except FileNotFoundError:
    print(f"❌ No se encontró el archivo: {NPY_PATH}")
    exit(1)

print("✅ Archivo cargado correctamente")
print("=" * 50)


# =========================
# INFORMACIÓN GENERAL
# =========================

print("📦 Tipo del objeto cargado:")
print(type(data))

print("\n📊 Número de registros:")
print(len(data))

print("=" * 50)


# =========================
# VER UN REGISTRO COMPLETO
# =========================

if len(data) > 0:
    print("🔍 Primer registro completo:")
    pprint(data[0])
else:
    print("⚠️ El archivo está vacío")
    exit(0)

print("=" * 50)


# =========================
# CAMPOS EXISTENTES
# =========================

campos = set()

for item in data:
    if isinstance(item, dict):
        campos.update(item.keys())

print("🧾 Campos detectados en el .npy:")
for c in sorted(campos):
    print(f"- {c}")

print("=" * 50)


# =========================
# EJEMPLO DE USO REAL
# =========================

print("🧠 Ejemplo de acceso seguro:")

ejemplo = data[0]

ubicacion_id = ejemplo.get("ubicacion_id")
tipo_info = ejemplo.get("tipo_info")
texto = ejemplo.get("texto")

print(f"ubicacion_id: {ubicacion_id}")
print(f"tipo_info: {tipo_info}")
print(f"texto: {texto}")

print("\n✅ Inspección finalizada")