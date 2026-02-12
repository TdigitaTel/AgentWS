import os
import numpy as np
import pandas as pd
import faiss
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Cargar Excel
df = pd.read_excel("data/Stocks.xlsx")

# Texto a vectorizar
textos = []
metadata = []
print("Iniciando proceso de carga")
for _, row in df.iterrows():
    texto = f"{row['Descrip.Propia']} {row['Grupo']} {row['Subgrupo']}"
    textos.append(texto)
    metadata.append({
        "articulo": row["Artículo"],
        "descripcion": row["Descrip.Propia"],
        "grupo": row["Grupo"],
        "subgrupo": row["Subgrupo"]
    })
    print("Registro: ",{texto})

# Crear embeddings
print("Iniciando proceso de embedding")
embeddings = []
for texto in textos:
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=texto
    ).data[0].embedding
    embeddings.append(emb)
    print("Registro: ",{texto})
embeddings = np.array(embeddings).astype("float32")

# Crear índice FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Guardar índice
faiss.write_index(index, "data/faiss.index")

# Guardar metadata
np.save("data/faiss_meta.npy", metadata)

print("Índice FAISS creado correctamente")