import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/webhook"

st.set_page_config(page_title="Agente de Stock", page_icon="📦")

st.title("📦 Agente de Stock Inteligente")
st.caption("Consulta productos como en un chat")

# Inicializar historial
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input del usuario
if prompt := st.chat_input("Escribe tu consulta…"):
    # Mostrar mensaje del usuario
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # Llamar a la API
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            response = requests.post(
                API_URL,
                json={"text": prompt},
                timeout=60
            )

            if response.status_code == 200:
                answer = response.json()["respuesta"]
            else:
                answer = "Error al contactar con el agente."

            st.markdown(answer)

    # Guardar respuesta
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )