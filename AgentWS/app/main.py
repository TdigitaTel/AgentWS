# app/main.py
import app.domain_config 
from fastapi import FastAPI
from pydantic import BaseModel
from app.agent import procesar_mensaje
from app.logger import get_logger

logger = get_logger("API")

app = FastAPI()

class Message(BaseModel):
    text: str
    session_id: str = "default"

@app.get("/")
def root():
    return {"status": "Agente activo"}

@app.post("/webhook")
def webhook(msg: Message):
    logger.info(f"Mensaje recibido: {msg.text}")
    return procesar_mensaje(msg.text, msg.session_id)