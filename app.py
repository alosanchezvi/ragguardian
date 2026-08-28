"""
API del Guardián del Páramo — endpoints:
  POST /chat    → streaming NDJSON ({"t": "trozo"} por línea y {"sources": [...]})
  GET  /temas   → temas disponibles
  GET  /health  → ping anti-sleep
"""
import os
import json
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import nucleo_chat as core

app = FastAPI(title="Guardián del Páramo API", version="1.0")

# Permitir todos los orígenes ("*") es la opción más segura 
# para evitar bloqueos de CORS desde Cloudflare Workers/Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class PeticionChat(BaseModel):
    pregunta: str = Field(min_length=2, max_length=500)
    tema: Optional[str] = None
    historial: list = Field(default_factory=list, max_length=12)


@app.on_event("startup")
def arrancar():
    core.MotorRAG.obtener()          # carga Qdrant + embeddings UNA sola vez


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/temas")
def temas():
    return {"temas": core.listar_temas()}


@app.post("/chat")
def chat(p: PeticionChat):
    def generador():
        # core.responder_stream YA entrega el string formateado como NDJSON
        # ("{"t": "texto"}\n") por lo que solo debemos hacer yield directamente.
        for linea_ndjson in core.responder_stream(p.pregunta, p.historial, p.tema):
            yield linea_ndjson

    return StreamingResponse(generador(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn
    # Render asigna dinámicamente el puerto a través de la variable de entorno $PORT
    puerto = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=puerto)