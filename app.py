"""
API del Guardián del Páramo — endpoints:
  POST /chat    → streaming NDJSON ({\"t\": \"trozo\"} por línea)
  GET  /temas   → temas disponibles
  GET  /health  → ping anti-sleep
"""
import json
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import nucleo_chat as core

app = FastAPI(title="Guardián del Páramo API", version="1.0")

# ⚠️ ACTUALIZA estos orígenes cuando tengas las URLs reales
ORIGENES_PERMITIDOS = [
    "https://TU-USUARIO.github.io",
    "https://TU-USUARIO.github.io/TU-REPO",
    "http://localhost:8000",       # pruebas locales del widget
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGENES_PERMITIDOS,
    allow_methods=["GET", "POST"],
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
        for trozo in core.responder_stream(p.pregunta, p.historial, p.tema):
            yield json.dumps({"t": trozo}, ensure_ascii=False) + "\n"

    return StreamingResponse(generador(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)