import os
from contextlib import asynccontextmanager
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ------------------------------------------------------------------
# 1. ARRANQUE INMEDIATO: nada pesado a nivel global
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # No precargamos motor; solo abrimos el puerto YA para Render
    yield
    # Limpieza futura aquí si la necesita

app = FastAPI(title="Guardián del Páramo", lifespan=lifespan)

# ------------------------------------------------------------------
# 2. HEALTH CHECK — Render lo usa para saber que estás vivo
# ------------------------------------------------------------------
@app.get("/")
@app.get("/health")
async def health():
    return {"status": "alive", "service": "guardian-paramo"}

# ------------------------------------------------------------------
# 3. MODELOS
# ------------------------------------------------------------------
class ChatRequest(BaseModel):
    pregunta: str
    historial: Optional[List[dict]] = None
    filtro_tema: Optional[str] = None

# ------------------------------------------------------------------
# 4. LAZY LOADING: el motor pesado solo se toca cuando llega un usuario
# ------------------------------------------------------------------
_motor_listo = False

def _inicializar_motor():
    """Carga pesada bajo demanda y una sola vez."""
    global _motor_listo
    if _motor_listo:
        return

    # Import diferido: estas librerías consumen RAM y tardan en cargar
    from nucleo_chat import MotorRAG
    MotorRAG.obtener()   # aquí conecta con Qdrant y HuggingFace
    _motor_listo = True

# ------------------------------------------------------------------
# 5. ENDPOINTS DE NEGOCIO
# ------------------------------------------------------------------
@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        _inicializar_motor()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Motor no listo: {exc}")

    from nucleo_chat import responder_stream

    def generador():
        for trozo in responder_stream(
            pregunta=req.pregunta,
            historial=req.historial,
            filtro_tema=req.filtro_tema
        ):
            yield trozo

    return StreamingResponse(
        generador(),
        media_type="application/x-ndjson"
    )

@app.get("/temas")
async def temas():
    try:
        _inicializar_motor()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    from nucleo_chat import listar_temas
    return {"temas": listar_temas()}