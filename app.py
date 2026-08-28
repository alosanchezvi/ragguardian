import os
import json
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ── IMPORTAMOS TU NÚCLEO SIN TOCARLO ──────────────────────────────
from nucleo_chat import responder_stream, listar_temas

# ── 1. CONFIGURACIÓN ───────────────────────────────────────────────
app = FastAPI(
    title="Guardián del Páramo — RAG API",
    description="API pública para el widget de Don Frailejón",
    version="1.0.0"
)

# CORS: permite que cualquier página web (incluida la tuya) llame a esta API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # En producción pon tu dominio exacto
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 2. MODELOS ─────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    pregunta: str
    tema: Optional[str] = None     # El widget envía esto; tu núcleo lo usa como filtro_tema
    historial: Optional[List[ChatMessage]] = []

# ── 3. ENDPOINTS ───────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "service": "Guardián del Páramo"}


@app.get("/temas")
def get_temas():
    """
    El widget pide los temas al cargar la página.
    Usa tu función listar_temas() que lee directo de Qdrant.
    """
    try:
        temas_crudos = listar_temas()
        # Mapeamos a lo que espera el widget: {id, label}
        temas = [
            {"id": t, "label": t.replace("_", " ").capitalize()}
            for t in temas_crudos
        ]
        # Si no hay temas en Qdrant, devolvemos uno por defecto para que no se rompa
        if not temas:
            temas = [{"id": "general", "label": "Charla libre"}]
    except Exception:
        # Si Qdrant falla, el widget igual carga
        temas = [{"id": "general", "label": "Charla libre"}]
    return {"temas": temas}


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Recibe la pregunta del usuario y devuelve la respuesta del RAG
    en NDJSON streaming. El widget la lee palabra por palabra.
    """
    # Convertimos el historial del widget al formato que espera tu núcleo
    historial_dicts = None
    if req.historial:
        historial_dicts = [
            {"role": m.role, "content": m.content}
            for m in req.historial
        ]

    # Llamamos directamente a tu generador responder_stream
    generador = responder_stream(
        pregunta=req.pregunta,
        historial=historial_dicts,
        filtro_tema=req.tema
    )

    return StreamingResponse(
        generador,
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no"}   # evita buffering en proxies
    )


# ── 4. ARRANQUE ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))