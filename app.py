from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional
import os
import json
import nucleo_chat as core

app = FastAPI(title="Guardián del Páramo RAG")

# Montar carpetas estáticas y templates si los usas
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates") if os.path.exists("templates") else None

@app.on_event("startup")
def arrancar():
    # Inicializa Qdrant una sola vez al arrancar de forma controlada
    try:
        core.MotorRAG.obtener()
    except Exception as e:
        print(f"Aviso en startup: {e}")

class PreguntaRequest(BaseModel):
    pregunta: str
    historial: Optional[List[dict]] = None
    filtro_tema: Optional[str] = None

@app.post("/api/chat")
def chat(body: PreguntaRequest):
    return StreamingResponse(
        core.responder_stream(body.pregunta, body.historial, body.filtro_tema),
        media_type="text/event-stream"
    )

@app.get("/api/temas")
def temas():
    try:
        return {"temas": core.listar_temas()}
    except Exception:
        return {"temas": []}

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    if templates and os.path.exists("templates/index.html"):
        return templates.TemplateResponse("index.html", {"request": request})
    return HTMLResponse("<h3>Servidor del Guardián del Páramo activo correctamente.</h3>")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)