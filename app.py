import os
import json
import asyncio
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag import process_document_to_chroma_db, answer_question

# ── 1. CONFIGURACIÓN ───────────────────────────────────────────────
working_dir = os.getcwd()

app = FastAPI(
    title="Guardián del Páramo — RAG API",
    description="API de conocimiento para el widget de Don Frailejón",
    version="1.0.0"
)

# CORS: permite que tu página web (cualquier origen) hable con esta API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # En producción pon tu dominio exacto
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temas disponibles (deben coincidir con los del widget)
TEMAS = [
    {"id": "general",    "label": "Charla libre"},
    {"id": "agua",       "label": "Agua y páramo"},
    {"id": "biodiversidad", "label": "Biodiversidad"},
    {"id": "clima",      "label": "Clima y neblina"},
    {"id": "cultura",    "label": "Cultura campesina"},
    {"id": "conservacion", "label": "Conservación"},
]

# ── 2. MODELOS DE DATOS ────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str          # "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    pregunta: str
    tema: str = "general"
    historial: Optional[List[ChatMessage]] = []

class ChatChunk(BaseModel):
    t: Optional[str] = None          # fragmento de texto (streaming)
    sources: Optional[List[Dict[str, str]]] = None
    done: Optional[bool] = None      # marca de fin (opcional)

# ── 3. ENDPOINTS ───────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "service": "Guardián del Páramo RAG"}


@app.get("/temas")
def get_temas():
    """Devuelve los temas que el widget muestra como botones."""
    return {"temas": TEMAS}


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Recibe la pregunta del usuario y devuelve la respuesta del RAG
    en formato NDJSON (streaming) para que el widget la muestre
    palabra por palabra como si Don Frailejón estuviera escribiendo.
    """

    async def ndjson_stream():
        # ── 3.1 Construir el contexto con historial ─────────────────
        # Si tu función answer_question no usa historial, lo ignoramos;
        # si lo usas, puedes pasarlo como parámetro extra.
        pregunta_completa = req.pregunta
        if req.historial:
            # Opcional: inyectar contexto de la conversación previa
            contexto = "\n".join(
                f"{'Usuario' if m.role == 'user' else 'Asistente'}: {m.content}"
                for m in req.historial[-6:]   # últimos 6 mensajes
            )
            pregunta_completa = f"{contexto}\nUsuario: {req.pregunta}"

        # ── 3.2 Llamar al RAG (bloqueante → lo corremos en thread) ──
        loop = asyncio.get_event_loop()
        try:
            respuesta_raw = await loop.run_in_executor(
                None,
                lambda: answer_question(pregunta_completa)
            )
        except Exception as e:
            yield json.dumps({"t": f"Uy, caminante: la neblina se comió la respuesta. Error: {str(e)}"}) + "\n"
            return

        # ── 3.3 Simular streaming dividiendo la respuesta ───────────
        # Si answer_question ya devuelve un generador, adapta esto.
        texto = str(respuesta_raw)
        chunk_size = 4   # caracteres por chunk (ajusta para más/menos velocidad)

        for i in range(0, len(texto), chunk_size):
            pedazo = texto[i:i+chunk_size]
            yield json.dumps({"t": pedazo}) + "\n"
            await asyncio.sleep(0.02)   # pequeña pausa para el efecto de escritura

        # ── 3.4 Enviar fuentes (opcional) ───────────────────────────
        # Si tu módulo rag te devuelve fuentes, descomenta y adapta:
        # fuentes = [{"title": "Documento base", "ref": "rag_interno"}]
        # yield json.dumps({"sources": fuentes}) + "\n"

        # Marca de finalización (el widget no la usa, pero es buena práctica)
        yield json.dumps({"done": True}) + "\n"

    return StreamingResponse(
        ndjson_stream(),
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no"}   # evita buffering en proxies
    )


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Sube un PDF y lo procesa en la base vectorial (ChromaDB).
    Equivalente al file_uploader de Streamlit.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos PDF")

    save_path = os.path.join(working_dir, file.filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())

    try:
        process_document_to_chroma_db(file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando documento: {str(e)}")

    return {"message": "Documento procesado exitosamente", "filename": file.filename}


# ── 4. ARRANQUE ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))