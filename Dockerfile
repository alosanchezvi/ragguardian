FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

# Copiar solo el código (ya NO copiamos qdrant_db porque usamos la nube)
COPY nucleo_chat.py app.py ./

# Ejecutar Uvicorn usando el puerto que asigne Render dinámicamente
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
