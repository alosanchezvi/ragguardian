FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1

WORKDIR /app

# 1. Dependencias primero (capa cacheada)
COPY requirements-server.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-server.txt

# 2. Solo los archivos de la app (no subas entornos virtuales ni cachés)
COPY nucleo_chat.py app.py ./

# 3. Render inyecta $PORT automáticamente
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --log-level info"]