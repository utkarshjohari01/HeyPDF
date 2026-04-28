# HeyPDF 2.0 — Dockerfile
# Builds the FastAPI backend. Run alongside the React frontend.
#
# Build:   docker build -t heypdf-backend .
# Run:     docker run -p 8000:8000 --env-file .env heypdf-backend

FROM python:3.11-slim

# Prevent .pyc files and enable stdout/stderr flushing
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install OS-level dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cache friendly)
COPY backend/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Pre-download the sentence-transformers model to bake it into the Docker cache.
# This prevents downloading a 90MB file every time Render wakes up from sleep.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy only the backend source
COPY backend/ .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
