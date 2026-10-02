# Dockerfile — RAG_Chatbot_V2
# Base: Python 3.11 slim
FROM python:3.11-slim

# System dependencies:
#   libpq-dev      : PostgreSQL client (psycopg-binary)
#   build-essential: compile some Python packages
#   libgl1, libglib2.0-0: PyMuPDF / OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    postgresql-client \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker layer cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browser for crawl4ai web scraping
RUN playwright install chromium --with-deps

# Copy application source
COPY . .
RUN chmod +x /app/scripts/entrypoint.sh

# Pre-create runtime directories (FileProcessor also calls os.makedirs, but be safe)
RUN mkdir -p \
    data/upload/pdf \
    data/upload/docx \
    data/upload/text \
    data/indexes/pdf \
    data/indexes/docx \
    data/indexes/text \
    data/indexes/unstructured \
    government_data/global_faiss_index \
    downloads \
    logs

EXPOSE 8000

# Default: API server. Overridden per-service in docker-compose.yaml
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
