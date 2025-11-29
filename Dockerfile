# Tender Documents Handler MCP - Docker Image
# ============================================

FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TENDER_UPLOAD_DIR=/app/uploads \
    TENDER_PROCESSED_DIR=/app/processed \
    TENDER_CACHE_DIR=/app/cache

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Document processing
    pandoc \
    poppler-utils \
    libreoffice-writer \
    # OCR
    tesseract-ocr \
    tesseract-ocr-eng \
    # Build tools
    build-essential \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create app directories
WORKDIR /app
RUN mkdir -p /app/uploads /app/processed /app/cache

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY tender_docs_mcp.py .
COPY __init__.py .

# Create non-root user
RUN useradd -m -u 1000 tender && \
    chown -R tender:tender /app

USER tender

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import tender_docs_mcp; print('OK')" || exit 1

# Default command - run the MCP server
CMD ["python", "-m", "tender_docs_mcp"]
