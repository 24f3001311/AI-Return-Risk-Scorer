# ============================================
# AI Return-Risk Scorer — Dockerfile
# Multi-stage build: Python backend + Vue build
# ============================================

# Stage 1: Build Vue.js frontend
FROM node:20-alpine AS frontend-build

WORKDIR /app/frontend

# Copy package files and install dependencies
COPY src/frontend/package*.json ./
RUN npm ci --production=false

# Copy frontend source and build
COPY src/frontend/ ./
RUN npm run build

# Stage 2: Python backend + serve built frontend
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ ./src/
COPY data/reference/ ./data/reference/

# Copy built frontend from Stage 1
COPY --from=frontend-build /app/frontend/dist ./src/frontend/dist

# Create models directory
RUN mkdir -p src/api/models

# Copy model artifacts (if pre-built)
COPY src/api/model[s]/ ./src/api/models/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

# Start the API server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
