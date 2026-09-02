# FROM python:3.10-slim

# WORKDIR /app

# COPY requirements.txt .

# RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

# EXPOSE 8000

# CMD [
#     "uvicorn",
#     "app.main:app",
#     "--host",
#     "0.0.0.0",
#     "--port",
#     "8000"
# ]

# ============================================================================
# PRODUCTION DOCKERFILE - Farmer AI Assistant
# Multi-stage build for minimal image size and security
# ============================================================================

# ============================================================================
# STAGE 1: BUILDER
# ============================================================================
FROM python:3.10-slim as builder

WORKDIR /app

# Install system dependencies needed for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ============================================================================
# STAGE 2: RUNTIME (Production Image)
# ============================================================================
FROM python:3.10-slim

# Labels for image metadata
LABEL maintainer="Farmer AI Team"
LABEL version="1.0"
LABEL description="Production Farmer AI RAG System"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    DJANGO_SETTINGS_MODULE=settings

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r farmer && useradd -r -g farmer farmer

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY --chown=farmer:farmer . .

# Create necessary directories
RUN mkdir -p logs data/eval/results data/processed && \
    chown -R farmer:farmer /app

# Switch to non-root user
USER farmer

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--access-log", \
     "--log-level", "info"]


# ============================================================================
# BUILD INSTRUCTIONS
# ============================================================================

# Build image:
# docker build -f Dockerfile.prod -t farmer-ai:latest .

# Run container:
# docker run -p 8000:8000 \
#   -e ENVIRONMENT=production \
#   -v $(pwd)/data:/app/data \
#   -v $(pwd)/logs:/app/logs \
#   farmer-ai:latest

# Push to registry:
# docker tag farmer-ai:latest <registry>/farmer-ai:latest
# docker push <registry>/farmer-ai:latest