# ==============================================================================
# API Dockerfile - Lightweight FastAPI service (NO GPU dependencies)
# Expected size: ~800MB (vs 4.5GB with GPU deps)
# ==============================================================================

# Multi-stage build for minimal final image
FROM python:3.14-slim@sha256:7a500125bc50693f2214e842a621440a1b1b9cbb2188f74ab045d29ed2ea5856 AS builder

# Install UV for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest@sha256:1025398289b62de8269e70c45b91ffa37c373f38118d7da036fb8bb8efc85d97 /uv /usr/local/bin/uv

WORKDIR /app

# Copy workspace configuration
COPY pyproject.toml uv.lock README.md ./

# Copy all workspace packages (needed for uv to resolve dependencies)
COPY packages/shared/ ./packages/shared/
COPY packages/api/ ./packages/api/

# Install ONLY API dependencies (no GPU, no PDF processing)
# This installs: rag_pipeline[vector,cloud] + FastAPI + LiteLLM
# Does NOT install: torch, transformers, opencv (saves 2GB+)
RUN uv sync --frozen --no-dev --package api

# ==============================================================================
# Production stage - Minimal runtime image
# ==============================================================================
FROM python:3.14-slim@sha256:7a500125bc50693f2214e842a621440a1b1b9cbb2188f74ab045d29ed2ea5856

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy workspace packages
COPY packages/shared/rag_pipeline/ ./rag_pipeline/
COPY packages/api/api/ ./api/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
