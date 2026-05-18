# ==============================================================================
# Worker Dockerfile - GPU-accelerated PDF processing with Dolphin VLM
# Expected size: ~4.5GB (includes PyTorch, Transformers, OpenCV, Dolphin model)
# Requires: NVIDIA GPU with CUDA 11.8+
# ==============================================================================

FROM nvidia/cuda:13.2.1-runtime-ubuntu22.04@sha256:f48aa5c157006835b67f43e77bea3dfe32a7d9a46eeb2de3f8af1b5a464d0334

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    git-lfs \
    curl \
    ca-certificates \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy workspace configuration
COPY pyproject.toml uv.lock README.md ./

# Copy workspace packages
COPY packages/shared/ ./packages/shared/
COPY packages/worker/ ./packages/worker/

# Install worker dependencies (includes GPU packages: torch, transformers, etc.)
# This installs: rag_pipeline[pdf,vector,cloud,cli] + all GPU dependencies
RUN uv sync --frozen --no-dev --package worker

# Download Dolphin model (~2GB) from HuggingFace
# Note: Model is downloaded to rag_pipeline/pdf_parsing/models/
RUN git lfs install && \
    mkdir -p /app/packages/shared/rag_pipeline/pdf_parsing/models && \
    git clone https://huggingface.co/ByteDance/Dolphin-1.5 /tmp/dolphin && \
    cd /tmp/dolphin && \
    git lfs pull && \
    mv /tmp/dolphin/* /app/packages/shared/rag_pipeline/pdf_parsing/models/ && \
    rm -rf /tmp/dolphin

# Set Python path to find packages
ENV PYTHONPATH="/app/packages/shared:/app/packages/worker:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Default environment variables for distributed worker
ENV WORKER_ID=0
ENV TOTAL_WORKERS=1
ENV S3_INPUT_PREFIX=raw_pdfs/
ENV S3_OUTPUT_PREFIX=processed/
ENV MAX_RETRIES=2
ENV CONCURRENT_PDFS=3

# GPU environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Default command - run distributed worker
CMD ["python", "packages/worker/worker/distributed_worker.py"]
