FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps with CPU-only PyTorch (saves ~2 GB)
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Create checkpoints directory
RUN mkdir -p /app/models/checkpoints

# Copy project files
COPY . .

# Remove unnecessary files from image
RUN rm -rf tests/ .github/ .vscode/ .pytest_cache/ .ruff_cache/ __pycache__/ \
    .git/ .env .env.template .pre-commit-config.yaml \
    audit_scripts_1_to_8.py debug_test.py verify_imports.py \
    pip_freeze_backup.txt to_uninstall.txt docker-compose.yml \
    training/ evaluation/ scripts/

# HF model cache dir (downloaded on first startup, cached by Railway across deploys)
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Single worker + longer timeout to survive model loading on first startup
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "75"]
