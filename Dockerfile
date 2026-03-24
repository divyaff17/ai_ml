FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps with CPU-only PyTorch (saves ~2 GB)
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt \
    && find /usr/local/lib/python3.11 -name "*.pyc" -delete \
    && find /usr/local/lib/python3.11 -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Copy project files
COPY . .

# Remove unnecessary files from image
RUN rm -rf tests/ .github/ .vscode/ .pytest_cache/ .ruff_cache/ __pycache__/ \
    .git/ .env .env.template .pre-commit-config.yaml \
    audit_scripts_1_to_8.py debug_test.py verify_imports.py \
    pip_freeze_backup.txt to_uninstall.txt docker-compose.yml \
    training/ evaluation/ scripts/

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
