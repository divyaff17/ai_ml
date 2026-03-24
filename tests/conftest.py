"""
tests/conftest.py — Shared pytest fixtures and environment bootstrap.

This file is auto-loaded by pytest before any tests run.
It sets all required environment variables so config.py doesn't raise
EnvironmentError when imported during test collection.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Add repo root to sys.path so all imports resolve ─────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Inject test env vars BEFORE config.py is imported ────────────────────────
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "test-supabase-key")
os.environ.setdefault("TAVILY_API_KEY", "tvly-test-key")
os.environ.setdefault("API_SECRET_KEY", "test-api-secret")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("DEEPFAKE_THRESHOLD", "0.5")
os.environ.setdefault("LOG_LEVEL", "WARNING")

# ── Valid API key header for httpx test client ────────────────────────────────
# Import config to get the actual value the middleware checks against
import config as _cfg

API_KEY = _cfg.API_SECRET_KEY
AUTH_HEADERS = {"X-API-Key": API_KEY}


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def sample_image() -> bytes:
    """
    Generate a random 224×224 RGB JPEG in-memory.
    Reused across test_api, test_model.
    """
    from PIL import Image as PILImage

    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = PILImage.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@pytest.fixture(scope="session")
def dummy_jpeg_bytes() -> bytes:
    """Alias kept for backward compat — delegates to a minimal 32×32 JPEG."""
    from PIL import Image as PILImage

    arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    img = PILImage.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@pytest.fixture(scope="session")
def sample_video() -> bytes:
    """
    Generate a 1-second blank MP4 (10 frames at 10 fps, 64×64) using OpenCV.
    Returns the raw MP4 bytes.
    """
    import cv2

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, 10.0, (64, 64))
        black_frame = np.zeros((64, 64, 3), dtype=np.uint8)
        for _ in range(10):
            writer.write(black_frame)
        writer.release()
        result_bytes = Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return result_bytes


@pytest.fixture(scope="session")
def dummy_wav_bytes() -> bytes:
    """Return a minimal 1-second 16 kHz mono WAV file as bytes."""
    buf = io.BytesIO()
    n_samples = 16_000
    samples = np.random.randn(n_samples).astype(np.float32)
    # Convert to int16 for WAV
    int_samples = (samples * 32767).astype(np.int16)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16_000)
        wf.writeframes(int_samples.tobytes())
    buf.seek(0)
    return buf.read()


@pytest.fixture
def mock_supabase_client() -> MagicMock:
    """Return a fully-mocked Supabase client for DB tests."""
    execute_mock = MagicMock(return_value=MagicMock(data={"id": "test-uuid"}))
    chain = MagicMock()
    for m in ("select", "insert", "update", "eq", "maybe_single", "range", "order"):
        getattr(chain, m).return_value = chain
    chain.execute = execute_mock
    client = MagicMock()
    client.table.return_value = chain
    return client


@pytest.fixture
def mock_supabase_for_app(mock_supabase_client):
    """
    Patch the Supabase client used by utils/supabase_utils.py so all
    retry_insert / retry_fetch calls hit the mock.
    """
    with patch("utils.supabase_utils._get_client", return_value=mock_supabase_client):
        yield mock_supabase_client
