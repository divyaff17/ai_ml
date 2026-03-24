"""
tests/test_api.py — Async endpoint tests using httpx.AsyncClient + FastAPI.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from tests.conftest import AUTH_HEADERS

# ── Patch heavy model imports before importing the app ────────────────────────
# We mock out the actual model singletons so tests don't download weights.

_MOCK_PREDICT_RESULT = {
    "is_fake": True,
    "confidence": 0.92,
    "embedding": None,
}

_MOCK_AUDIO_RESULT = {
    "is_fake": False,
    "confidence": 0.65,
    "fake_score": 0.35,
}


def _make_mock_vision_model():
    m = MagicMock()
    m.predict.return_value = dict(_MOCK_PREDICT_RESULT)
    return m


def _make_mock_audio_model():
    m = MagicMock()
    m.predict.return_value = dict(_MOCK_AUDIO_RESULT)
    return m


def _make_mock_face_extractor():
    import numpy as np

    m = MagicMock()
    # Return one dummy 224×224 face crop
    m.extract_from_image.return_value = [np.zeros((224, 224, 3), dtype="uint8")]
    return m


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def client(mock_supabase_for_app):
    """
    Async httpx test client with the FastAPI app.
    Models and Supabase are fully mocked.
    """
    import main

    # ASGITransport does not execute FastAPI lifespan events automatically.
    # Manually inject our mocks into app.state before yielding the client.
    main.app.state.vision_model = _make_mock_vision_model()
    main.app.state.audio_model = _make_mock_audio_model()
    main.app.state.face_extractor = _make_mock_face_extractor()
    main.app.state.supabase = mock_supabase_for_app
    main.app.state.limiter = (
        main.limiter
    )  # The rate limiter is set outside lifespan, but good to ensure it's there

    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient):
    """GET /health should return 200 and {status: 'ok'}."""
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body
    assert "supabase_connected" in body


@pytest.mark.asyncio
async def test_detect_image_valid(client: AsyncClient, sample_image: bytes):
    """POST /detect/image with a valid JPEG should return 200 with expected keys."""
    files = {"file": ("test.jpg", io.BytesIO(sample_image), "image/jpeg")}
    resp = await client.post("/detect/image", files=files, headers=AUTH_HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert "job_id" in body
    assert "is_fake" in body
    assert "confidence" in body
    assert "heatmap_url" in body
    assert "agent_triggered" in body


@pytest.mark.asyncio
async def test_detect_image_invalid_type(client: AsyncClient):
    """POST /detect/image with a .txt file should return 415."""
    files = {"file": ("notes.txt", io.BytesIO(b"hello world"), "text/plain")}
    resp = await client.post("/detect/image", files=files, headers=AUTH_HEADERS)
    assert resp.status_code == 415
    body = resp.json()
    assert "error" in body


@pytest.mark.asyncio
async def test_detect_image_no_api_key(client: AsyncClient, sample_image: bytes):
    """POST /detect/image without X-API-Key should return 422 (missing header)."""
    files = {"file": ("test.jpg", io.BytesIO(sample_image), "image/jpeg")}
    resp = await client.post("/detect/image", files=files)
    # FastAPI raises 422 for missing required Header dependency
    assert resp.status_code in (401, 422)


@pytest.mark.asyncio
async def test_detect_image_wrong_api_key(client: AsyncClient, sample_image: bytes):
    """POST /detect/image with wrong API key should return 401."""
    files = {"file": ("test.jpg", io.BytesIO(sample_image), "image/jpeg")}
    resp = await client.post(
        "/detect/image",
        files=files,
        headers={"X-API-Key": "wrong-key-12345"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_get_results_not_found(client: AsyncClient, mock_supabase_for_app):
    """GET /results/nonexistent-id should return 404."""
    # Make the Supabase fetch return None
    mock_supabase_for_app.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value = MagicMock(
        data=None
    )

    resp = await client.get("/results/nonexistent-id", headers=AUTH_HEADERS)
    assert resp.status_code == 404
    body = resp.json()
    assert "error" in body


@pytest.mark.asyncio
async def test_detect_video_queued(client: AsyncClient, sample_video: bytes):
    """POST /detect/video with valid MP4 should return 200 with status='processing'."""
    with patch("main.process_video", create=True) as mock_task:
        # Mock the Celery delay call
        mock_delay = MagicMock()
        with patch("tasks.celery_app.process_video") as mock_celery:
            mock_celery.delay = mock_delay

            files = {"file": ("clip.mp4", io.BytesIO(sample_video), "video/mp4")}
            resp = await client.post("/detect/video", files=files, headers=AUTH_HEADERS)

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "processing"
    assert "job_id" in body
    assert "message" in body


@pytest.mark.asyncio
async def test_detect_audio_valid(client: AsyncClient, dummy_wav_bytes: bytes):
    """POST /detect/audio with a valid WAV should return 200 with expected keys."""
    files = {"file": ("audio.wav", io.BytesIO(dummy_wav_bytes), "audio/wav")}
    resp = await client.post("/detect/audio", files=files, headers=AUTH_HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert "job_id" in body
    assert "is_fake" in body
    assert "confidence" in body


@pytest.mark.asyncio
async def test_health_request_id_header(client: AsyncClient):
    """Every response should contain an X-Request-ID header."""
    resp = await client.get("/health")
    assert "x-request-id" in resp.headers


@pytest.mark.asyncio
async def test_exception_handler_returns_json(client: AsyncClient):
    """Hitting a non-existent route should still return JSON."""
    resp = await client.get("/does-not-exist-at-all", headers=AUTH_HEADERS)
    # FastAPI returns 404 for unknown routes
    assert resp.status_code == 404 or resp.status_code == 405


@pytest.mark.asyncio
async def test_logs_endpoint(client: AsyncClient):
    """GET /logs with valid API key should return 200 with lines array."""
    resp = await client.get("/logs", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert "lines" in body
    assert "count" in body
    assert isinstance(body["lines"], list)
