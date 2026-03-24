import uuid
import secrets
from contextlib import asynccontextmanager
from typing import Optional
import time

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from loguru import logger
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from config import settings
from models.vision_model import (
    DeepfakeVisionModel,
    FaceExtractor,
    fuse_scores,
    temporal_score,
)
from models.audio_model import DeepfakeAudioModel
from agents.investigation_agent import run_investigation
from tasks.celery_app import process_video
from middleware.auth import APIKeyMiddleware
from utils.supabase_utils import retry_insert, retry_fetch
from supabase import create_client

# ── Allowed file types & size limits ──────────────────────────────────────────
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
_AUDIO_EXTS = {".mp3", ".wav", ".flac"}

from collections import deque

_LOG_RING: deque[str] = deque(maxlen=500)


def _ring_buffer_sink(message) -> None:
    _LOG_RING.append(str(message).rstrip("\n"))


logger.add(_ring_buffer_sink, level=settings.LOG_LEVEL, format="{message}")


def _load_vision_model() -> DeepfakeVisionModel:
    """Load the vision model in a thread so it doesn't block the event loop."""
    import os

    model = DeepfakeVisionModel()
    if os.environ.get("APP_ENV") != "test":
        try:
            model.load_checkpoint(settings.MODEL_CHECKPOINT)
        except Exception as e:
            logger.warning(f"Could not load vision checkpoint: {e}")
    return model


def _load_audio_model() -> DeepfakeAudioModel:
    """Load the audio model in a thread so it doesn't block the event loop."""
    import os

    model = DeepfakeAudioModel()
    if os.environ.get("APP_ENV") != "test":
        try:
            model.load_checkpoint(settings.AUDIO_CHECKPOINT)
        except Exception as e:
            logger.warning(f"Could not load audio checkpoint: {e}")
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    logger.info("Loading models...")
    loop = asyncio.get_event_loop()

    with ThreadPoolExecutor() as pool:
        # Load models in threads so the event loop stays alive
        # and uvicorn doesn't kill the worker for being unresponsive
        vision_future = loop.run_in_executor(pool, _load_vision_model)
        audio_future = loop.run_in_executor(pool, _load_audio_model)

        app.state.vision_model = await vision_future
        app.state.audio_model = await audio_future

    app.state.face_extractor = FaceExtractor()
    app.state.supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    logger.info("All models loaded. API ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Deepfake Detector API", lifespan=lifespan)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(APIKeyMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:7860", "https://*.hf.space"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    request_id = (
        request.state.request_id if hasattr(request.state, "request_id") else "unknown"
    )
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation error",
            "status_code": 422,
            "request_id": request_id,
            "detail": exc.errors(),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    request_id = (
        request.state.request_id if hasattr(request.state, "request_id") else "unknown"
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": str(exc.detail),
            "status_code": exc.status_code,
            "request_id": request_id,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = (
        request.state.request_id if hasattr(request.state, "request_id") else "unknown"
    )
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "request_id": request_id,
        },
    )


def _validate_file_ext(filename: str, allowed_exts: set[str]) -> str:
    import os

    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(allowed_exts)}",
        )
    return ext


async def _read_file_bytes(file: UploadFile, max_bytes: int, label: str) -> bytes:
    data = await file.read()
    assert isinstance(data, bytes)
    if len(data) > max_bytes:
        mb = max_bytes / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"{label.capitalize()} file too large. Maximum is {mb:.0f} MB.",
        )
    return data


@app.post("/detect/image")
@limiter.limit("10/minute")
async def detect_image(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> JSONResponse:
    _validate_file_ext(file.filename, _IMAGE_EXTS)
    image_bytes = await _read_file_bytes(
        file, settings.MAX_IMAGE_MB * 1024 * 1024, "image"
    )
    job_id = str(uuid.uuid4())

    model = request.app.state.vision_model
    extractor = request.app.state.face_extractor
    sb = request.app.state.supabase

    faces = extractor.extract_from_image(image_bytes)

    result = model.predict(image_bytes)
    if result.get("error"):
        return JSONResponse(
            status_code=200,
            content={
                "job_id": job_id,
                "is_fake": None,
                "confidence": 0.0,
                "error": result["error"],
            },
        )

    is_fake = result["is_fake"]
    confidence = result["confidence"]
    heatmap_url = None

    if confidence > settings.AGENT_TRIGGER_THRESHOLD and is_fake:
        background_tasks.add_task(run_investigation, job_id, {"confidence": result["confidence"], "media_type": "image"})

    return JSONResponse(
        status_code=200,
        content={
            "job_id": job_id,
            "is_fake": is_fake,
            "confidence": confidence,
            "heatmap_url": heatmap_url,
            "agent_triggered": confidence > settings.AGENT_TRIGGER_THRESHOLD
            and is_fake,
        },
    )


@app.post("/detect/video")
@limiter.limit("5/minute")
async def detect_video(request: Request, file: UploadFile = File(...)) -> JSONResponse:
    ext = _validate_file_ext(file.filename, _VIDEO_EXTS)
    video_bytes = await _read_file_bytes(
        file, settings.MAX_VIDEO_MB * 1024 * 1024, "video"
    )
    job_id = str(uuid.uuid4())
    media_url = f"http://dummy.url/{job_id}.mp4"

    try:
        process_video.delay(job_id, media_url)
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "job_id": job_id,
                "status": "error",
                "message": "Failed to queue processing task.",
            },
        )

    return JSONResponse(
        status_code=200,
        content={
            "job_id": job_id,
            "status": "processing",
            "message": f"Poll /results/{job_id}",
        },
    )


@app.post("/detect/audio")
@limiter.limit("10/minute")
async def detect_audio(request: Request, file: UploadFile = File(...)) -> JSONResponse:
    _validate_file_ext(file.filename, _AUDIO_EXTS)
    audio_bytes = await _read_file_bytes(
        file, settings.MAX_AUDIO_MB * 1024 * 1024, "audio"
    )
    job_id = str(uuid.uuid4())

    model = request.app.state.audio_model
    result = model.predict(audio_bytes)

    return JSONResponse(
        status_code=200,
        content={
            "job_id": job_id,
            "is_fake": result["is_fake"],
            "confidence": result["confidence"],
        },
    )


@app.get("/results/{job_id}")
async def get_results(request: Request, job_id: str) -> JSONResponse:
    sb = request.app.state.supabase
    # Fetch both detections row AND agent_logs row
    detection_res = (
        sb.table("detections").select("*").eq("job_id", job_id).maybe_single().execute()
    )

    if not detection_res.data:
        raise HTTPException(
            status_code=404, detail=f"No detection found for job_id '{job_id}'."
        )

    row = detection_res.data
    status_val = row.get("status", "unknown")

    if status_val == "pending" or status_val == "processing":
        return JSONResponse(
            status_code=200, content={"job_id": job_id, "status": status_val}
        )

    result = {
        "job_id": row.get("job_id"),
        "status": status_val,
        "media_type": row.get("media_type"),
        "is_fake": row.get("is_fake"),
        "confidence": row.get("confidence"),
        "heatmap_url": row.get("heatmap_url"),
        "agent_triggered": row.get("agent_triggered", False),
        "created_at": row.get("created_at"),
    }

    if result["agent_triggered"]:
        agent_res = sb.table("agent_logs").select("*").eq("job_id", job_id).execute()
        if agent_res.data:
            result["agent_logs"] = agent_res.data

    return JSONResponse(status_code=200, content=result)


@app.get("/health")
async def health_check(request: Request) -> JSONResponse:
    model_loaded = (
        hasattr(request.app.state, "vision_model")
        and request.app.state.vision_model is not None
    )
    supabase_connected = False

    try:
        sb = request.app.state.supabase
        sb.table("detections").select("id").limit(1).execute()
        supabase_connected = True
    except Exception:
        pass

    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "model_loaded": model_loaded,
            "supabase_connected": supabase_connected,
        },
    )


@app.get("/logs")
async def get_logs() -> JSONResponse:
    recent = list(_LOG_RING)
    return JSONResponse(
        status_code=200, content={"lines": recent, "count": len(recent)}
    )
