"""
tasks/celery_app.py — Celery application factory and async detection tasks.

Workers are started with
    celery -A tasks.celery_app worker --loglevel=info

Tasks
-----
process_video  — full multi-modal video pipeline (vision + temporal + audio → fuse)
"""

from __future__ import annotations

import io
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
from celery import Celery
from loguru import logger

import config

# ── Loguru console setup ──────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    level=config.LOG_LEVEL,
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>celery</cyan> | {message}"
    ),
)

# ── Celery application ───────────────────────────────────────────────────────

celery = Celery(
    "deepfake_detector",
    broker=config.REDIS_URL,
    backend=config.REDIS_URL,
    include=["tasks.celery_app"],
)

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    result_expires=3600,
    worker_prefetch_multiplier=1,  # one task at a time for GPU-heavy work
)


# ── Supabase helpers ──────────────────────────────────────────────────────────


def _get_supabase():
    """Lazily create and return a Supabase client."""
    from supabase import create_client

    return create_client(config.SUPABASE_URL, config.SUPABASE_KEY)


def _update_detection(job_id: str, updates: dict[str, Any]) -> None:
    """Update a row in the ``detections`` table by ``job_id``."""
    try:
        _get_supabase().table("detections").update(updates).eq(
            "job_id", job_id
        ).execute()
        logger.info(
            "Detection updated — job_id={} keys={}", job_id, list(updates.keys())
        )
    except Exception as exc:
        logger.error("Supabase update failed for job {}: {}", job_id, exc)


def _upload_heatmap(heatmap_bytes: bytes, job_id: str) -> Optional[str]:
    """Upload Grad-CAM heatmap PNG to Supabase Storage and return its public URL."""
    try:
        sb = _get_supabase()
        obj_path = f"heatmaps/{job_id}.png"
        sb.storage.from_("heatmaps").upload(
            path=obj_path,
            file=heatmap_bytes,
            file_options={"content-type": "image/png"},
        )
        return str(sb.storage.from_("heatmaps").get_public_url(obj_path))
    except Exception as exc:
        logger.warning("Heatmap upload failed for job {}: {}", job_id, exc)
        return None


def _download_from_url(url: str) -> bytes:
    """
    Download raw bytes from a public URL (Supabase Storage or any HTTP source).

    Parameters
    ----------
    url : str  Public URL to the media file.

    Returns
    -------
    bytes
    """
    import httpx

    resp = httpx.get(url, timeout=120, follow_redirects=True)
    resp.raise_for_status()
    return resp.content


def _extract_audio_track(video_path: str, out_wav_path: str) -> bool:
    """
    Extract the audio track from a video file using ffmpeg.

    Parameters
    ----------
    video_path   : Path to the input video.
    out_wav_path : Path for the output 16 kHz mono WAV file.

    Returns
    -------
    bool
        ``True`` if extraction succeeded, ``False`` otherwise.
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-vn",  # no video
                "-acodec",
                "pcm_s16le",  # WAV
                "-ar",
                "16000",  # 16 kHz
                "-ac",
                "1",  # mono
                out_wav_path,
            ],
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0:
            err_msg = result.stderr.decode(errors="ignore")[:500] if result.stderr else "Unknown ffmpeg error"
            logger.warning(
                "ffmpeg audio extraction failed: {}",
                err_msg,
            )
            return False
        return Path(out_wav_path).stat().st_size > 0
    except FileNotFoundError:
        logger.warning("ffmpeg not found — cannot extract audio track.")
        return False
    except Exception as exc:
        logger.warning("Audio extraction error: {}", exc)
        return False


# =============================================================================
# process_video  Celery task
# =============================================================================


@celery.task(
    bind=True,
    name="tasks.process_video",
    max_retries=2,
    default_retry_delay=10,
    acks_late=True,
)
def process_video(self, job_id: str, media_url: Optional[str]) -> dict:
    """
    Full multi-modal video deepfake detection pipeline.

    Steps
    -----
    1. Download video from Supabase Storage URL.
    2. Extract face crops via ``FaceExtractor.extract_from_video()``.
    3. Run ``DeepfakeVisionModel.predict()`` on each face crop.
    4. Compute ``temporal_score()`` across consecutive frame embeddings.
    5. Extract audio track (ffmpeg → 16 kHz WAV) and run ``AudioDetector``.
    6. Fuse all modality scores via ``fuse_scores()``.
    7. Generate Grad-CAM heatmap for the highest-confidence frame.
    8. Update ``detections`` table with final results.
    9. If fused confidence > 0.85 and is_fake: trigger investigation agent.

    Parameters
    ----------
    job_id    : Unique detection job identifier.
    media_url : Public URL to the video in Supabase Storage.

    Returns
    -------
    dict  Final result row.
    """
    logger.info("=== process_video START — job_id={} ===", job_id)
    _update_detection(job_id, {"status": "processing"})

    # ── 1. Download video ─────────────────────────────────────────────────────
    try:
        if media_url:
            video_bytes = _download_from_url(media_url)
        else:
            logger.error("No media_url for job {}", job_id)
            _update_detection(job_id, {"status": "error"})
            return {"job_id": job_id, "status": "error", "detail": "No media URL."}
    except Exception as exc:
        logger.error("Download failed for job {}: {}", job_id, exc)
        _update_detection(job_id, {"status": "error"})
        raise self.retry(exc=exc)

    # Write to a temp file for OpenCV / ffmpeg
    tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_video.write(video_bytes)
    tmp_video.flush()
    tmp_video_path = tmp_video.name
    tmp_video.close()

    try:
        # ── 2. Extract faces from frames ──────────────────────────────────────
        from models.vision_model import (
            DeepfakeVisionModel,
            FaceExtractor,
            temporal_score,
            fuse_scores,
        )

        extractor = FaceExtractor()
        face_results = extractor.extract_from_video(video_bytes, max_frames=30)

        logger.info(
            "Extracted {} face crops from video for job {}", len(face_results), job_id
        )

        # ── 3. Per-frame vision predictions ───────────────────────────────────
        vision_model = DeepfakeVisionModel.get()
        frame_scores: list[float] = []
        face_arrays: list[np.ndarray] = []
        best_frame_idx: int = -1
        best_confidence: float = 0.0
        best_face_bytes: Optional[bytes] = None

        from PIL import Image as PILImage

        for frame_idx, face_array in face_results:
            # Convert face array to JPEG bytes for predict()
            img = PILImage.fromarray(face_array.astype(np.uint8))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            face_jpg = buf.getvalue()

            pred = vision_model.predict(face_jpg)
            frame_scores.append(
                pred["confidence"] if pred["is_fake"] else 1 - pred["confidence"]
            )
            face_arrays.append(face_array)

            if pred["confidence"] > best_confidence:
                best_confidence = pred["confidence"]
                best_frame_idx = frame_idx
                best_face_bytes = face_jpg

        # Mean vision score across all frames
        vision_score = float(np.mean(frame_scores)) if frame_scores else 0.0
        logger.info(
            "Vision score: {:.4f} ({} frames) for job {}",
            vision_score,
            len(frame_scores),
            job_id,
        )

        # ── 4. Temporal consistency score ─────────────────────────────────────
        t_score = 0.0
        if len(face_arrays) >= 2:
            t_score = temporal_score(face_arrays, vision_model)
        logger.info("Temporal score: {:.4f} for job {}", t_score, job_id)

        # ── 5. Audio track analysis ───────────────────────────────────────────
        audio_score: Optional[float] = None
        tmp_wav_path = tmp_video_path.replace(".mp4", ".wav")

        if _extract_audio_track(tmp_video_path, tmp_wav_path):
            try:
                from models.audio_model import AudioDetector

                audio_model = AudioDetector.get()
                wav_bytes = Path(tmp_wav_path).read_bytes()
                audio_pred = audio_model.predict(wav_bytes)
                audio_score = audio_pred["fake_score"]
                logger.info("Audio score: {:.4f} for job {}", audio_score, job_id)
            except Exception as exc:
                logger.warning(
                    "Audio model inference failed for job {}: {}", job_id, exc
                )
            finally:
                Path(tmp_wav_path).unlink(missing_ok=True)
        else:
            logger.info(
                "No audio track extracted for job {} — audio score = None", job_id
            )

        # ── 6. Fuse scores ────────────────────────────────────────────────────
        fused_score, is_fake = fuse_scores(vision_score, t_score, audio_score)
        confidence = float(f"{fused_score:.4f}")
        logger.info(
            "Fused score: {:.4f}, is_fake={} for job {}", fused_score, is_fake, job_id
        )

        # ── 7. Grad-CAM heatmap for best frame ───────────────────────────────
        heatmap_url: Optional[str] = None
        if best_face_bytes:
            try:
                from training.evaluate import generate_gradcam

                heatmap_bytes = generate_gradcam(vision_model, best_face_bytes)
                if heatmap_bytes:
                    heatmap_url = _upload_heatmap(heatmap_bytes, job_id)
            except Exception as exc:
                logger.warning("Grad-CAM failed for job {}: {}", job_id, exc)

        # ── 8. Update detections table ────────────────────────────────────────
        agent_triggered = is_fake and confidence > 0.85

        updates = {
            "status": "completed",
            "is_fake": is_fake,
            "confidence": confidence,
            "vision_score": float(f"{vision_score:.4f}"),
            "temporal_score": float(f"{t_score:.4f}"),
            "audio_score": float(f"{audio_score:.4f}") if audio_score is not None else None,
            "fused_score": float(f"{fused_score:.4f}"),
            "heatmap_url": heatmap_url,
            "agent_triggered": agent_triggered,
        }
        _update_detection(job_id, updates)
        logger.success(
            "=== process_video COMPLETE — job_id={} is_fake={} confidence={:.4f} ===",
            job_id,
            is_fake,
            confidence,
        )

        # ── 9. Trigger investigation agent ────────────────────────────────────
        if agent_triggered:
            try:
                from agents.investigation_agent import run_investigation

                logger.info("Triggering investigation agent for job {}", job_id)
                run_investigation(
                    job_id,
                    {
                        "media_type": "video",
                        "is_fake": is_fake,
                        "confidence": confidence,
                        "vision_score": float(f"{vision_score:.4f}"),
                        "temporal_score": float(f"{t_score:.4f}"),
                        "audio_score": (
                            float(f"{audio_score:.4f}") if audio_score is not None else None
                        ),
                    },
                )
            except Exception as exc:
                logger.error("Investigation agent failed for job {}: {}", job_id, exc)

        return {
            "job_id": job_id,
            "status": "completed",
            "is_fake": is_fake,
            "confidence": confidence,
        }

    except Exception as exc:
        logger.exception("process_video FAILED for job {}: {}", job_id, exc)
        _update_detection(job_id, {"status": "error"})
        raise self.retry(exc=exc)

    finally:
        Path(tmp_video_path).unlink(missing_ok=True)
    
    return {}
