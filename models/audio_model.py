"""
models/audio_model.py — Wav2Vec2-based audio deepfake detector.

Classes
-------
DeepfakeAudioModel  — Loads `Wav2Vec2ForSequenceClassification` with binary
                      labels (real=0, fake=1).  Preprocesses raw audio bytes
                      with torchaudio (MP3 / WAV / FLAC → 16 kHz mono →
                      normalised → pad/truncate to 5 s) then classifies.

Functions
---------
extract_audio_from_video(video_bytes)  — Extracts the audio track from a
                                         video using ffmpeg, returning WAV bytes.
train_audio_model()                    — Stub for future ASVspoof 2019 fine-tuning.
"""

from __future__ import annotations

import io
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
import os

import torch
import torch.nn.functional as F
import torchaudio
from loguru import logger
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Processor,
)

import config

# ── Audio constants ───────────────────────────────────────────────────────────
_SAMPLE_RATE = 16_000
_TARGET_SAMPLES = _SAMPLE_RATE * 5  # exactly 5 seconds → 80 000 samples
_MODEL_ID = "facebook/wav2vec2-base"


# =============================================================================
# DeepfakeAudioModel
# =============================================================================


class DeepfakeAudioModel:
    """
    Singleton Wav2Vec2ForSequenceClassification audio deepfake detector.

    The model is loaded from ``facebook/wav2vec2-base`` with ``num_labels=2``
    (class 0 = real, class 1 = fake).  Softmax is applied to the logits and
    the probability of class 1 is returned as the confidence score.

    Audio preprocessing
    -------------------
    1. Accept raw MP3 / WAV / FLAC bytes.
    2. Decode with ``torchaudio.load()`` via ``io.BytesIO``.
    3. Resample to 16 000 Hz if the source sample rate differs.
    4. Convert to mono (average channels).
    5. Normalise waveform to ``[-1, 1]``.
    6. Truncate or zero-pad to exactly 5 seconds (80 000 samples).

    Usage
    -----
    >>> model  = DeepfakeAudioModel.get()
    >>> result = model.predict(audio_bytes)
    >>> print(result["is_fake"], result["confidence"])
    """

    _instance: Optional["DeepfakeAudioModel"] = None

    def __init__(self, checkpoint_path: Optional[str] = None) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading DeepfakeAudioModel on {} …", self.device)

        # ── Set HF token if available (avoids rate-limit warnings) ────────────
        _hf_token: Optional[str] = os.environ.get("HF_TOKEN")
        if not _hf_token:
            try:
                import config as _cfg
                _hf_token = getattr(_cfg.settings, "HF_TOKEN", None)
            except Exception:
                pass
        if _hf_token:
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", _hf_token)

        # ── Processor (feature extractor) ─────────────────────────────────────
        self._processor = Wav2Vec2Processor.from_pretrained(
            _MODEL_ID, token=_hf_token
        )

        # ── Classifier model ─────────────────────────────────────────────────
        self._model = Wav2Vec2ForSequenceClassification.from_pretrained(
            _MODEL_ID,
            num_labels=2,
            problem_type="single_label_classification",
            token=_hf_token,
        )
        self._model.eval().to(self.device)

        self.model_version = "wav2vec2-base-seq-clf"

        # Optionally load fine-tuned weights
        self.load_checkpoint(checkpoint_path)
        logger.info("DeepfakeAudioModel ready — version='{}'.", self.model_version)

    # ── Singleton ─────────────────────────────────────────────────────────────

    @classmethod
    def get(cls, checkpoint_path: Optional[str] = None) -> "DeepfakeAudioModel":
        """
        Return (and lazily create) the singleton model instance.

        Parameters
        ----------
        checkpoint_path : str or None
            Only used on the *first* call when the singleton is created.
        """
        if cls._instance is None:
            cls._instance = cls(checkpoint_path)
        return cls._instance

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def load_checkpoint(self, path: Optional[str]) -> None:
        """
        Load fine-tuned weights from a ``.pth`` or HuggingFace checkpoint dir.

        If *path* is ``None`` or does not exist, the base pre-trained weights
        are kept.

        Parameters
        ----------
        path : str or None
            Path to a ``state_dict`` file or a directory containing a
            HuggingFace ``config.json`` + ``model.safetensors``.
        """
        if path is None:
            logger.info("No audio checkpoint — using base pre-trained weights.")
            return

        p = Path(path)
        if not p.exists():
            logger.info(
                "No audio checkpoint at '{}' — using base pre-trained weights. "
                "Expected on first deploy before training.", path
            )
            return

        try:
            if p.is_dir():
                # HuggingFace-format checkpoint directory
                self._model = (
                    Wav2Vec2ForSequenceClassification.from_pretrained(
                        str(p),
                        num_labels=2,
                    )
                    .eval()
                    .to(self.device)
                )
                logger.info("Loaded HuggingFace audio checkpoint from '{}'.", path)
            else:
                # Flat state_dict .pth file
                state = torch.load(str(p), map_location=self.device)
                self._model.load_state_dict(state, strict=False)
                logger.info("Loaded audio state_dict from '{}'.", path)
        except Exception as exc:
            logger.error(
                "Failed to load audio checkpoint '{}': {} — keeping base weights.",
                path,
                exc,
            )

    # ── Audio preprocessing ───────────────────────────────────────────────────

    def _preprocess(self, audio_bytes: bytes) -> torch.Tensor:
        """
        Decode, resample, normalise, and pad/truncate raw audio bytes.

        Parameters
        ----------
        audio_bytes : bytes
            Raw MP3, WAV, or FLAC file contents.

        Returns
        -------
        torch.Tensor
            Shape ``(1, 80000)`` — mono waveform, 16 kHz, 5 seconds.

        Raises
        ------
        ValueError
            If the bytes cannot be decoded by torchaudio.
        """
        try:
            waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
            # waveform: (channels, samples)
        except Exception as exc:
            raise ValueError(f"Cannot decode audio bytes: {exc}") from exc

        # ── Convert to mono ───────────────────────────────────────────────────
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # (1, samples)

        # ── Resample to 16 kHz ────────────────────────────────────────────────
        if sr != _SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=_SAMPLE_RATE
            )
            waveform = resampler(waveform)

        # ── Normalise to [-1, 1] ──────────────────────────────────────────────
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak

        # ── Pad or truncate to exactly 5 seconds (80 000 samples) ────────────
        n_samples = waveform.shape[1]
        if n_samples > _TARGET_SAMPLES:
            waveform = waveform[:, :_TARGET_SAMPLES]
        elif n_samples < _TARGET_SAMPLES:
            pad_amount = _TARGET_SAMPLES - n_samples
            waveform = F.pad(waveform, (0, pad_amount), mode="constant", value=0.0)

        return waveform  # (1, 80000)

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, audio_bytes: bytes) -> dict:
        """
        Run binary deepfake classification on raw audio bytes.

        Parameters
        ----------
        audio_bytes : bytes
            Raw MP3, WAV, or FLAC audio file.

        Returns
        -------
        dict
            On success::

                {
                    "is_fake":       bool,
                    "confidence":    float,   # probability of the winning class
                    "fake_score":    float,   # probability of class 1 (fake)
                    "model_version": str,
                }

            On failure::

                {"is_fake": None, "confidence": 0.0, "error": str}
        """
        # ── Preprocess ────────────────────────────────────────────────────────
        try:
            waveform = self._preprocess(audio_bytes)  # (1, 80000)
        except ValueError as exc:
            logger.warning("predict (audio): decode failed — {}", exc)
            return {"is_fake": None, "confidence": 0.0, "error": "audio_decode_failed"}

        # ── Feature extraction via Wav2Vec2Processor ──────────────────────────
        try:
            inputs = self._processor(
                waveform.squeeze(0).numpy(),  # (80000,) numpy
                sampling_rate=_SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs.input_values.to(self.device)  # (1, 80000)
        except Exception as exc:
            logger.error("predict (audio): processor failed — {}", exc)
            return {
                "is_fake": None,
                "confidence": 0.0,
                "error": "audio_processing_failed",
            }

        # ── Inference ─────────────────────────────────────────────────────────
        try:
            with torch.no_grad():
                outputs = self._model(input_values)
                logits = outputs.logits  # (1, 2)
                probs = torch.softmax(logits, dim=1)  # (1, 2)
                fake_p = float(probs[0, 1].item())

            is_fake = fake_p >= config.DEEPFAKE_THRESHOLD
            confidence = round(max(fake_p, 1.0 - fake_p), 4)

            logger.info(
                "Audio prediction — fake_score={:.4f} is_fake={} confidence={:.4f}",
                fake_p,
                is_fake,
                confidence,
            )

            return {
                "is_fake": is_fake,
                "confidence": confidence,
                "fake_score": round(fake_p, 4),
                "model_version": self.model_version,
            }

        except RuntimeError as exc:
            logger.error("predict (audio): runtime error — {}", exc)
            return {"is_fake": None, "confidence": 0.0, "error": "inference_failed"}
        except Exception as exc:
            logger.error("predict (audio): unexpected error — {}", exc)
            return {"is_fake": None, "confidence": 0.0, "error": "inference_failed"}


# =============================================================================
# Backward-compatible alias
# =============================================================================

# The previous version of this file used the name ``AudioDetector``.
# Keep it as an alias so existing imports (main.py, tasks/celery_app.py, tests)
# continue to work without modification.
AudioDetector = DeepfakeAudioModel


# =============================================================================
# extract_audio_from_video
# =============================================================================


def extract_audio_from_video(video_bytes: bytes) -> Optional[bytes]:
    """
    Extract the audio track from a video file using ffmpeg.

    The video bytes are written to a temp file, ffmpeg extracts the audio as
    a 16 kHz mono PCM WAV, and the resulting WAV bytes are returned.

    Parameters
    ----------
    video_bytes : bytes
        Raw MP4 / AVI / MOV / MKV video file.

    Returns
    -------
    bytes or None
        WAV audio bytes, or ``None`` if the video has no audio track or
        ffmpeg is not installed / fails.
    """
    tmp_video_path: Optional[str] = None
    tmp_wav_path: Optional[str] = None

    try:
        # ── Write video to a temp file ────────────────────────────────────────
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
            tmp_video.write(video_bytes)
            tmp_video_path = tmp_video.name

        tmp_wav_path = tmp_video_path.replace(".mp4", ".wav")

        # ── Run ffmpeg ────────────────────────────────────────────────────────
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                tmp_video_path,
                "-vn",  # strip video
                "-acodec",
                "pcm_s16le",  # output WAV PCM 16-bit
                "-ar",
                "16000",  # 16 kHz
                "-ac",
                "1",  # mono
                tmp_wav_path,
            ],
            capture_output=True,
            timeout=120,
        )

        if result.returncode != 0:
            stderr = result.stderr.decode(errors="ignore")[:500]
            if "does not contain any stream" in stderr or "no audio" in stderr.lower():
                logger.warning("extract_audio_from_video: video has no audio track.")
            else:
                logger.warning("extract_audio_from_video: ffmpeg failed — {}", stderr)
            return None

        wav_path = Path(tmp_wav_path)
        if not wav_path.exists() or wav_path.stat().st_size == 0:
            logger.warning("extract_audio_from_video: output WAV is empty.")
            return None

        audio_bytes = wav_path.read_bytes()
        logger.info(
            "extract_audio_from_video: extracted {} bytes of 16 kHz mono WAV.",
            len(audio_bytes),
        )
        return audio_bytes

    except FileNotFoundError:
        logger.warning("extract_audio_from_video: ffmpeg not found on PATH.")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("extract_audio_from_video: ffmpeg timed out after 120 s.")
        return None
    except Exception as exc:
        logger.error("extract_audio_from_video: unexpected error — {}", exc)
        return None

    finally:
        # ── Clean up temp files ───────────────────────────────────────────────
        if tmp_video_path:
            Path(tmp_video_path).unlink(missing_ok=True)
        if tmp_wav_path:
            Path(tmp_wav_path).unlink(missing_ok=True)


# =============================================================================
# Training stub — ASVspoof 2019
# =============================================================================

# Integration note:
# For full fine-tuning of this audio model, use the **ASVspoof 2019 LA** dataset
# (Logical Access partition).  The dataset provides bonafide + spoof utterances
# across 19 different speech synthesis / voice conversion attack types.
#
# Download: https://datashare.ed.ac.uk/handle/10283/3336
# Paper:    "ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection"
#
# The recommended training approach:
# 1. Load the ASVspoof 2019 LA training split.
# 2. Preprocess all utterances with the same pipeline as _preprocess() above
#    (resample → normalise → pad/truncate to 5 s).
# 3. Feed through Wav2Vec2Processor and fine-tune the classification head
#    while optionally un-freezing the last N transformer layers.
# 4. Use CrossEntropyLoss with label=0 for bonafide, label=1 for spoof.
# 5. Validate on the ASVspoof 2019 LA dev set using EER (Equal Error Rate)
#    and AUC-ROC.  Target EER < 5 % before switching to the eval set.


def train_audio_model(
    data_dir: str,
    output_dir: str = "models/checkpoints/audio/",
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 2e-5,
    freeze_feature_extractor: bool = True,
) -> None:
    """
    Fine-tune ``DeepfakeAudioModel`` on the ASVspoof 2019 LA dataset.

    This is a **stub** — the full implementation should:

    1. Load the ASVspoof 2019 LA training split from *data_dir*.
       Each sample is a FLAC file with a protocol line indicating
       ``bonafide`` (label 0) or ``spoof`` (label 1).

    2. Build a PyTorch ``Dataset`` that reads each FLAC file, calls
       ``_preprocess()`` to get a fixed-length 80 000-sample tensor,
       and returns ``(input_values, label)``.

    3. Instantiate ``DeepfakeAudioModel`` and optionally freeze the
       Wav2Vec2 feature-extraction CNN (``freeze_feature_extractor=True``)
       while un-freezing the last 4 transformer encoder layers.

    4. Train with ``AdamW(lr=2e-5, weight_decay=0.01)`` and
       ``CrossEntropyLoss``.  Use a linear warmup scheduler for the
       first 500 steps.

    5. Evaluate on the dev set after every epoch.  Track:
       - EER (Equal Error Rate)
       - AUC-ROC
       - minDCF (minimum Detection Cost Function)

    6. Save the best model (by EER) to *output_dir*.

    Parameters
    ----------
    data_dir   : Path to the ASVspoof 2019 LA dataset root.
    output_dir : Where to save checkpoints and logs.
    epochs     : Number of training epochs.
    batch_size : Mini-batch size.
    lr         : Peak learning rate for AdamW.
    freeze_feature_extractor : Whether to freeze the CNN feature extractor.
    """
    raise NotImplementedError(
        "train_audio_model() is a stub.  Implement the full training loop "
        "following the docstring above, using the ASVspoof 2019 LA dataset."
    )
