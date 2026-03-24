"""
models/vision_model.py
======================
Complete vision detection module for deepfake identification.

Classes
-------
DeepfakeVisionModel  — EfficientNet-B4 binary classifier with embedding extraction.
FaceExtractor        — OpenCV Haar-cascade face detector for images and videos.

Functions
---------
temporal_score(frame_face_list, model)               — Temporal inconsistency metric.
fuse_scores(image_score, temporal_score, audio_score) — Weighted ensemble fusion.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms
from loguru import logger

# ── ImageNet normalisation constants ─────────────────────────────────────────
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# ── Standard inference transform ─────────────────────────────────────────────
_INFER_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]
)

# ── Embedding projection dim ──────────────────────────────────────────────────
_EMBED_DIM = 512

# ── EfficientNet-B4 penultimate feature dim ───────────────────────────────────
_BACKBONE_FEATURES = 1792


# =============================================================================
# DeepfakeVisionModel
# =============================================================================


class DeepfakeVisionModel:
    """
    EfficientNet-B4 binary deepfake image classifier.

    The backbone is loaded from *timm* with ImageNet pre-trained weights.
    The final classifier head is replaced with ``Linear(1792, 2)`` for
    binary real/fake prediction.

    A separate ``Linear(1792, 512)`` projection head is attached for
    512-dimensional face embedding extraction.

    Parameters
    ----------
    checkpoint_path : str or None
        Optional path to a ``.pth`` state-dict file.  If the file exists,
        fine-tuned weights are loaded; otherwise ImageNet weights are kept.

    Usage
    -----
    >>> model  = DeepfakeVisionModel()
    >>> result = model.predict(image_bytes)
    >>> print(result["is_fake"], result["confidence"])
    """

    _instance: Optional["DeepfakeVisionModel"] = None

    def __init__(self, checkpoint_path: Optional[str] = None) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Backbone ─────────────────────────────────────────────────────────
        self._backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=True,
            num_classes=0,  # remove timm's default head; we add our own
            global_pool="avg",
        )

        # ── Classification head ───────────────────────────────────────────────
        self._classifier = nn.Linear(_BACKBONE_FEATURES, 2)

        # ── Embedding projection head ─────────────────────────────────────────
        self._embed_proj = nn.Sequential(
            nn.Linear(_BACKBONE_FEATURES, _EMBED_DIM),
            nn.LayerNorm(_EMBED_DIM),
        )

        self._backbone.eval().to(self.device)
        self._classifier.eval().to(self.device)
        self._embed_proj.eval().to(self.device)

        self.model_version = "efficientnet_b4"
        self.load_checkpoint(checkpoint_path)

    # ── Singleton ─────────────────────────────────────────────────────────────

    @classmethod
    def get(cls, checkpoint_path: Optional[str] = None) -> "DeepfakeVisionModel":
        """
        Return (and lazily create) the singleton model instance.

        Parameters
        ----------
        checkpoint_path : str or None
            Only used on the *first* call when the singleton is created.

        Returns
        -------
        DeepfakeVisionModel
        """
        if cls._instance is None:
            cls._instance = cls(checkpoint_path)
        return cls._instance

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def load_checkpoint(self, path: Optional[str]) -> None:
        """
        Load fine-tuned weights from a ``.pth`` checkpoint file.

        If *path* is ``None`` or the file does not exist, the ImageNet
        pre-trained backbone weights are kept and the heads stay randomly
        initialised (suitable for inference after fine-tuning, or for quick
        testing before training).

        Parameters
        ----------
        path : str or None
            Filesystem path to a ``torch.save(state_dict, path)`` file.
        """
        if path is None:
            logger.info("No checkpoint path given — using pretrained backbone weights.")
            return

        ckpt = Path(path)
        if not ckpt.exists():
            logger.warning(
                "Checkpoint '{}' not found — using pretrained backbone weights.", path
            )
            return

        try:
            state = torch.load(str(ckpt), map_location=self.device)
            # Support state dicts saved at full-model level or as sub-keys
            if "backbone" in state:
                self._backbone.load_state_dict(state["backbone"], strict=False)
                self._classifier.load_state_dict(state["classifier"], strict=False)
                self._embed_proj.load_state_dict(state["embed_proj"], strict=False)
            else:
                # Flat state dict — best-effort load
                self._backbone.load_state_dict(state, strict=False)
            logger.info("Loaded checkpoint from '{}'.", path)
        except Exception as exc:
            logger.error(
                "Failed to load checkpoint '{}': {}. Using pretrained weights.",
                path,
                exc,
            )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _decode_image(self, image_bytes: bytes) -> Image.Image:
        """
        Decode raw image bytes into a PIL RGB image.

        Parameters
        ----------
        image_bytes : bytes
            JPEG, PNG, or any PIL-supported format.

        Returns
        -------
        PIL.Image.Image

        Raises
        ------
        ValueError
            If the bytes cannot be decoded as an image.
        """
        try:
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Cannot decode image bytes: {exc}") from exc

    def _extract_features(self, image_bytes: bytes) -> torch.Tensor:
        """
        Run the EfficientNet-B4 backbone on an image and return pooled features.

        Parameters
        ----------
        image_bytes : bytes
            Raw image bytes.

        Returns
        -------
        torch.Tensor
            Shape ``(1, 1792)`` feature tensor on ``self.device``.
        """
        img = self._decode_image(image_bytes)
        tensor = _INFER_TRANSFORM(img).unsqueeze(0).to(self.device)  # (1, 3, 224, 224)

        with torch.no_grad():
            features = self._backbone(tensor)  # (1, 1792) after global avg-pool

        return features

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, image_bytes: bytes) -> dict:
        """
        Run binary deepfake classification on a single image.

        Parameters
        ----------
        image_bytes : bytes
            Raw image bytes (JPEG / PNG / etc.).

        Returns
        -------
        dict
            On success::

                {"is_fake": bool, "confidence": float, "embedding": Tensor}

            On failure::

                {"is_fake": None, "confidence": 0.0, "error": str}

            Possible error strings: ``"no_face_detected"``, ``"inference_failed"``.
        """
        try:
            features = self._extract_features(image_bytes)  # (1, 1792)
        except ValueError as exc:
            # Bad image bytes — e.g. corrupt file or unsupported format
            logger.warning("predict: image decode failed — {}", exc)
            return {"is_fake": None, "confidence": 0.0, "error": "no_face_detected"}

        try:
            logits = self._classifier(features)  # (1, 2)
            probs = torch.softmax(logits, dim=1)  # (1, 2)
            fake_prob = float(probs[0, 1].item())
            is_fake = fake_prob >= 0.5
            confidence = round(max(fake_prob, 1.0 - fake_prob), 4)

            with torch.no_grad():
                embedding = self._embed_proj(features).squeeze(0).cpu()  # (512,)

            return {
                "is_fake": is_fake,
                "confidence": confidence,
                "embedding": embedding,
            }

        except RuntimeError as exc:
            # Covers CUDA OOM, cuDNN errors, NCCL failures, etc.
            logger.error("predict: runtime error during inference — {}", exc)
            return {"is_fake": None, "confidence": 0.0, "error": "inference_failed"}
        except Exception as exc:
            logger.error("predict: unexpected inference error — {}", exc)
            return {"is_fake": None, "confidence": 0.0, "error": "inference_failed"}

    def get_embedding(self, image_bytes: bytes) -> Optional[torch.Tensor]:
        """
        Extract a 512-dimensional face embedding for a single image.

        The embedding is L2-normalised and suitable for cosine-similarity
        comparisons.

        Parameters
        ----------
        image_bytes : bytes
            Raw image bytes.

        Returns
        -------
        torch.Tensor or None
            Shape ``(512,)`` on CPU, L2-normalised.  Returns ``None`` if
            feature extraction or inference fails.
        """
        try:
            features = self._extract_features(image_bytes)  # (1, 1792)
            with torch.no_grad():
                emb = self._embed_proj(features).squeeze(0).cpu()  # (512,)
                emb = F.normalize(emb, p=2, dim=0)
            return emb
        except ValueError as exc:
            logger.warning("get_embedding: image decode failed — {}", exc)
            return None
        except RuntimeError as exc:
            logger.error("get_embedding: runtime error — {}", exc)
            return None
        except Exception as exc:
            logger.error("get_embedding: unexpected error — {}", exc)
            return None


# =============================================================================
# FaceExtractor
# =============================================================================


class FaceExtractor:
    """
    OpenCV Haar-cascade based face detector.

    Used to crop face regions from images and to sample faces from video
    frames before passing them to :class:`DeepfakeVisionModel`.

    Usage
    -----
    >>> extractor = FaceExtractor()
    >>> faces = extractor.extract_from_image(image_bytes)
    """

    def __init__(self) -> None:
        # Load Haar cascade bundled with OpenCV
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            logger.warning(
                "Haar cascade could not be loaded from '%s'. "
                "Face detection will return empty results.",
                cascade_path,
            )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _detect_faces(self, bgr_frame: np.ndarray) -> list[np.ndarray]:
        """
        Run the Haar cascade on a BGR frame and return cropped face arrays.

        Parameters
        ----------
        bgr_frame : np.ndarray
            OpenCV-style BGR image array.

        Returns
        -------
        list of np.ndarray
            Each element is an RGB ``uint8`` crop of a detected face,
            resized to 224 × 224 pixels.
        """
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        rects = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48),
        )

        crops: list[np.ndarray] = []
        for x, y, w, h in rects if len(rects) > 0 else []:
            # Add a 20 % margin around the bounding box
            margin = int(max(w, h) * 0.2)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(bgr_frame.shape[1], x + w + margin)
            y2 = min(bgr_frame.shape[0], y + h + margin)

            crop = bgr_frame[y1:y2, x1:x2]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = cv2.resize(crop, (224, 224))
            crops.append(crop)

        return crops

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_from_image(self, image_bytes: bytes) -> list[np.ndarray]:
        """
        Detect and crop all faces from a single image.

        Parameters
        ----------
        image_bytes : bytes
            Raw JPEG / PNG image bytes.

        Returns
        -------
        list of np.ndarray
            List of ``(224, 224, 3)`` RGB ``uint8`` face crops.
            Returns an empty list (with a logged warning) if no faces are found.
        """
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception as exc:
            logger.warning("FaceExtractor: could not decode image bytes — {}", exc)
            return []

        faces = self._detect_faces(bgr)
        if not faces:
            logger.warning(
                "FaceExtractor: no faces detected in image. "
                "Ensure the image contains a clearly visible frontal face."
            )
        return faces

    def extract_from_video(
        self,
        video_bytes: bytes,
        max_frames: int = 30,
    ) -> list[tuple[int, np.ndarray]]:
        """
        Sample frames from a video and extract one face per sampled frame.

        Frames are sampled evenly so that exactly *max_frames* frames are
        inspected (or fewer if the video is shorter).

        Parameters
        ----------
        video_bytes : bytes
            Raw MP4 / AVI video bytes.
        max_frames : int
            Maximum number of frames to sample from the video.

        Returns
        -------
        list of (int, np.ndarray)
            Each element is ``(frame_index, face_crop)`` where ``face_crop``
            is a ``(224, 224, 3)`` RGB ``uint8`` array.
            Returns an empty list (with a logged warning) if no faces are found
            in any sampled frame.
        """
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        results: list[tuple[int, np.ndarray]] = []

        try:
            cap = cv2.VideoCapture(tmp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames <= 0:
                logger.warning(
                    "FaceExtractor: video has 0 frames or could not be opened."
                )
                cap.release()
                return []

            # Step size so we sample `max_frames` evenly across the video
            step = max(1, total_frames // max_frames)

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % step == 0 and len(results) < max_frames:
                    faces = self._detect_faces(frame)
                    if faces:
                        results.append(
                            (frame_idx, faces[0])
                        )  # keep the first face per frame

                frame_idx += 1

            cap.release()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        if not results:
            logger.warning(
                "FaceExtractor: no faces detected in any of the sampled video frames. "
                "Try a video with a clearly visible frontal face."
            )

        return results


# =============================================================================
# Module-level functions
# =============================================================================


def temporal_score(
    frame_face_list: list[np.ndarray],
    model: DeepfakeVisionModel,
) -> float:
    """
    Compute a temporal inconsistency score from consecutive face crops.

    For each consecutive pair of faces the cosine similarity of their
    512-dim embeddings is computed.  The final score is::

        1 - mean(cosine_similarities)

    Interpretation
    --------------
    * Score ≈ 0  →  embeddings are very consistent across frames → likely **real**.
    * Score ≈ 1  →  embeddings vary wildly across frames → likely **fake / manipulated**.

    Parameters
    ----------
    frame_face_list : list of np.ndarray
        Ordered list of ``(224, 224, 3)`` RGB face crops from consecutive frames.
    model : DeepfakeVisionModel
        An initialised model instance used to extract embeddings.

    Returns
    -------
    float
        Temporal inconsistency score in ``[0, 1]``.
        Returns ``0.0`` if fewer than two faces are provided.
    """
    if len(frame_face_list) < 2:
        logger.warning("temporal_score: fewer than 2 frames provided — returning 0.0.")
        return 0.0

    # Convert each face array to JPEG bytes so get_embedding() can decode them
    def _face_to_bytes(face: np.ndarray) -> bytes:
        img = Image.fromarray(face.astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    embeddings: list[torch.Tensor] = []
    for face in frame_face_list:
        try:
            emb = model.get_embedding(_face_to_bytes(face))  # (512,) L2-normalised
            embeddings.append(emb)
        except Exception as exc:
            logger.warning("temporal_score: embedding failed for a frame — %s", exc)

    if len(embeddings) < 2:
        return 0.0

    similarities: list[float] = []
    for i in range(len(embeddings) - 1):
        # Both tensors are already L2-normalised → dot product == cosine similarity
        sim = float(torch.dot(embeddings[i], embeddings[i + 1]).item())
        sim = max(-1.0, min(1.0, sim))  # clamp to valid cosine range
        similarities.append(sim)

    mean_sim = float(np.mean(similarities))
    score = round(1.0 - mean_sim, 4)
    return float(np.clip(score, 0.0, 1.0))


def fuse_scores(
    image_score: float,
    temporal_score_val: float,
    audio_score: Optional[float],
) -> tuple[float, bool]:
    """
    Combine modality scores into a final deepfake verdict using weighted fusion.

    Default weights
    ~~~~~~~~~~~~~~~
    * With audio: ``image=0.50, temporal=0.30, audio=0.20``
    * Without audio: ``image=0.65, temporal=0.35``

    Parameters
    ----------
    image_score : float
        Fake probability from the vision model, in ``[0, 1]``.
    temporal_score_val : float
        Temporal inconsistency score from :func:`temporal_score`, in ``[0, 1]``.
    audio_score : float or None
        Fake probability from the audio model, in ``[0, 1]``.
        Pass ``None`` if no audio track is available; weights are redistributed.

    Returns
    -------
    tuple of (float, bool)
        ``(final_score, is_fake)`` where *final_score* is the weighted ensemble
        score in ``[0, 1]`` and *is_fake* is ``True`` when ``final_score >= 0.5``.
    """
    if audio_score is None:
        # Redistribute audio weight proportionally to image and temporal
        w_image = 0.65
        w_temporal = 0.35
        final = w_image * image_score + w_temporal * temporal_score_val
    else:
        w_image = 0.50
        w_temporal = 0.30
        w_audio = 0.20
        final = (
            w_image * image_score
            + w_temporal * temporal_score_val
            + w_audio * audio_score
        )

    final = round(float(np.clip(final, 0.0, 1.0)), 4)
    is_fake = final >= 0.5
    return final, is_fake
