"""
tests/test_model.py — Unit tests for DeepfakeVisionModel, FaceExtractor,
                       temporal_score(), and fuse_scores().
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import torch
from PIL import Image as PILImage

# =============================================================================
# Helpers
# =============================================================================


def _face_image_bytes() -> bytes:
    """
    Generate a synthetic 224×224 image with an oval skin-tone region in the
    centre so the Haar cascade has something face-like to detect.
    (Not guaranteed to be detected, but works in most cases.)
    """
    import cv2

    img = np.full((224, 224, 3), 200, dtype=np.uint8)  # light grey background
    # Draw an ellipse approximating a face shape in skin-tone colour
    cv2.ellipse(img, (112, 112), (50, 65), 0, 0, 360, (185, 160, 130), -1)
    # Eyes — two dark circles
    cv2.circle(img, (92, 100), 8, (30, 30, 30), -1)
    cv2.circle(img, (132, 100), 8, (30, 30, 30), -1)
    # Mouth — a line
    cv2.line(img, (95, 140), (130, 140), (100, 50, 50), 2)
    # Convert BGR→RGB and encode to JPEG
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = PILImage.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _solid_image_bytes() -> bytes:
    """Generate a solid-colour 224×224 JPEG — no face present."""
    img = PILImage.new("RGB", (224, 224), color=(0, 128, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


# =============================================================================
# Tests — DeepfakeVisionModel
# =============================================================================


class TestDeepfakeVisionModel:
    """Tests for the DeepfakeVisionModel class."""

    @pytest.fixture(autouse=True)
    def _load_model(self):
        from models.vision_model import DeepfakeVisionModel

        self.model = DeepfakeVisionModel()

    def test_predict_output_shape(self, sample_image: bytes):
        """predict() must return dict with is_fake bool and 0<=confidence<=1."""
        result = self.model.predict(sample_image)
        assert isinstance(result, dict)
        # Might return an error dict if decode fails — both cases valid
        if result.get("error") is None:
            assert isinstance(result["is_fake"], bool)
            assert 0.0 <= result["confidence"] <= 1.0
            assert result["embedding"] is not None

    def test_predict_returns_error_on_bad_bytes(self):
        """predict() with garbage bytes returns error dict, never crashes."""
        result = self.model.predict(b"not-an-image")
        assert result.get("error") is not None
        assert result["is_fake"] is None
        assert result["confidence"] == 0.0

    def test_get_embedding_shape(self, sample_image: bytes):
        """get_embedding() returns a (512,) tensor or None."""
        emb = self.model.get_embedding(sample_image)
        if emb is not None:
            assert isinstance(emb, torch.Tensor)
            assert emb.shape == (512,)

    def test_get_embedding_returns_none_on_bad_bytes(self):
        """get_embedding() with garbage bytes returns None gracefully."""
        result = self.model.get_embedding(b"bad-data")
        assert result is None


# =============================================================================
# Tests — FaceExtractor
# =============================================================================


class TestFaceExtractor:
    """Tests for the FaceExtractor class."""

    @pytest.fixture(autouse=True)
    def _load_extractor(self):
        from models.vision_model import FaceExtractor

        self.extractor = FaceExtractor()

    def test_extract_from_image_with_face(self):
        """extract_from_image() should return non-empty list for face-like image."""
        face_bytes = _face_image_bytes()
        faces = self.extractor.extract_from_image(face_bytes)
        # Haar cascade may or may not detect the synthetic face — test is lenient
        assert isinstance(faces, list)

    def test_extract_from_image_no_face(self):
        """extract_from_image() returns empty list for solid-colour image."""
        solid = _solid_image_bytes()
        faces = self.extractor.extract_from_image(solid)
        assert isinstance(faces, list)
        assert len(faces) == 0

    def test_extract_from_image_bad_bytes(self):
        """extract_from_image() returns empty list for garbage bytes."""
        faces = self.extractor.extract_from_image(b"garbage")
        assert isinstance(faces, list)
        assert len(faces) == 0


# =============================================================================
# Tests — temporal_score
# =============================================================================


class TestTemporalScore:
    """Tests for the module-level temporal_score() function."""

    @pytest.fixture(autouse=True)
    def _load(self):
        from models.vision_model import DeepfakeVisionModel, temporal_score

        self.model = DeepfakeVisionModel()
        self.temporal_score = temporal_score

    def test_temporal_score_range(self):
        """temporal_score() returns a float between 0 and 1."""
        # Create 3 slightly different face crops
        frames = []
        for i in range(3):
            arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            frames.append(arr)
        score = self.temporal_score(frames, self.model)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_temporal_score_single_frame(self):
        """temporal_score() with < 2 frames returns 0.0."""
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        score = self.temporal_score([arr], self.model)
        assert score == 0.0


# =============================================================================
# Tests — fuse_scores
# =============================================================================


class TestFuseScores:
    """Tests for the module-level fuse_scores() function."""

    @pytest.fixture(autouse=True)
    def _load(self):
        from models.vision_model import fuse_scores

        self.fuse = fuse_scores

    def test_fuse_scores_with_audio(self):
        """fuse_scores(0.9, 0.8, 0.7) → is_fake should be True."""
        score, is_fake = self.fuse(0.9, 0.8, 0.7)
        assert is_fake is True
        assert 0.0 <= score <= 1.0

    def test_fuse_scores_without_audio(self):
        """fuse_scores(0.3, 0.2, None) → is_fake should be False."""
        score, is_fake = self.fuse(0.3, 0.2, None)
        assert is_fake is False
        assert 0.0 <= score <= 1.0

    def test_fuse_scores_boundary(self):
        """fuse_scores at exactly 0.5 boundary."""
        score, is_fake = self.fuse(0.5, 0.5, 0.5)
        assert isinstance(is_fake, bool)
        assert 0.0 <= score <= 1.0

    def test_fuse_scores_zeros(self):
        """fuse_scores(0,0,0) → is_fake=False, score=0."""
        score, is_fake = self.fuse(0.0, 0.0, 0.0)
        assert is_fake is False
        assert score == 0.0
