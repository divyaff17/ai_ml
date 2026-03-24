"""
training/evaluate.py
====================
Full evaluation and explainability module for the deepfake vision detector.

Functions
---------
evaluate(model, dataloader)
    Run inference on the full test set, print and save metrics, plot
    confusion matrix and ROC curve.

generate_gradcam(model, image_bytes)
    Produce a Grad-CAM heatmap overlay for a single image.

save_heatmap_to_supabase(heatmap_bytes, job_id, supabase_client)
    Upload a heatmap PNG to Supabase Storage and return its public URL.

Usage
-----
    python training/evaluate.py \\
        --data_dir  /path/to/faceforensics \\
        --checkpoint models/checkpoints/best_model.pth
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)
from torch.utils.data import DataLoader
from loguru import logger

# ── Loguru setup ──────────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
)

# ── Output directory ──────────────────────────────────────────────────────────
_EVAL_DIR = Path("evaluation")
_SAMPLES_DIR = _EVAL_DIR / "samples"


def _ensure_dirs() -> None:
    """Create output directories if they do not already exist."""
    _EVAL_DIR.mkdir(parents=True, exist_ok=True)
    _SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# evaluate
# =============================================================================


def evaluate(
    model: "DeepfakeVisionModel",  # type: ignore[name-defined]
    dataloader: DataLoader,
) -> dict:
    """
    Run inference on an entire dataset split and compute evaluation metrics.

    Outputs
    -------
    * Prints AUC-ROC, F1, Precision, Recall, Accuracy to stdout.
    * Saves ``evaluation/confusion_matrix.png``.
    * Saves ``evaluation/roc_curve.png``.
    * Saves ``evaluation/metrics.json``.

    Parameters
    ----------
    model : DeepfakeVisionModel
        A loaded :class:`models.vision_model.DeepfakeVisionModel` instance.
    dataloader : DataLoader
        DataLoader for the test (or val) split.

    Returns
    -------
    dict
        ``{
            "accuracy":  float,
            "auc":       float,
            "f1":        float,
            "precision": float,
            "recall":    float,
        }``
    """
    _ensure_dirs()
    device = model.device

    # ── Inference loop ────────────────────────────────────────────────────────
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[float] = []

    model._backbone.eval()
    model._classifier.eval()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)  # (B, 3, 224, 224)
            feats = model._backbone(images)  # (B, 1792)
            logits = model._classifier(feats)  # (B, 2)
            probs = torch.softmax(logits, dim=1)[:, 1]  # (B,) fake-class prob
            preds = (probs >= 0.5).long()

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    # ── Metrics ───────────────────────────────────────────────────────────────
    accuracy = round(accuracy_score(all_labels, all_preds), 4)
    auc = round(roc_auc_score(all_labels, all_probs), 4)
    f1 = round(f1_score(all_labels, all_preds, zero_division=0), 4)
    precision = round(precision_score(all_labels, all_preds, zero_division=0), 4)
    recall = round(recall_score(all_labels, all_preds, zero_division=0), 4)

    metrics = {
        "accuracy": accuracy,
        "auc": auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

    # ── Print ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  Evaluation Results")
    print("=" * 50)
    for name, value in metrics.items():
        print(f"  {name:<10} : {value:.4f}")
    print("=" * 50 + "\n")

    logger.info("Metrics: {}", metrics)

    # ── Save metrics.json ─────────────────────────────────────────────────────
    metrics_path = _EVAL_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Metrics saved → '{}'.", metrics_path)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    display = ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"])

    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Deepfake Detector", fontsize=13, fontweight="bold")
    plt.tight_layout()

    cm_path = _EVAL_DIR / "confusion_matrix.png"
    fig.savefig(str(cm_path), dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved → '{}'.", cm_path)

    # ── ROC curve ─────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(all_labels, all_probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="royalblue", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve — Deepfake Detector", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    plt.tight_layout()

    roc_path = _EVAL_DIR / "roc_curve.png"
    fig.savefig(str(roc_path), dpi=150)
    plt.close(fig)
    logger.info("ROC curve saved → '{}'.", roc_path)

    return metrics


# =============================================================================
# generate_gradcam
# =============================================================================


def generate_gradcam(
    model: "DeepfakeVisionModel",  # type: ignore[name-defined]
    image_bytes: bytes,
) -> Optional[bytes]:
    """
    Generate a Grad-CAM heatmap overlay for a single image.

    Uses the last convolutional block of the EfficientNet-B4 backbone as the
    target layer.  The heatmap is blended onto the original image at 40 % alpha
    and returned as PNG bytes.

    Parameters
    ----------
    model : DeepfakeVisionModel
        A loaded model instance.
    image_bytes : bytes
        Raw JPEG / PNG image bytes.

    Returns
    -------
    bytes or None
        PNG-encoded heatmap overlay, or ``None`` if Grad-CAM fails for any
        reason (missing library, unsupported layer, etc.).
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        logger.warning(
            "generate_gradcam: 'pytorch-grad-cam' is not installed — returning None."
        )
        return None

    try:
        # ── Decode and preprocess image ───────────────────────────────────────
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(pil_img, dtype=np.float32) / 255.0  # (H, W, 3) in [0,1]

        from torchvision import transforms

        _tx = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        tensor = _tx(pil_img).unsqueeze(0).to(model.device)  # (1, 3, 224, 224)

        # ── Combined model for GradCAM ────────────────────────────────────────
        # GradCAM needs a single nn.Module whose forward() returns logits.
        class _CombinedModel(nn.Module):
            def __init__(self, backbone: nn.Module, classifier: nn.Module) -> None:
                super().__init__()
                self.backbone = backbone
                self.classifier = classifier

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.classifier(self.backbone(x))

        combined = _CombinedModel(model._backbone, model._classifier).to(model.device)
        combined.eval()

        # ── Target layer: last block of the EfficientNet backbone ─────────────
        # timm EfficientNet-B4 exposes `.blocks` as a Sequential of MBConv blocks.
        try:
            target_layer = [model._backbone.blocks[-1]]
        except (AttributeError, IndexError):
            logger.warning(
                "generate_gradcam: could not access model._backbone.blocks[-1]. "
                "Falling back to the last child module."
            )
            target_layer = [list(model._backbone.children())[-1]]

        # ── Determine predicted class for the target ──────────────────────────
        with torch.no_grad():
            logits = combined(tensor)
            pred_class = int(logits.argmax(dim=1).item())

        targets = [ClassifierOutputTarget(pred_class)]

        # ── Compute Grad-CAM ──────────────────────────────────────────────────
        with GradCAM(model=combined, target_layers=target_layer) as cam:
            grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]  # (224, 224)

        # ── Resize original image to 224×224 for overlay ──────────────────────
        img_resized = np.array(pil_img.resize((224, 224)), dtype=np.float32) / 255.0

        # Blend: 40 % alpha for the heatmap overlay
        visualization = show_cam_on_image(
            img_resized, grayscale_cam, use_rgb=True, image_weight=0.60
        )

        # ── Encode as PNG bytes ───────────────────────────────────────────────
        out = io.BytesIO()
        Image.fromarray(visualization).save(out, format="PNG")
        logger.info(
            "generate_gradcam: heatmap generated successfully (pred_class={}).",
            pred_class,
        )
        return out.getvalue()

    except Exception as exc:
        logger.warning("generate_gradcam: failed — {} — returning None.", exc)
        return None


# =============================================================================
# save_heatmap_to_supabase
# =============================================================================


def save_heatmap_to_supabase(
    heatmap_bytes: bytes,
    job_id: str,
    supabase_client: object,
) -> str:
    """
    Upload a heatmap PNG to the Supabase Storage bucket ``'heatmaps'``.

    The file is stored at ``heatmaps/{job_id}.png``.  The function then
    retrieves and returns the public URL for that object.

    Parameters
    ----------
    heatmap_bytes : bytes
        PNG-encoded heatmap overlay bytes.
    job_id : str
        Unique job identifier used as the storage filename.
    supabase_client : supabase.Client
        An authenticated Supabase client instance.

    Returns
    -------
    str
        Public URL of the uploaded heatmap, e.g.
        ``https://<project>.supabase.co/storage/v1/object/public/heatmaps/<job_id>.png``.

    Raises
    ------
    RuntimeError
        If the upload or URL retrieval fails.
    """
    bucket = "heatmaps"
    obj_path = f"heatmaps/{job_id}.png"

    try:
        supabase_client.storage.from_(bucket).upload(
            path=obj_path,
            file=heatmap_bytes,
            file_options={"content-type": "image/png"},
        )
        logger.info("Heatmap uploaded to bucket '{}' at path '{}'.", bucket, obj_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to upload heatmap for job '{job_id}' to Supabase Storage: {exc}"
        ) from exc

    try:
        url_response = supabase_client.storage.from_(bucket).get_public_url(obj_path)
        public_url = str(url_response)
        logger.info("Heatmap public URL: {}", public_url)
        return public_url
    except Exception as exc:
        raise RuntimeError(
            f"Heatmap uploaded but could not retrieve public URL for job '{job_id}': {exc}"
        ) from exc


# =============================================================================
# Main block
# =============================================================================


def main() -> None:
    """
    CLI entry point for evaluation.

    Steps
    -----
    1. Load the best checkpoint from ``models/checkpoints/best_model.pth``.
    2. Build the test DataLoader from ``--data_dir``.
    3. Run :func:`evaluate` over the full test set.
    4. Pick 5 random test samples and generate Grad-CAM heatmaps.
    5. Save heatmaps to ``evaluation/samples/``.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate the deepfake vision detector and generate Grad-CAM heatmaps."
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Root FaceForensics++ dataset directory (contains real/, fake/, splits.csv).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/checkpoints/best_model.pth",
        help="Path to the .pth checkpoint file.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    _ensure_dirs()

    # ── Resolve imports ───────────────────────────────────────────────────────
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
    from models.vision_model import DeepfakeVisionModel
    from training.train import FaceForensicsDataset, get_transforms

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error("Checkpoint not found at '{}'. Train the model first.", ckpt_path)
        sys.exit(1)

    model = DeepfakeVisionModel(checkpoint_path=str(ckpt_path))
    logger.info("Model loaded from '{}'.", ckpt_path)

    # ── Test DataLoader ───────────────────────────────────────────────────────
    test_ds = FaceForensicsDataset(
        root_dir=args.data_dir,
        split="test",
        transform=get_transforms("test"),
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    # ── Full evaluation ───────────────────────────────────────────────────────
    logger.info("Running evaluation on {} test samples …", len(test_ds))
    metrics = evaluate(model, test_loader)
    logger.success(
        "Evaluation complete. AUC={:.4f}, F1={:.4f}.", metrics["auc"], metrics["f1"]
    )

    # ── Grad-CAM on 5 random samples ─────────────────────────────────────────
    n_samples = min(5, len(test_ds))
    sample_idxs = random.sample(range(len(test_ds)), k=n_samples)

    logger.info("Generating Grad-CAM heatmaps for {} random test samples …", n_samples)

    for rank, idx in enumerate(sample_idxs, start=1):
        img_path, label = test_ds.samples[idx]
        image_bytes = img_path.read_bytes()

        heatmap = generate_gradcam(model, image_bytes)

        if heatmap is None:
            logger.warning(
                "Sample {}/{}: Grad-CAM returned None — skipping.", rank, n_samples
            )
            continue

        label_str = "fake" if label == 1 else "real"
        out_name = f"sample_{rank:02d}_{label_str}_{img_path.stem}.png"
        out_path = _SAMPLES_DIR / out_name
        out_path.write_bytes(heatmap)
        logger.info("Sample {}/{}: heatmap saved → '{}'.", rank, n_samples, out_path)

    logger.success(
        "Done. Outputs in '{}/'.\n"
        "  confusion_matrix.png\n"
        "  roc_curve.png\n"
        "  metrics.json\n"
        "  samples/",
        _EVAL_DIR,
    )


if __name__ == "__main__":
    main()
