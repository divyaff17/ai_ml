"""
training/train.py
=================
Complete model training script for the deepfake vision detector.

Components
----------
FaceForensicsDataset — PyTorch Dataset for the FaceForensics++ folder layout.
get_transforms(split) — Returns appropriate augmentation pipeline per split.
train()               — Main training loop with early stopping, AUC tracking,
                        CSV logging, and best-checkpoint saving.

Usage
-----
    python training/train.py \\
        --data_dir   /path/to/faceforensics \\
        --epochs     20 \\
        --batch_size 32 \\
        --lr         1e-4 \\
        --output_dir models/checkpoints/
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score
from loguru import logger

# ── Loguru setup ──────────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
)

# ── ImageNet constants ────────────────────────────────────────────────────────
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# ── Early-stopping patience ───────────────────────────────────────────────────
_PATIENCE = 5

# ── CSV log header ────────────────────────────────────────────────────────────
_CSV_HEADER = ["epoch", "train_loss", "val_loss", "val_auc"]


# =============================================================================
# FaceForensicsDataset
# =============================================================================


class FaceForensicsDataset(Dataset):
    """
    PyTorch Dataset for the FaceForensics++ folder layout.

    Expected directory structure::

        root_dir/
            real/   *.jpg   (label = 0)
            fake/   *.jpg   (label = 1)
            splits.csv      (columns: filename, split)

    The ``splits.csv`` file must have at least two columns:

    * ``filename`` — basename of the image file (e.g. ``frame_001.jpg``).
    * ``split``    — one of ``"train"``, ``"val"``, or ``"test"``.

    Only files whose ``split`` column matches the *split* argument are loaded.

    Parameters
    ----------
    root_dir : str or Path
        Root directory containing ``real/``, ``fake/``, and ``splits.csv``.
    split : str
        Which dataset split to use: ``"train"``, ``"val"``, or ``"test"``.
    transform : callable or None
        Optional torchvision transform applied to each PIL image.

    Raises
    ------
    FileNotFoundError
        If ``root_dir/splits.csv`` does not exist.
    ValueError
        If an unknown *split* value is provided.
    """

    _VALID_SPLITS = {"train", "val", "test"}

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        transform: Optional[object] = None,
    ) -> None:
        if split not in self._VALID_SPLITS:
            raise ValueError(
                f"split must be one of {self._VALID_SPLITS}, got '{split}'."
            )

        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        splits_csv = self.root_dir / "splits.csv"
        if not splits_csv.exists():
            raise FileNotFoundError(
                f"splits.csv not found at '{splits_csv}'. "
                "Generate it with one row per image: filename,split"
            )

        df = pd.read_csv(splits_csv, dtype=str)
        if "filename" not in df.columns or "split" not in df.columns:
            raise ValueError("splits.csv must contain 'filename' and 'split' columns.")

        split_files: set[str] = set(df.loc[df["split"] == split, "filename"].tolist())

        self.samples: list[tuple[Path, int]] = []

        for label, subfolder in [(0, "real"), (1, "fake")]:
            folder = self.root_dir / subfolder
            if not folder.is_dir():
                logger.warning("Expected folder '{}' not found — skipping.", folder)
                continue
            for img_path in sorted(folder.glob("*.jpg")):
                if img_path.name in split_files:
                    self.samples.append((img_path, label))

        if not self.samples:
            logger.warning(
                "FaceForensicsDataset(split='{}') loaded 0 samples from '{}'. "
                "Check splits.csv and that real/ and fake/ folders contain .jpg files.",
                split,
                root_dir,
            )
        else:
            n_real = sum(1 for _, lbl in self.samples if lbl == 0)
            n_fake = len(self.samples) - n_real
            logger.info(
                "FaceForensicsDataset(split='{}') — {} samples ({} real, {} fake).",
                split,
                len(self.samples),
                n_real,
                n_fake,
            )

    def __len__(self) -> int:
        """Return the total number of samples in this split."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Return one sample.

        Parameters
        ----------
        idx : int
            Integer index.

        Returns
        -------
        tuple of (torch.Tensor, int)
            ``(image_tensor, label)`` where label is 0 (real) or 1 (fake).
        """
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# =============================================================================
# get_transforms
# =============================================================================


def get_transforms(split: str) -> transforms.Compose:
    """
    Return the appropriate torchvision transform pipeline for a dataset split.

    Train augmentations
    -------------------
    * ``RandomHorizontalFlip(p=0.5)``
    * ``ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)``
    * ``RandomResizedCrop(224, scale=(0.8, 1.0))``
    * ``ToTensor()``
    * ``Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])``

    Val / test augmentations
    ------------------------
    * ``Resize(256)``
    * ``CenterCrop(224)``
    * ``ToTensor()``
    * ``Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])``

    Parameters
    ----------
    split : str
        ``"train"``, ``"val"``, or ``"test"``.

    Returns
    -------
    torchvision.transforms.Compose
    """
    normalize = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

    if split == "train":
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )


# =============================================================================
# Internal helpers
# =============================================================================


def _build_model_for_training(
    device: torch.device,
) -> tuple[nn.Module, nn.Module, nn.Module]:
    """
    Instantiate and return the three trainable components of DeepfakeVisionModel.

    Returns the backbone, classifier head, and embedding projection as
    separate ``nn.Module`` objects so they can be optimised together.

    Parameters
    ----------
    device : torch.device

    Returns
    -------
    tuple of (backbone, classifier, embed_proj)
    """
    import timm

    backbone = timm.create_model(
        "efficientnet_b4", pretrained=True, num_classes=0, global_pool="avg"
    )
    classifier = nn.Linear(1792, 2)
    embed_proj = nn.Sequential(nn.Linear(1792, 512), nn.LayerNorm(512))
    return (
        backbone.to(device),
        classifier.to(device),
        embed_proj.to(device),
    )


def _forward(
    backbone: nn.Module,
    classifier: nn.Module,
    images: torch.Tensor,
) -> torch.Tensor:
    """
    Combined backbone + classifier forward pass.

    Parameters
    ----------
    backbone   : nn.Module
    classifier : nn.Module
    images     : torch.Tensor  shape (B, 3, 224, 224)

    Returns
    -------
    torch.Tensor
        Logits of shape (B, 2).
    """
    features = backbone(images)  # (B, 1792)
    return classifier(features)  # (B, 2)


def _run_epoch(
    backbone: nn.Module,
    classifier: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    is_train: bool,
) -> tuple[float, float, list[int], list[float]]:
    """
    Run one full epoch (train or eval).

    Parameters
    ----------
    backbone, classifier : nn.Module
        Model components.
    loader : DataLoader
        Data loader for the current split.
    criterion : nn.Module
        Loss function (CrossEntropyLoss).
    optimizer : Optimizer or None
        Pass None for eval mode.
    device : torch.device
    is_train : bool
        If True, backpropagation is performed.

    Returns
    -------
    tuple of (mean_loss, auc, all_labels, all_probs)
        *auc* is 0.0 if only one class is present in the batch.
    """
    backbone.train(is_train)
    classifier.train(is_train)

    total_loss = 0.0
    all_labels: list[int] = []
    all_probs: list[float] = []

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = _forward(backbone, classifier, images)
            loss = criterion(logits, labels)

            if is_train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().tolist())
            total_loss += loss.item()

    mean_loss = total_loss / max(len(loader), 1)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0  # only one class present

    return mean_loss, auc, all_labels, all_probs


def _save_checkpoint(
    backbone: nn.Module,
    classifier: nn.Module,
    embed_proj: nn.Module,
    output_path: Path,
) -> None:
    """
    Save model state dicts to a ``.pth`` file.

    The saved dict has keys ``"backbone"``, ``"classifier"``, and
    ``"embed_proj"`` so :meth:`DeepfakeVisionModel.load_checkpoint` can
    restore all three heads correctly.

    Parameters
    ----------
    backbone, classifier, embed_proj : nn.Module
    output_path : Path
        Target ``.pth`` file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "backbone": backbone.state_dict(),
            "classifier": classifier.state_dict(),
            "embed_proj": embed_proj.state_dict(),
        },
        str(output_path),
    )
    logger.info("Checkpoint saved → '{}'.", output_path)


def _init_csv_log(log_path: Path) -> None:
    """
    Create (or overwrite) the training log CSV with a header row.

    Parameters
    ----------
    log_path : Path
        Target CSV file path.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_CSV_HEADER)


def _append_csv_row(log_path: Path, row: list) -> None:
    """
    Append one epoch row to the training log CSV.

    Parameters
    ----------
    log_path : Path
    row : list
        Values matching ``_CSV_HEADER``.
    """
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


# =============================================================================
# train()
# =============================================================================


def train() -> None:
    """
    Main training entry point.

    Parses CLI arguments, builds data loaders and model, runs the training
    loop with early stopping and AUC-based model selection, and prints a
    final summary.

    CLI arguments
    -------------
    --data_dir   : Root directory of the FaceForensics++ dataset.
    --epochs     : Maximum number of epochs (default 20).
    --batch_size : Mini-batch size (default 32).
    --lr         : AdamW learning rate (default 1e-4).
    --output_dir : Directory to save the best checkpoint and CSV log.
    """
    # ── Argument parsing ──────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Train the Deepfake Vision Detector (EfficientNet-B4)."
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Root FaceForensics++ dataset directory (contains real/, fake/, splits.csv).",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/checkpoints/",
        help="Directory for best_model.pth and training_log.csv.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ckpt_path = output_dir / "best_model.pth"
    csv_log_path = Path("training") / "training_log.csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    _init_csv_log(csv_log_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: {}", device)
    logger.info(
        "Config: epochs={}, batch={}, lr={}, data='{}'",
        args.epochs,
        args.batch_size,
        args.lr,
        args.data_dir,
    )

    # ── Dataset & DataLoaders ─────────────────────────────────────────────────
    train_ds = FaceForensicsDataset(
        root_dir=args.data_dir,
        split="train",
        transform=get_transforms("train"),
    )
    val_ds = FaceForensicsDataset(
        root_dir=args.data_dir,
        split="val",
        transform=get_transforms("val"),
    )

    n_workers = min(4, torch.get_num_threads())
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=device.type == "cuda",
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    backbone, classifier, embed_proj = _build_model_for_training(device)

    all_params = (
        list(backbone.parameters())
        + list(classifier.parameters())
        + list(embed_proj.parameters())
    )

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # ── Training state ────────────────────────────────────────────────────────
    best_val_auc: float = 0.0
    best_epoch: int = 0
    patience_count: int = 0
    start_time: float = time.monotonic()

    logger.info("Starting training — max {} epoch(s).", args.epochs)

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        train_loss, train_auc, _, _ = _run_epoch(
            backbone=backbone,
            classifier=classifier,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            is_train=True,
        )

        # ── Validate ───────────────────────────────────────────────────────
        val_loss, val_auc, _, _ = _run_epoch(
            backbone=backbone,
            classifier=classifier,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            is_train=False,
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        logger.info(
            "Epoch {:3d}/{} | train_loss={:.4f} | val_loss={:.4f} | "
            "val_auc={:.4f} | lr={:.2e}",
            epoch,
            args.epochs,
            train_loss,
            val_loss,
            val_auc,
            current_lr,
        )

        # ── Log to CSV ─────────────────────────────────────────────────────
        _append_csv_row(
            csv_log_path,
            [epoch, round(train_loss, 6), round(val_loss, 6), round(val_auc, 6)],
        )

        # ── Model selection ────────────────────────────────────────────────
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_count = 0
            _save_checkpoint(backbone, classifier, embed_proj, ckpt_path)
            logger.success(
                "  ↑ New best val AUC: {:.4f} (epoch {})", best_val_auc, epoch
            )
        else:
            patience_count += 1
            logger.info("  No improvement. Patience: {}/{}", patience_count, _PATIENCE)

        # ── Early stopping ─────────────────────────────────────────────────
        if patience_count >= _PATIENCE:
            logger.warning(
                "Early stopping triggered — val AUC did not improve for {} consecutive epochs.",
                _PATIENCE,
            )
            break

    # ── Final summary ──────────────────────────────────────────────────────────
    elapsed = time.monotonic() - start_time
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)

    print("\n" + "=" * 60)
    print("  Training complete")
    print(f"  Best val AUC : {best_val_auc:.4f}  (epoch {best_epoch})")
    print(f"  Checkpoint   : {ckpt_path}")
    print(f"  Training log : {csv_log_path}")
    print(f"  Total time   : {h:02d}h {m:02d}m {s:02d}s")
    print("=" * 60 + "\n")

    logger.info(
        "Done. Best val AUC={:.4f} at epoch {}. Total time={:02d}h{:02d}m{:02d}s.",
        best_val_auc,
        best_epoch,
        h,
        m,
        s,
    )


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    train()
