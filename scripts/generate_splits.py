"""
scripts/generate_splits.py
==========================
Utility script that scans root_dir/real/ and root_dir/fake/ for .jpg files
and generates a splits.csv file needed by FaceForensicsDataset.

Split ratios: 70% train / 15% val / 15% test  (randomised, reproducible seed).

Usage
-----
    python scripts/generate_splits.py --data_dir /path/to/dataset
    python scripts/generate_splits.py --data_dir /path/to/dataset --seed 99
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from loguru import logger


def generate_splits(
    data_dir: str | Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Path:
    """
    Scan data_dir/real/ and data_dir/fake/ for .jpg files and write splits.csv.

    Parameters
    ----------
    data_dir    : Root dataset directory.
    train_ratio : Fraction of files assigned to the train split.
    val_ratio   : Fraction of files assigned to the val split.
                  Remainder goes to test.
    seed        : Random seed for reproducibility.

    Returns
    -------
    Path
        Absolute path to the written splits.csv.
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"'{root}' is not a directory.")

    filenames: list[str] = []
    for subfolder in ("real", "fake"):
        folder = root / subfolder
        if folder.is_dir():
            found = sorted(p.name for p in folder.glob("*.jpg"))
            filenames.extend(found)
            logger.info("Found {} files in {}/", len(found), subfolder)
        else:
            logger.warning("'{}' folder not found — skipping.", folder)

    if not filenames:
        raise FileNotFoundError(
            f"No .jpg files found in '{root}/real/' or '{root}/fake/'."
        )

    random.seed(seed)
    random.shuffle(filenames)

    n = len(filenames)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits: dict[str, str] = {}
    for i, fname in enumerate(filenames):
        if i < n_train:
            splits[fname] = "train"
        elif i < n_train + n_val:
            splits[fname] = "val"
        else:
            splits[fname] = "test"

    csv_path = root / "splits.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "split"])
        for fname, split in splits.items():
            writer.writerow([fname, split])

    counts = {s: list(splits.values()).count(s) for s in ("train", "val", "test")}
    logger.info(
        "splits.csv written → '{}' | train={} val={} test={}",
        csv_path,
        counts["train"],
        counts["val"],
        counts["test"],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate splits.csv for FaceForensicsDataset."
    )
    parser.add_argument("--data_dir", required=True, help="Root dataset directory.")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    csv_path = generate_splits(
        data_dir=args.data_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"Done — splits.csv written to: {csv_path}")


if __name__ == "__main__":
    main()
