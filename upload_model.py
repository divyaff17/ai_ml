"""
upload_model.py — Utility script to upload trained weights to the HuggingFace Hub.

Run this script once after training completes. It requires a valid HF token
to be set in the HF_TOKEN environment variable or via `huggingface-cli login`.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi
from loguru import logger

# Replace with your actual HuggingFace username/organization
REPO_ID = "YOUR_HF_USERNAME/deepfake-detector-weights"
MODEL_PATH = Path("models") / "checkpoints" / "best_model.pth"


def upload():
    opt_token = os.environ.get("HF_TOKEN")
    if not opt_token:
        logger.warning(
            "HF_TOKEN environment variable not set. Assuming you are logged in via `huggingface-cli login`."
        )

    if not MODEL_PATH.exists():
        logger.error(f"Model file not found at {MODEL_PATH}")
        sys.exit(1)

    api = HfApi(token=opt_token)

    # Create the repository if it doesn't exist
    try:
        api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
        logger.info(f"Repository {REPO_ID} verified/created.")
    except Exception as e:
        logger.error(f"Failed to create/verify repository {REPO_ID}: {e}")
        sys.exit(1)

    logger.info(f"Uploading {MODEL_PATH} to {REPO_ID} as best_model.pth...")
    try:
        api.upload_file(
            path_or_fileobj=str(MODEL_PATH),
            path_in_repo="best_model.pth",
            repo_id=REPO_ID,
            repo_type="model",
        )
        logger.success(f"Successfully uploaded best_model.pth to {REPO_ID}.")
    except Exception as e:
        logger.error(f"Failed to upload model file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    upload()
