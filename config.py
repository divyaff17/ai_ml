"""
config.py — Application settings loaded from .env via pydantic-settings.
"""

from pydantic_settings import BaseSettings
from pydantic import field_validator
from loguru import logger
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""
    TAVILY_API_KEY: str = ""
    API_SECRET_KEY: str = "dev-secret-key-change-in-production"
    REDIS_URL: str = "redis://localhost:6379/0"
    MODEL_CHECKPOINT: str = str(BASE_DIR / "models" / "checkpoints" / "best_model.pth")
    AUDIO_CHECKPOINT: str = str(BASE_DIR / "models" / "checkpoints" / "audio_model.pth")
    MAX_IMAGE_MB: int = 50
    MAX_VIDEO_MB: int = 500
    MAX_AUDIO_MB: int = 100
    FAKE_THRESHOLD: float = 0.5
    AGENT_TRIGGER_THRESHOLD: float = 0.85
    API_URL: str = "http://localhost:8000"
    LOG_LEVEL: str = "INFO"

    @field_validator("SUPABASE_URL")
    @classmethod
    def supabase_url_must_be_set(cls, v):
        if not v or v == "":
            raise ValueError("SUPABASE_URL must be set in .env")
        return v

    @field_validator("SUPABASE_KEY")
    @classmethod
    def supabase_key_must_be_set(cls, v):
        if not v or v == "":
            raise ValueError("SUPABASE_KEY must be set in .env")
        return v

    model_config = {"env_file": ".env", "extra": "ignore"}


try:
    settings = Settings()
except Exception as e:
    logger.warning(f"Settings load warning: {e}")
    logger.warning("Some features may not work without .env")
    settings = Settings.model_construct(
        SUPABASE_URL="",
        SUPABASE_KEY="",
        TAVILY_API_KEY="",
        API_SECRET_KEY="dev-secret-key",
        REDIS_URL="redis://localhost:6379/0",
        API_URL="http://localhost:8000",
        LOG_LEVEL="INFO",
    )

# Configure loguru
LOG_LEVEL = settings.LOG_LEVEL

logger.remove()
logger.add(sys.stderr, level=LOG_LEVEL)
logger.add(
    BASE_DIR / "logs" / "app.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)

# Re-export individual settings as module-level constants for backward compat
SUPABASE_URL = settings.SUPABASE_URL
SUPABASE_KEY = settings.SUPABASE_KEY
TAVILY_API_KEY = settings.TAVILY_API_KEY
API_SECRET_KEY = settings.API_SECRET_KEY
REDIS_URL = settings.REDIS_URL
MODEL_CHECKPOINT = settings.MODEL_CHECKPOINT
AUDIO_CHECKPOINT = settings.AUDIO_CHECKPOINT
MAX_IMAGE_MB = settings.MAX_IMAGE_MB
MAX_VIDEO_MB = settings.MAX_VIDEO_MB
MAX_AUDIO_MB = settings.MAX_AUDIO_MB
FAKE_THRESHOLD = settings.FAKE_THRESHOLD
AGENT_TRIGGER_THRESHOLD = settings.AGENT_TRIGGER_THRESHOLD
API_URL = settings.API_URL

# Create necessary dirs
(BASE_DIR / "logs").mkdir(exist_ok=True)
(BASE_DIR / "models" / "checkpoints").mkdir(parents=True, exist_ok=True)
(BASE_DIR / "evaluation").mkdir(exist_ok=True)


def setup_logging():
    """No-op kept for backward compat — logging is configured at import time."""
    logger.info(
        "Logging initialised — level={level} file={file}",
        level=LOG_LEVEL,
        file=str(BASE_DIR / "logs" / "app.log"),
    )
