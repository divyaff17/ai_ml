"""Part 12 — Import verification script."""

import sys

results = []


def chk(label, fn):
    try:
        fn()
        results.append(f"  OK   {label}")
    except Exception as e:
        results.append(f"  FAIL {label}: {e}")


# Core stdlib
chk("uuid", lambda: __import__("uuid"))
chk("pathlib", lambda: __import__("pathlib"))
chk("io", lambda: __import__("io"))

# PyTorch
chk("torch", lambda: __import__("torch"))
chk("torchvision", lambda: __import__("torchvision"))
chk("torchaudio", lambda: __import__("torchaudio"))
chk("timm", lambda: __import__("timm"))

# HuggingFace
chk("transformers", lambda: __import__("transformers"))
chk("huggingface_hub", lambda: __import__("huggingface_hub"))

# Vision
chk("cv2", lambda: __import__("cv2"))
chk("PIL", lambda: __import__("PIL"))
chk("pytorch_grad_cam", lambda: __import__("pytorch_grad_cam"))

# FastAPI
chk("fastapi", lambda: __import__("fastapi"))
chk("uvicorn", lambda: __import__("uvicorn"))
chk("starlette", lambda: __import__("starlette"))
chk("multipart", lambda: __import__("multipart"))
chk("httpx", lambda: __import__("httpx"))

# Task queue
chk("celery", lambda: __import__("celery"))
chk("redis", lambda: __import__("redis"))

# Supabase
chk("supabase", lambda: __import__("supabase"))

# LangChain
chk("langchain", lambda: __import__("langchain"))
chk("langchain_core", lambda: __import__("langchain_core"))
chk("langchain_community", lambda: __import__("langchain_community"))
chk("langchain_anthropic", lambda: __import__("langchain_anthropic"))
chk("anthropic", lambda: __import__("anthropic"))
chk("tavily", lambda: __import__("tavily"))

# Gradio
chk("gradio", lambda: __import__("gradio"))

# ML
chk("sklearn", lambda: __import__("sklearn"))
chk("numpy", lambda: __import__("numpy"))
chk("matplotlib", lambda: __import__("matplotlib"))
chk("soundfile", lambda: __import__("soundfile"))
chk("librosa", lambda: __import__("librosa"))

# Utilities
chk("dotenv", lambda: __import__("dotenv"))
chk("loguru", lambda: __import__("loguru"))
chk("slowapi", lambda: __import__("slowapi"))
chk("pydantic", lambda: __import__("pydantic"))
chk("pydantic_settings", lambda: __import__("pydantic_settings"))

# Project modules
chk("config", lambda: __import__("config"))
chk("models.vision_model", lambda: __import__("models.vision_model"))
chk("models.audio_model", lambda: __import__("models.audio_model"))
chk("agents.investigation_agent", lambda: __import__("agents.investigation_agent"))
chk("tasks.celery_app", lambda: __import__("tasks.celery_app"))
chk("middleware.auth", lambda: __import__("middleware.auth"))
chk("utils.supabase_utils", lambda: __import__("utils.supabase_utils"))
chk("main", lambda: __import__("main"))

print("\n" + "=" * 50 + " IMPORT VERIFICATION RESULTS " + "=" * 50)
for r in results:
    print(r)

fails = [r for r in results if "FAIL" in r]
print(f"\n{'=' * 50}")
if fails:
    print(f"FAILED: {len(fails)} import(s) need fixing:")
    for f in fails:
        print(f)
    sys.exit(1)
else:
    print(f"ALL {len(results)} IMPORTS OK -- project is fully wired")
    sys.exit(0)
