"""
hf_app.py — HuggingFace Spaces specific Gradio frontend entrypoint.

Matches app.py, but uses a different URL resolving strategy for the backend
hosted on Railway, and demonstrates hf_hub_download usage if local weights
are needed by the frontend or shared utils.
"""

from __future__ import annotations

import io
import json
import os
import time

import gradio as gr
import httpx
import matplotlib
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from loguru import logger

matplotlib.use("Agg")

load_dotenv()

# --- HuggingFace specific ---
try:
    # Download weights from HF model hub (if needed by your agent or UI directly)
    weights_path = hf_hub_download(
        repo_id="divya/deepfake-detector-weights",  # Replace with actual user
        filename="best_model.pth",
    )
    logger.info("Weights downloaded to: {}", weights_path)
except Exception as e:
    logger.warning("Could not download weights from HF hub: {}", e)

# ── Config (injected via HF Spaces Secrets) ──────────────────────────────
# API_URL points to Railway backend URL from HF Space secret
API_URL = os.environ.get("API_URL", "https://deepfake-detector-api.up.railway.app")
API_SECRET_KEY = os.environ.get("API_SECRET_KEY", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

_HEADERS = {"X-API-Key": API_SECRET_KEY}
_CLIENT = httpx.Client(timeout=120)
_POLL_INTERVAL = 3
_POLL_TIMEOUT = 120


def _get_supabase():
    try:
        from supabase import create_client

        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as exc:
        logger.warning("Supabase init failed: {}", exc)
        return None


def _detect_image(file_path: str):
    if not file_path:
        return (
            "⚠️ Please upload an image.",
            None,
            None,
            gr.update(visible=False),
            "",
            "",
            "",
            "",
        )
    try:
        with open(file_path, "rb") as f:
            content = f.read()
        filename = os.path.basename(file_path)
        mime = (
            "image/jpeg"
            if filename.lower().endswith((".jpg", ".jpeg"))
            else "image/png"
        )
        resp = _CLIENT.post(
            f"{API_URL}/detect/image",
            headers=_HEADERS,
            files={"file": (filename, io.BytesIO(content), mime)},
        )
        if resp.status_code != 200:
            return (
                f"❌ API error: {resp.text}",
                None,
                None,
                gr.update(visible=False),
                "",
                "",
                "",
                "",
            )

        data = resp.json()
        is_fake = data.get("is_fake")
        confidence = data.get("confidence", 0.0)
        heatmap_url = data.get("heatmap_url")
        job_id = data.get("job_id", "")

        verdict = (
            f"🔴 FAKE DETECTED\nConf: {confidence:.1%}"
            if is_fake
            else f"🟢 LIKELY REAL\nConf: {1-confidence:.1%}"
        )
        verdict += f"\nJob: {job_id}"

        heatmap_img = None
        if heatmap_url:
            try:
                hm_resp = httpx.get(heatmap_url, timeout=10)
                if hm_resp.status_code == 200:
                    from PIL import Image

                    heatmap_img = Image.open(io.BytesIO(hm_resp.content))
            except Exception:
                pass

        return (
            verdict,
            file_path,
            heatmap_img,
            gr.update(visible=data.get("agent_triggered", False)),
            "",
            "",
            "",
            "",
        )
    except Exception as e:
        return (f"❌ Error: {e}", None, None, gr.update(visible=False), "", "", "", "")


def _detect_video(file_path: str):
    if not file_path:
        return "⚠️ Please upload a video.", "{}"
    try:
        with open(file_path, "rb") as f:
            content = f.read()
        filename = os.path.basename(file_path)
        resp = _CLIENT.post(
            f"{API_URL}/detect/video",
            headers=_HEADERS,
            files={"file": (filename, io.BytesIO(content), "video/mp4")},
        )
        if resp.status_code != 200:
            return f"❌ API error: {resp.text}", "{}"
        data = resp.json()
        job_id = data.get("job_id", "")

        deadline = time.time() + _POLL_TIMEOUT
        while time.time() < deadline:
            time.sleep(_POLL_INTERVAL)
            r = _CLIENT.get(f"{API_URL}/results/{job_id}", headers=_HEADERS)
            if r.status_code == 200:
                row = r.json()
                if row.get("status") not in ("processing", "pending"):
                    is_fake = row.get("is_fake")
                    conf = row.get("confidence", 0.0)
                    verdict = (
                        f"🔴 FAKE\nConf: {conf:.1%}"
                        if is_fake
                        else f"🟢 REAL\nConf: {1-conf:.1%}"
                    )
                    return verdict, json.dumps(row, indent=2)
        return f"⏳ Timed out job {job_id}.", "{}"
    except Exception as e:
        return f"❌ Error: {e}", "{}"


def _detect_audio(file_path: str):
    if not file_path:
        return "⚠️ Please upload audio.", "{}"
    try:
        with open(file_path, "rb") as f:
            content = f.read()
        filename = os.path.basename(file_path)
        resp = _CLIENT.post(
            f"{API_URL}/detect/audio",
            headers=_HEADERS,
            files={"file": (filename, io.BytesIO(content), "audio/wav")},
        )
        if resp.status_code != 200:
            return f"❌ API error: {resp.text}", "{}"
        data = resp.json()
        is_fake = data.get("is_fake")
        conf = data.get("confidence", 0.0)
        verdict = (
            f"🔴 FAKE\nConf: {conf:.1%}" if is_fake else f"🟢 REAL\nConf: {1-conf:.1%}"
        )
        return verdict, json.dumps(data, indent=2)
    except Exception as e:
        return f"❌ Error: {e}", "{}"


def build_ui():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔍 Deepfake Detector (HF Space)")
        with gr.Tabs():
            with gr.Tab("Image"):
                img_in = gr.Image(type="filepath")
                img_btn = gr.Button("Detect")
                img_out_verdict = gr.Textbox(label="Verdict")
                img_out_heatmap = gr.Image(label="Heatmap")
                img_btn.click(
                    _detect_image,
                    inputs=[img_in],
                    outputs=[
                        img_out_verdict,
                        gr.Image(visible=False),
                        img_out_heatmap,
                        gr.Column(visible=False),
                        gr.Textbox(),
                        gr.Textbox(),
                        gr.Markdown(),
                        gr.Markdown(),
                    ],
                )
            with gr.Tab("Video"):
                vid_in = gr.Video()
                vid_btn = gr.Button("Detect")
                vid_out = gr.Textbox(label="Verdict")
                vid_raw = gr.JSON()
                vid_btn.click(
                    _detect_video, inputs=[vid_in], outputs=[vid_out, vid_raw]
                )
            with gr.Tab("Audio"):
                aud_in = gr.Audio(type="filepath")
                aud_btn = gr.Button("Detect")
                aud_out = gr.Textbox(label="Verdict")
                aud_raw = gr.JSON()
                aud_btn.click(
                    _detect_audio, inputs=[aud_in], outputs=[aud_out, aud_raw]
                )
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)
