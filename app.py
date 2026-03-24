"""
app.py — Gradio frontend for the Deepfake Detector API.

Three‑tab interface:
  1. Deepfake Detector  — upload image/video/audio, detect, show results + heatmap
  2. Live Dashboard     — auto‑refreshing metrics, charts, latest detections table
  3. API Docs           — endpoint reference with copy‑paste curl examples

Run with:
    python app.py
"""

from __future__ import annotations

import io
import json
import os
import time
from datetime import datetime, timezone

import gradio as gr
import httpx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from loguru import logger

matplotlib.use("Agg")  # non-interactive backend — safe for threads

load_dotenv()

# ── Config (never hardcode credentials) ──────────────────────────────────────
API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

_HEADERS = {"X-API-Key": API_SECRET_KEY}
_CLIENT = httpx.Client(timeout=120)
_POLL_INTERVAL = 3  # seconds between /results polls
_POLL_TIMEOUT = 120  # max wait for video processing


# ═════════════════════════════════════════════════════════════════════════════
# Supabase helper (direct access for the dashboard tab)
# ═════════════════════════════════════════════════════════════════════════════


def _get_supabase():
    """Return a lazily-created Supabase client for dashboard queries."""
    try:
        from supabase import create_client

        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as exc:
        logger.warning("Supabase client init failed: {}", exc)
        return None


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1  —  Deepfake Detector
# ═════════════════════════════════════════════════════════════════════════════


def _detect_image(file_path: str):
    """Upload an image to POST /detect/image and return formatted results."""
    if not file_path:
        return (
            "⚠️  Please upload an image first.",
            None,
            None,  # original, heatmap
            gr.update(visible=False),  # investigation section
            "",
            "",
            "",
            "",  # assessment, summary, sources, exif
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
                f"❌  API returned {resp.status_code}: {resp.text}",
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
        agent_triggered = data.get("agent_triggered", False)

        # Build verdict label
        if is_fake is None:
            verdict = f"⚠️  INCONCLUSIVE\nError: {data.get('error', 'Unknown')}"
        elif is_fake:
            verdict = f"🔴  FAKE DETECTED\nConfidence: {confidence:.1%}"
        else:
            verdict = f"🟢  LIKELY REAL\nConfidence: {1 - confidence:.1%}"

        verdict += f"\nJob ID: {data.get('job_id', 'N/A')}"

        # Heatmap image (download from URL if available)
        heatmap_img = None
        if heatmap_url:
            try:
                hm_resp = httpx.get(heatmap_url, timeout=10)
                if hm_resp.status_code == 200:
                    from PIL import Image

                    heatmap_img = Image.open(io.BytesIO(hm_resp.content))
            except Exception:
                pass

        # Investigation report (if agent triggered, poll for it)
        show_investigation = False
        assessment = summary = sources_md = exif_md = ""

        if agent_triggered:
            job_id = data.get("job_id", "")
            time.sleep(2)  # small delay to let the agent start
            for _ in range(10):
                try:
                    r = _CLIENT.get(f"{API_URL}/results/{job_id}", headers=_HEADERS)
                    if r.status_code == 200:
                        result = r.json()
                        logs = result.get("agent_logs")
                        if logs and isinstance(logs, list) and len(logs) > 0:
                            log = logs[0]
                            report = log.get("report", {})
                            if isinstance(report, str):
                                try:
                                    report = json.loads(report)
                                except Exception:
                                    report = {}
                            show_investigation = True
                            assessment = report.get("confidence_assessment", "unknown")
                            summary = report.get("summary", "No summary available.")
                            src_list = report.get("sources", [])
                            sources_md = (
                                "\n".join(f"- [{s}]({s})" for s in src_list)
                                if src_list
                                else "No sources found."
                            )
                            flags = report.get("exif_flags", [])
                            exif_md = ", ".join(flags) if flags else "✅ None found"
                            break
                except Exception:
                    pass
                time.sleep(2)

        return (
            verdict,
            file_path,  # original image
            heatmap_img,  # gradcam overlay (or None)
            gr.update(visible=show_investigation),
            assessment,
            summary,
            sources_md,
            exif_md,
        )
    except Exception as exc:
        logger.exception("Image detection failed")
        return (
            f"❌  Error: {exc}",
            None,
            None,
            gr.update(visible=False),
            "",
            "",
            "",
            "",
        )


def _detect_video(file_path: str):
    """Upload a video and poll until done."""
    if not file_path:
        return "⚠️  Please upload a video first.", "{}"
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
            return f"❌  API returned {resp.status_code}: {resp.text}", "{}"

        data = resp.json()
        job_id = data.get("job_id", "")

        # Poll /results/{job_id}
        deadline = time.time() + _POLL_TIMEOUT
        while time.time() < deadline:
            time.sleep(_POLL_INTERVAL)
            r = _CLIENT.get(f"{API_URL}/results/{job_id}", headers=_HEADERS)
            if r.status_code == 200:
                row = r.json()
                st = row.get("status", "processing")
                if st not in ("processing", "pending"):
                    is_fake = row.get("is_fake")
                    confidence = row.get("confidence", 0.0)
                    if st == "error":
                        verdict = f"❌  Processing error\nJob: {job_id}"
                    elif is_fake:
                        verdict = f"🔴  FAKE DETECTED\nConfidence: {confidence:.1%}\nJob: {job_id}"
                    else:
                        verdict = f"🟢  LIKELY REAL\nConfidence: {1 - confidence:.1%}\nJob: {job_id}"
                    return verdict, json.dumps(row, indent=2, default=str)

        return f"⏳  Timed out polling for job {job_id}.", json.dumps(data, indent=2)
    except Exception as exc:
        logger.exception("Video detection failed")
        return f"❌  Error: {exc}", "{}"


def _detect_audio(file_path: str):
    """Upload audio to POST /detect/audio."""
    if not file_path:
        return "⚠️  Please upload an audio file first.", "{}"
    try:
        with open(file_path, "rb") as f:
            content = f.read()

        filename = os.path.basename(file_path)
        mime = "audio/mpeg" if filename.lower().endswith(".mp3") else "audio/wav"
        resp = _CLIENT.post(
            f"{API_URL}/detect/audio",
            headers=_HEADERS,
            files={"file": (filename, io.BytesIO(content), mime)},
        )
        if resp.status_code != 200:
            return f"❌  API returned {resp.status_code}: {resp.text}", "{}"

        data = resp.json()
        is_fake = data.get("is_fake")
        confidence = data.get("confidence", 0.0)

        if is_fake:
            verdict = f"🔴  FAKE AUDIO DETECTED\nConfidence: {confidence:.1%}"
        else:
            verdict = f"🟢  LIKELY REAL AUDIO\nConfidence: {1 - confidence:.1%}"

        verdict += f"\nJob ID: {data.get('job_id', 'N/A')}"
        return verdict, json.dumps(data, indent=2, default=str)
    except Exception as exc:
        logger.exception("Audio detection failed")
        return f"❌  Error: {exc}", "{}"


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2  —  Live Dashboard
# ═════════════════════════════════════════════════════════════════════════════


def _fetch_dashboard():
    """Query Supabase for dashboard metrics and return all components."""
    sb = _get_supabase()
    empty_fig = plt.figure(figsize=(4, 3))
    plt.close(empty_fig)

    defaults = (
        "0",
        "0",
        "0.00",
        "0",  # metric cards
        empty_fig,  # pie chart
        empty_fig,  # line chart
        [],  # data table
    )

    if sb is None:
        return defaults

    try:
        # Fetch last 100 detections ordered by created_at desc
        result = (
            sb.table("detections")
            .select(
                "job_id, media_type, is_fake, confidence, agent_triggered, created_at, status"
            )
            .order("created_at", desc=True)
            .limit(100)
            .execute()
        )
        rows = result.data if result and result.data else []
    except Exception as exc:
        logger.warning("Dashboard fetch failed: {}", exc)
        return defaults

    if not rows:
        return defaults

    # ── Metric cards ─────────────────────────────────────────────────────────
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_rows = [r for r in rows if str(r.get("created_at", "")).startswith(today_str)]

    total_today = str(len(today_rows))
    fake_today = str(sum(1 for r in today_rows if r.get("is_fake")))
    avg_conf = (
        f"{np.mean([r.get('confidence', 0) for r in rows]):.2f}" if rows else "0.00"
    )
    agent_today = str(sum(1 for r in today_rows if r.get("agent_triggered")))

    # ── Pie chart: real vs fake ──────────────────────────────────────────────
    n_fake = sum(1 for r in rows if r.get("is_fake"))
    n_real = len(rows) - n_fake

    pie_fig, ax1 = plt.subplots(figsize=(5, 4))
    if n_fake + n_real > 0:
        colours = ["#ef4444", "#22c55e"]
        ax1.pie(
            [n_fake, n_real],
            labels=["Fake", "Real"],
            colors=colours,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 12, "color": "#1e293b"},
        )
    ax1.set_title(
        "Detection Distribution (last 100)",
        fontsize=13,
        fontweight="bold",
        color="#1e293b",
    )
    pie_fig.patch.set_facecolor("#f8fafc")
    plt.tight_layout()

    # ── Line chart: confidence over time (last 50) ───────────────────────────
    recent = rows[:50]
    recent.reverse()  # oldest → newest
    confs = [r.get("confidence", 0) for r in recent]
    idxs = list(range(1, len(confs) + 1))

    line_fig, ax2 = plt.subplots(figsize=(7, 3.5))
    ax2.plot(idxs, confs, color="#6366f1", linewidth=2, marker="o", markersize=3)
    ax2.fill_between(idxs, confs, alpha=0.15, color="#6366f1")
    ax2.axhline(y=0.5, color="#94a3b8", linestyle="--", linewidth=1, label="Threshold")
    ax2.set_xlabel("Detection №", fontsize=11)
    ax2.set_ylabel("Confidence", fontsize=11)
    ax2.set_title(
        "Confidence Trend (last 50)", fontsize=13, fontweight="bold", color="#1e293b"
    )
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    line_fig.patch.set_facecolor("#f8fafc")
    plt.tight_layout()

    # ── Data table: last 10 ──────────────────────────────────────────────────
    table_data = []
    for r in rows[:10]:
        table_data.append(
            [
                str(r.get("job_id", ""))[:8],
                r.get("media_type", ""),
                "🔴 Fake" if r.get("is_fake") else "🟢 Real",
                f"{r.get('confidence', 0):.3f}",
                str(r.get("created_at", ""))[:19],
            ]
        )

    return (
        total_today,
        fake_today,
        avg_conf,
        agent_today,
        pie_fig,
        line_fig,
        table_data,
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3  —  API Documentation (static Markdown)
# ═════════════════════════════════════════════════════════════════════════════

_API_DOCS_MD = """
# 📡 API Reference

All endpoints require an `X-API-Key` header (except `GET /health`).

---

## `GET /health`
Liveness probe — no auth required.
```bash
curl http://localhost:8000/health
```
**Response:**
```json
{"status": "ok", "model_loaded": true, "supabase_connected": true}
```

---

## `POST /detect/image`
Synchronous image deepfake detection with Grad-CAM heatmap.

**Accepted formats:** `.jpg`, `.jpeg`, `.png`, `.webp` — max 50 MB

```bash
curl -X POST http://localhost:8000/detect/image \\
  -H "X-API-Key: YOUR_KEY" \\
  -F "file=@photo.jpg"
```
**Response:**
```json
{
  "job_id": "abc-123",
  "is_fake": true,
  "confidence": 0.9432,
  "heatmap_url": "https://..../heatmaps/abc-123.png",
  "agent_triggered": true
}
```

---

## `POST /detect/video`
Asynchronous video detection — returns immediately, poll `/results/{job_id}`.

**Accepted formats:** `.mp4`, `.avi`, `.mov`, `.mkv` — max 500 MB

```bash
curl -X POST http://localhost:8000/detect/video \\
  -H "X-API-Key: YOUR_KEY" \\
  -F "file=@clip.mp4"
```
**Response:**
```json
{"job_id": "def-456", "status": "processing", "message": "Poll /results/def-456"}
```

---

## `POST /detect/audio`
Synchronous audio deepfake detection.

**Accepted formats:** `.mp3`, `.wav`, `.flac` — max 100 MB

```bash
curl -X POST http://localhost:8000/detect/audio \\
  -H "X-API-Key: YOUR_KEY" \\
  -F "file=@voice.wav"
```
**Response:**
```json
{"job_id": "ghi-789", "is_fake": false, "confidence": 0.1204}
```

---

## `GET /results/{job_id}`
Poll detection results (used for async video jobs).

```bash
curl http://localhost:8000/results/def-456 \\
  -H "X-API-Key: YOUR_KEY"
```
**Response:** Full detection row including `agent_logs` if the investigation agent ran.

---

## `GET /logs?n=50`
Fetch last *n* structured log lines (max 500). Requires API key.

```bash
curl "http://localhost:8000/logs?n=20" \\
  -H "X-API-Key: YOUR_KEY"
```
"""


# ═════════════════════════════════════════════════════════════════════════════
# Gradio Blocks layout
# ═════════════════════════════════════════════════════════════════════════════

_CSS = """
.verdict-fake { background: linear-gradient(135deg, #fecaca, #fca5a5) !important;
                border-left: 6px solid #ef4444 !important; padding: 16px !important;
                font-size: 1.3rem !important; font-weight: 700 !important; }
.verdict-real { background: linear-gradient(135deg, #bbf7d0, #86efac) !important;
                border-left: 6px solid #22c55e !important; padding: 16px !important;
                font-size: 1.3rem !important; font-weight: 700 !important; }
.metric-card  { text-align: center; padding: 20px; border-radius: 12px;
                background: linear-gradient(135deg, #f1f5f9, #e2e8f0);
                box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
.metric-card h2 { margin: 0; font-size: 2.2rem; color: #1e293b; }
.metric-card p  { margin: 4px 0 0; font-size: 0.85rem; color: #64748b; }
.badge-fake     { display: inline-block; padding: 4px 12px; border-radius: 20px;
                  background: #ef4444; color: #fff; font-weight: 600; }
.badge-real     { display: inline-block; padding: 4px 12px; border-radius: 20px;
                  background: #22c55e; color: #fff; font-weight: 600; }
.badge-warn     { display: inline-block; padding: 4px 12px; border-radius: 20px;
                  background: #f97316; color: #fff; font-weight: 600; }
.badge-ok       { display: inline-block; padding: 4px 12px; border-radius: 20px;
                  background: #22c55e; color: #fff; font-weight: 600; }
"""

with gr.Blocks(
    title="Deepfake Detector",
    theme=gr.themes.Soft(),
    css=_CSS,
) as demo:

    gr.Markdown(
        """
        # 🕵️ Deepfake Detector
        > AI-powered detection for images, videos, and audio — backed by **EfficientNet-B4**, **Wav2Vec2**, and a **LangChain investigation agent**.
        """
    )

    # ────────────────────────────────────────────────────────────────────────
    # TAB 1 — Detector
    # ────────────────────────────────────────────────────────────────────────
    with gr.Tab("🔍 Deepfake Detector"):
        gr.Markdown(
            "### Upload a file to check whether it is AI-generated or manipulated."
        )

        with gr.Tabs() as media_tabs:

            # ──── Image sub-tab ──────────────────────────────────────────
            with gr.Tab("🖼️ Image"):
                with gr.Row():
                    img_input = gr.Image(
                        type="filepath", label="Upload Image (JPG / PNG / WebP)"
                    )
                img_btn = gr.Button("🔎 Analyze Image", variant="primary", size="lg")

                img_verdict = gr.Textbox(label="Verdict", lines=4, interactive=False)

                with gr.Row():
                    img_original = gr.Image(label="Original Image", interactive=False)
                    img_heatmap = gr.Image(label="Grad-CAM Heatmap", interactive=False)

                # Investigation Report (collapsible)
                with gr.Accordion(
                    "🔎 Investigation Report", open=False, visible=False
                ) as inv_section:
                    inv_assessment = gr.Textbox(
                        label="Confidence Assessment", interactive=False
                    )
                    inv_summary = gr.Textbox(
                        label="Summary", lines=4, interactive=False
                    )
                    inv_sources = gr.Markdown(label="Sources")
                    inv_exif = gr.Textbox(label="EXIF Flags", interactive=False)

                img_btn.click(
                    fn=_detect_image,
                    inputs=[img_input],
                    outputs=[
                        img_verdict,
                        img_original,
                        img_heatmap,
                        inv_section,
                        inv_assessment,
                        inv_summary,
                        inv_sources,
                        inv_exif,
                    ],
                )

            # ──── Video sub-tab ──────────────────────────────────────────
            with gr.Tab("🎬 Video"):
                vid_input = gr.Video(label="Upload Video (MP4 / AVI / MOV)")
                vid_btn = gr.Button("🔎 Analyze Video", variant="primary", size="lg")
                vid_verdict = gr.Textbox(label="Verdict", lines=4, interactive=False)
                vid_raw = gr.Code(label="Full API Response", language="json")
                vid_btn.click(
                    fn=_detect_video, inputs=[vid_input], outputs=[vid_verdict, vid_raw]
                )

            # ──── Audio sub-tab ──────────────────────────────────────────
            with gr.Tab("🎙️ Audio"):
                aud_input = gr.Audio(
                    type="filepath", label="Upload Audio (MP3 / WAV / FLAC)"
                )
                aud_btn = gr.Button("🔎 Analyze Audio", variant="primary", size="lg")
                aud_verdict = gr.Textbox(label="Verdict", lines=4, interactive=False)
                aud_raw = gr.Code(label="Full API Response", language="json")
                aud_btn.click(
                    fn=_detect_audio, inputs=[aud_input], outputs=[aud_verdict, aud_raw]
                )

    # ────────────────────────────────────────────────────────────────────────
    # TAB 2 — Live Dashboard
    # ────────────────────────────────────────────────────────────────────────
    with gr.Tab("📊 Live Dashboard"):
        gr.Markdown(
            "### Real-time detection analytics — auto-refreshes every 30 seconds"
        )
        refresh_btn = gr.Button("🔄 Refresh Now", variant="secondary")

        with gr.Row():
            m_total = gr.HTML(
                '<div class="metric-card"><h2>0</h2><p>Total Detections Today</p></div>'
            )
            m_fake = gr.HTML(
                '<div class="metric-card"><h2>0</h2><p>Fakes Detected Today</p></div>'
            )
            m_conf = gr.HTML(
                '<div class="metric-card"><h2>0.00</h2><p>Avg Confidence (last 100)</p></div>'
            )
            m_agent = gr.HTML(
                '<div class="metric-card"><h2>0</h2><p>Agent Triggers Today</p></div>'
            )

        with gr.Row():
            pie_plot = gr.Plot(label="Detection Distribution")
            line_plot = gr.Plot(label="Confidence Trend")

        dash_table = gr.Dataframe(
            headers=["Job ID", "Type", "Verdict", "Confidence", "Created At"],
            label="Last 10 Detections",
            interactive=False,
        )

        def _update_dashboard():
            (
                total,
                fake,
                conf,
                agent,
                pie_fig,
                line_fig,
                table,
            ) = _fetch_dashboard()

            return (
                f'<div class="metric-card"><h2>{total}</h2><p>Total Detections Today</p></div>',
                f'<div class="metric-card"><h2>{fake}</h2><p>Fakes Detected Today</p></div>',
                f'<div class="metric-card"><h2>{conf}</h2><p>Avg Confidence (last 100)</p></div>',
                f'<div class="metric-card"><h2>{agent}</h2><p>Agent Triggers Today</p></div>',
                pie_fig,
                line_fig,
                table,
            )

        refresh_btn.click(
            fn=_update_dashboard,
            inputs=[],
            outputs=[m_total, m_fake, m_conf, m_agent, pie_plot, line_plot, dash_table],
        )

        # Load dashboard data on page load
        demo.load(
            fn=_update_dashboard,
            inputs=[],
            outputs=[m_total, m_fake, m_conf, m_agent, pie_plot, line_plot, dash_table],
        )

    # ────────────────────────────────────────────────────────────────────────
    # TAB 3 — API Docs
    # ────────────────────────────────────────────────────────────────────────
    with gr.Tab("📄 API Docs"):
        gr.Markdown(_API_DOCS_MD)


# ═════════════════════════════════════════════════════════════════════════════
# Launch
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
