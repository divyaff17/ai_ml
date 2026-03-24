---
title: Deepfake Detector
emoji: 🔍
colorFrom: purple
colorTo: red
sdk: gradio
sdk_version: 4.19.2
app_file: hf_app.py
pinned: true
hardware: t4-medium
license: mit
---

# Deepfake Detector

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-yellow.svg)](https://huggingface.co/spaces)
[![CI/CD](https://img.shields.io/github/actions/workflow/status/USERNAME/deepfake-detector/deploy.yml?branch=main)](https://github.com/USERNAME/deepfake-detector/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A production-ready **Deepfake Detection API + Agentic AI** system.

| Component | Technology |
|-----------|-----------|
| REST API | FastAPI + slowapi rate limiting |
| Vision model | EfficientNet-B4 (timm) + Grad-CAM |
| Audio model | Wav2Vec2 + MLP head |
| Async tasks | Celery + Redis |
| Investigation agent | LangChain + Tavily web search |
| Database | Supabase (PostgreSQL) |
| Demo UI | Gradio |
| Auth | X-API-Key header |

---

## Project Structure

```
deepfake-detector/
├── main.py                      # FastAPI entry point
├── config.py                    # Environment variable loader
├── app.py                       # Gradio demo frontend
├── requirements.txt             # Pinned dependencies
├── Dockerfile
├── docker-compose.yml
├── .env.template                # Copy to .env and fill in values
├── .gitignore
├── db/
│   └── schema.sql               # Supabase table definitions
├── models/
│   ├── vision_model.py          # EfficientNet-B4 detector + Grad-CAM
│   ├── audio_model.py           # Wav2Vec2 + MLP detector
│   └── checkpoints/             # Fine-tuned .pth files (git-ignored)
├── training/
│   ├── train.py                 # Fine-tuning script (AdamW + cosine LR)
│   └── evaluate.py              # Accuracy / AUC / confusion matrix
├── agents/
│   └── investigation_agent.py   # LangChain agent + Tavily + Supabase logging
├── tasks/
│   └── celery_app.py            # Celery tasks: image, video, audio, investigate
├── middleware/
│   └── auth.py                  # X-API-Key FastAPI dependency
└── tests/
    ├── test_api.py              # FastAPI endpoint tests (httpx AsyncClient)
    ├── test_model.py            # Vision + audio model unit tests
    ├── test_agent.py            # Investigation agent unit tests
    └── test_supabase.py         # Supabase CRUD mock tests
```

---

## Quick Start

### 1. Configure environment

```bash
cp .env.template .env
# Edit .env with your real credentials
```

### 2. Set up Supabase

Paste `db/schema.sql` into the [Supabase SQL editor](https://app.supabase.com) and run it.

### 3. Start with Docker Compose

```bash
docker compose up --build
```

| Service | URL |
|---------|-----|
| API docs (Swagger) | http://localhost:8000/docs |
| Gradio UI | http://localhost:7860 |
| Flower (Celery) | http://localhost:5555 |

### 4. Local development (without Docker)

```bash
pip install -r requirements.txt

# Start Redis
redis-server &

# Start Celery worker
celery -A tasks.celery_app worker --loglevel=info &

# Start API
uvicorn main:app --reload

# Start Gradio
python app.py
```

---

## API Reference

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/detect/image` | ✅ | Submit image (JPG/PNG) |
| `POST` | `/detect/video` | ✅ | Submit video (MP4/AVI) |
| `POST` | `/detect/audio` | ✅ | Submit audio (WAV/MP3) |
| `GET`  | `/result/{job_id}` | ✅ | Poll result |
| `GET`  | `/health` | ❌ | Liveness probe |

**Auth**: All protected endpoints require the `X-API-Key` header set to your `API_SECRET_KEY`.

### Example

```bash
# Submit an image
curl -X POST http://localhost:8000/detect/image \
  -H "X-API-Key: your-secret" \
  -F "file=@photo.jpg"
# → {"job_id": "...", "status": "pending", "media_type": "image"}

# Poll result
curl http://localhost:8000/result/<job_id> -H "X-API-Key: your-secret"
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Fine-tuning

```bash
# Prepare dataset:
# data/train/{real,fake}/*.jpg
# data/val/{real,fake}/*.jpg

python training/train.py \
  --data_dir  data/ \
  --epochs    15 \
  --batch     32 \
  --lr        1e-4 \
  --out       models/checkpoints/vision_finetuned.pth

# Evaluate
python training/evaluate.py \
  --data_dir   data/ \
  --checkpoint models/checkpoints/vision_finetuned.pth
```

Recommended datasets:
- Vision: [FaceForensics++](https://github.com/ondyari/FaceForensics)
- Audio: [ASVspoof 2019](https://www.asvspoof.org/)

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SUPABASE_URL` | ✅ | Supabase project URL |
| `SUPABASE_KEY` | ✅ | Supabase anon or service-role key |
| `TAVILY_API_KEY` | ✅ | Tavily web search API key |
| `API_SECRET_KEY` | ✅ | Bearer secret for X-API-Key auth |
| `REDIS_URL` | ✅ | Redis connection string |
| `DEEPFAKE_THRESHOLD` | ❌ | Score threshold 0–1 (default: 0.60) |

---

## CI/CD Setup

To enable the GitHub Actions deployment pipeline, add the following secrets to your repository (**Settings → Secrets and variables → Actions**):

| Secret Name | Description |
|-------------|-------------|
| `DOCKER_USERNAME` | Your Docker Hub username |
| `DOCKER_TOKEN` | Your Docker Hub personal access token |
| `RAILWAY_WEBHOOK_URL` | Webhook URL for Railway redeployment |
| `SLACK_WEBHOOK` | Webhook URL for Slack deployment notifications |

---

---

## Deployment

### 1. Train the model
```bash
python training/train.py --data_dir /path/to/faceforensics --epochs 20
```

### 2. Upload weights to HuggingFace Hub
```bash
python upload_model.py --token YOUR_HF_TOKEN
```

### 3. Deploy API to Railway
- Push to GitHub, connect Railway, set env vars from `.env.template`

### 4. Deploy demo to HuggingFace Spaces
- Create new Space at `huggingface.co/new-space`
- Push this repo (HF Spaces auto-deploys from the repo)
- Add Space secrets: `SUPABASE_URL`, `SUPABASE_KEY`, `API_URL`, `TAVILY_API_KEY`

### 5. Verify deployment
```bash
curl -X GET https://your-railway-url.up.railway.app/health \
  -H 'X-API-Key: your-api-key'
```

---

## License

MIT
