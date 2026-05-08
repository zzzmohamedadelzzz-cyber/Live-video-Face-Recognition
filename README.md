# Face Recognition App — DEBI Hackathon

Real-time face recognition via WebSocket streaming, FastAPI, dlib/face_recognition, and Qdrant Cloud vector DB.

## Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + uvicorn |
| Face detection | face_recognition (dlib HOG) |
| Vector DB | Qdrant Cloud (128-d cosine) |
| Frontend | HTML + Vanilla JS + Canvas |
| Container | Docker + Docker Compose |
| CI/CD | GitHub Actions → Railway |

## Quick start (local, no Docker)

```bash
# 1. Install deps (requires cmake + build-essential for dlib)
pip install -r requirements.txt

# 2. Copy and fill in credentials
cp .env.example .env   # then edit .env

# 3. Run
uvicorn app:app --reload
# Open http://localhost:8000
```

## Quick start (Docker)

```bash
cp .env.example .env   # fill in credentials
docker compose up --build
# Open http://localhost:8000
```

## Environment variables

| Variable | Description | Default |
|---|---|---|
| `QDRANT_URL` | Qdrant cluster URL | — |
| `QDRANT_API_KEY` | Qdrant API key | — |
| `QDRANT_COLLECTION` | Collection name | `DEBI_FACE_RECO` |
| `TOLERANCE` | Cosine score threshold (0–1) | `0.85` |

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `WS` | `/ws/recognize` | Streaming recognition |
| `POST` | `/api/register` | Register a face `{name, frame}` |
| `GET` | `/api/persons` | List registered persons |
| `DELETE` | `/api/persons/{name}` | Delete all embeddings for a person |

## Tests

```bash
pytest tests/ -v   # face_recognition is mocked — no dlib needed
```

## CI/CD

GitHub Actions (`.github/workflows/ci-cd.yml`):
1. **Lint** — `ruff check .`
2. **Test** — `pytest` (mocked face_recognition, no build time)
3. **Docker build & push** → GitHub Container Registry (`ghcr.io`)
4. **Railway** auto-deploys on every push to `main`

## Deploy to Railway

1. Push repo to GitHub
2. [railway.app](https://railway.app) → New Project → Deploy from GitHub Repo
3. Set env vars: `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`, `TOLERANCE`
4. Every `git push main` → GitHub Actions tests → Railway deploys automatically

## Architecture

```
Browser (getUserMedia)
  │  base64 JPEG every 100 ms
  ▼
FastAPI WebSocket /ws/recognize
  │  face_recognition (HOG, 50% scale)
  │  128-d embedding → Qdrant cosine search
  ▼
JSON { faces: [{box, label, score, color}] }
  │
  ▼
Canvas overlay (bounding boxes + labels)
```

## Accuracy tips

- Register **3–5 photos** per person at different angles/lighting
- Threshold `0.85` — lower → more permissive, higher → stricter
- Switch to CNN model in `detector.py` if HOG accuracy is poor (GPU recommended)
