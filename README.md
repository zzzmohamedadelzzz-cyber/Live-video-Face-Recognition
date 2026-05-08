# Live Face Recognition

Real-time face recognition streamed over WebSocket — **ArcFace 512-d embeddings**, **MTCNN detection**, **Qdrant Cloud** vector database, and a **60 FPS canvas UI** decoupled from the server processing loop.

---

## Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| Face detection | MTCNN (deepface 0.0.99) |
| Face embedding | ArcFace — 512-d, 99.6 % LFW accuracy |
| Vector database | Qdrant Cloud — cosine similarity |
| Frontend | Vanilla JS + HTML5 Canvas |
| Container | Docker + Docker Compose |
| CI/CD | GitHub Actions → GitHub Container Registry |

---

## Architecture

```
Browser (getUserMedia)
  │
  │  base64 JPEG frame  (latest snapshot every 3 seconds, after prior reply)
  ▼
FastAPI  /ws/recognize  (WebSocket)
  │
  ├─ every 4th frame ──► MTCNN detect + ArcFace embed (~200 ms)
  │                       512-d vector → Qdrant cosine search
  │                       returns label + confidence
  │
  └─ other frames ──────► MTCNN detect only (~50 ms)
                           reuse last labels, match by box centre
  │
  │  JSON  { faces: [{ box, label, score, color }] }
  ▼
Browser
  ├─ ws.onmessage  → updates lastFaces cache
  │
  └─ requestAnimationFrame loop (60 FPS)
       reads lastFaces, redraws canvas overlay every frame
       → smooth UI regardless of server round-trip time
```

### Two-speed pipeline

| Path | Trigger | What runs | Latency |
|---|---|---|---|
| **Slow** (recognition) | Every 4th frame | MTCNN + ArcFace + Qdrant search | ~200 ms |
| **Fast** (detection only) | All other frames | MTCNN only | ~50 ms |

The fast path keeps boxes on screen between recognition ticks. Labels are carried forward and matched to new boxes by nearest bounding-box centre distance.

### Decoupled render loop

The browser render loop (`requestAnimationFrame`) runs at the display's native refresh rate (typically 60 FPS) and is completely independent of the WebSocket. It simply redraws the last received face data on each animation frame. The server loop sends one latest camera snapshot, waits for the server reply, then samples again after 3 seconds. This prevents stale frames from building up when recognition is slower than the camera.

---

## Quick start (local, no Docker)

```bash
# 1. Python 3.11+ recommended
pip install -r requirements.txt

# 2. Copy and fill in credentials
cp .env.example .env

# 3. Start
uvicorn app:app --reload --host 0.0.0.0 --port 8000
# Open http://localhost:8000
```

> **Windows note:** set `PYTHONUTF8=1` in your `.env` (already in `.env.example`) to prevent deepface's emoji-heavy logger from crashing on cp1252 terminals.

---

## Quick start (Docker)

```bash
cp .env.example .env      # fill in credentials
docker compose up --build
# Open http://localhost:8000
```

---

## Environment variables

| Variable | Description | Default |
|---|---|---|
| `QDRANT_URL` | Qdrant cluster HTTPS URL | required |
| `QDRANT_API_KEY` | Qdrant API key | required |
| `QDRANT_COLLECTION` | Collection name | `DEBI_FACE_RECO` |
| `TOLERANCE` | Cosine similarity threshold (0–1). Higher = stricter match | `0.68` |
| `RECOGNIZE_EVERY` | Run full ArcFace every N frames; fast detection on the rest | `1` |
| `PYTHONUTF8=1` | Force UTF-8 output (required on Windows) | — |

---

## API reference

| Method | Path | Body / Params | Description |
|---|---|---|---|
| `GET` | `/` | — | Serves the web UI |
| `WS` | `/ws/recognize` | `{ frame: "data:image/jpeg;base64,…" }` | Streaming recognition loop |
| `POST` | `/api/register` | `{ name: string, frame: string }` | Detect face in frame and store embedding |
| `GET` | `/api/persons` | — | List all registered person names |
| `DELETE` | `/api/persons/{name}` | — | Delete all embeddings for a person |

### WebSocket message format

**Client → Server**
```json
{ "frame": "data:image/jpeg;base64,/9j/4AAQ..." }
```

**Server → Client**
```json
{
  "faces": [
    {
      "box":   { "top": 80, "right": 320, "bottom": 280, "left": 120 },
      "label": "Alice",
      "score": 0.91,
      "color": "green"
    }
  ]
}
```

`color` is `"green"` (known), `"red"` (unknown), or `"orange"` (awaiting recognition).

---

## Qdrant collection

The collection is created automatically on startup with:

- **Vector size:** 512 (ArcFace)
- **Distance metric:** Cosine
- **Payload index:** `name` field (keyword) — required for filtered deletes on Qdrant Cloud

If the collection already exists with a different vector size (e.g. from a previous FaceNet/128-d setup) it is automatically dropped and recreated.

Multiple embeddings per person are allowed and all contribute to search — registering someone from several angles improves accuracy.

---

## Project structure

```
.
├── app.py                  # FastAPI app, WebSocket handler, REST endpoints
├── core/
│   ├── detector.py         # detect_only() and detect_faces() — MTCNN + ArcFace
│   └── qdrant_store.py     # Qdrant upsert / search / list / delete
├── templates/
│   └── index.html          # Full frontend — camera, canvas overlay, register UI
├── tests/
│   ├── conftest.py         # deepface mocked via sys.modules for CI
│   ├── test_detector.py
│   └── test_qdrant_store.py
├── .env.example
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Tests

```bash
pytest tests/ -v
```

deepface and TensorFlow are mocked via `sys.modules` in `conftest.py` — no GPU or model download needed in CI.

---

## CI/CD

GitHub Actions (`.github/workflows/ci-cd.yml`):

1. **Lint** — `ruff check .`
2. **Test** — `pytest` with mocked deepface (no TF install)
3. **Docker build & push** → GitHub Container Registry (`ghcr.io`)

---

## Accuracy tips

- Register **3–5 photos** per person from different angles and lighting conditions
- `TOLERANCE=0.68` is a good starting point — lower values are stricter (fewer false positives), higher values are more permissive (fewer false negatives)
- ArcFace at 512-d outperforms FaceNet (128-d) significantly on real-world conditions; no config change needed
- MTCNN is robust to partial occlusion and moderate angles — if detection is still missing faces, ensure good frontal lighting
