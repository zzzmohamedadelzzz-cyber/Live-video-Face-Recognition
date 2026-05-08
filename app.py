import asyncio
import json
import logging
import math
import os
import sys
from contextlib import asynccontextmanager

# Force UTF-8 so deepface's emoji-heavy logger can't crash on Windows cp1252.
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from core.detector import decode_frame, detect_faces, detect_only
from core.qdrant_store import QdrantStore

load_dotenv()

store: QdrantStore | None = None

# Run full ArcFace recognition every N frames; use fast detect_only for the rest.
# Lower  → more accurate labels, slower boxes.
# Higher → faster boxes, labels update less often.
RECOGNIZE_EVERY = int(os.getenv("RECOGNIZE_EVERY", "1"))
TOLERANCE       = float(os.getenv("TOLERANCE", "0.68"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store
    store = QdrantStore()
    yield


app = FastAPI(title="Face Recognition API", lifespan=lifespan)


# ── helpers ──────────────────────────────────────────────────────────────────

def _box_center(box: dict) -> tuple[float, float]:
    return ((box["left"] + box["right"]) / 2, (box["top"] + box["bottom"]) / 2)


def _match_boxes_to_labels(
    boxes: list[dict],
    cached: list[dict],
    max_dist: float = 120.0,
) -> list[dict]:
    """Assign cached recognition labels to freshly-detected boxes by nearest centre."""
    results = []
    for box in boxes:
        cx, cy = _box_center(box)
        best, best_d = None, float("inf")
        for c in cached:
            ccx, ccy = _box_center(c["box"])
            d = math.hypot(cx - ccx, cy - ccy)
            if d < best_d:
                best, best_d = c, d
        if best and best_d <= max_dist:
            results.append({"box": box, "label": best["label"],
                            "score": best["score"], "color": best["color"]})
        else:
            # Face newly appeared — show box while waiting for next recognition tick
            results.append({"box": box, "label": "…", "score": 0.0, "color": "orange"})
    return results


# ── routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.websocket("/ws/recognize")
async def recognize(ws: WebSocket):
    await ws.accept()
    frame_count   = 0
    cached_results: list[dict] = []

    try:
        while True:
            data = await ws.receive_text()
            try:
                msg   = json.loads(data)
                frame = decode_frame(msg["frame"])
                frame_count += 1

                if frame_count % RECOGNIZE_EVERY == 0:
                    # ── Slow path: full ArcFace recognition ──────────────────
                    faces = await asyncio.to_thread(detect_faces, frame)
                    results: list[dict] = []
                    for box, encoding in faces:
                        name, score = store.search(encoding)
                        if score >= TOLERANCE:
                            label, color = name, "green"
                        else:
                            label, color = "Unknown", "red"
                        results.append({
                            "box":   box,
                            "label": label,
                            "score": round(score, 3),
                            "color": color,
                        })
                    cached_results = results

                else:
                    # ── Fast path: detection only, reuse last labels ──────────
                    boxes   = await asyncio.to_thread(detect_only, frame)
                    results = _match_boxes_to_labels(boxes, cached_results)

                await ws.send_text(json.dumps({"faces": results}))

            except Exception as exc:
                await ws.send_text(json.dumps({"faces": [], "error": str(exc)}))

    except WebSocketDisconnect:
        pass


class RegisterRequest(BaseModel):
    name: str
    frame: str


@app.post("/api/register")
async def register(req: RegisterRequest):
    frame = decode_frame(req.frame)
    faces = await asyncio.to_thread(detect_faces, frame)
    if not faces:
        return {"success": False, "message": "No face detected in the frame."}
    _, encoding = faces[0]
    store.upsert(encoding, req.name)
    return {"success": True, "message": f"Registered '{req.name}' successfully."}


@app.get("/api/persons")
async def list_persons():
    return {"persons": store.list_persons()}


@app.delete("/api/persons/{name}")
async def delete_person(name: str):
    store.delete_person(name)
    return {"success": True, "message": f"Deleted '{name}'."}
