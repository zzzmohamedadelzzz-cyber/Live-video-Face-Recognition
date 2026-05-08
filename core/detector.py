import base64
import io
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_MODEL_NAME = "ArcFace"   # 512-d, 99.6% LFW accuracy
_DETECTOR   = "mtcnn"     # CNN-based, robust to lighting / angles


def decode_frame(b64_string: str) -> np.ndarray:
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    img_bytes = base64.b64decode(b64_string)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img)


def detect_only(frame: np.ndarray) -> list[dict]:
    """Fast path: MTCNN face detection only (~50 ms). No embedding."""
    from deepface.modules.detection import extract_faces  # lazy import

    bgr = frame[:, :, ::-1].copy()
    try:
        faces = extract_faces(
            img_path=bgr,
            detector_backend=_DETECTOR,
            enforce_detection=True,
            align=False,            # skip alignment — saves ~10 ms
        )
        return [
            {
                "top":    f["facial_area"]["y"],
                "right":  f["facial_area"]["x"] + f["facial_area"]["w"],
                "bottom": f["facial_area"]["y"] + f["facial_area"]["h"],
                "left":   f["facial_area"]["x"],
            }
            for f in faces
        ]
    except Exception as exc:
        logger.debug("detect_only: %s", exc)
        return []


def detect_faces(frame: np.ndarray, scale: float = 0.5) -> list[tuple[dict, list[float]]]:
    """Slow path: MTCNN + ArcFace (~200 ms). Returns boxes + 512-d embeddings."""
    from deepface.modules.representation import represent  # lazy import

    bgr = frame[:, :, ::-1].copy()
    try:
        results = represent(
            img_path=bgr,
            model_name=_MODEL_NAME,
            detector_backend=_DETECTOR,
            enforce_detection=True,
            align=True,
        )
    except Exception as exc:
        logger.debug("detect_faces: %s", exc)
        return []

    faces = []
    for r in results:
        fa = r["facial_area"]
        box = {
            "top":    fa["y"],
            "right":  fa["x"] + fa["w"],
            "bottom": fa["y"] + fa["h"],
            "left":   fa["x"],
        }
        faces.append((box, r["embedding"]))

    return faces
