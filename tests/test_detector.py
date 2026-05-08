import base64
import io
import sys

import numpy as np
import pytest
from PIL import Image

EMBED_DIM = 512  # ArcFace


def _make_b64(w: int = 100, h: int = 100) -> str:
    img = Image.new("RGB", (w, h), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _repr_mock():
    return sys.modules["deepface.modules.representation"]


# ── decode_frame ─────────────────────────────────────────────────────────────

def test_decode_frame_plain_b64():
    from core.detector import decode_frame

    frame = decode_frame(_make_b64(80, 60))
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (60, 80, 3)


def test_decode_frame_data_url():
    from core.detector import decode_frame

    b64 = "data:image/jpeg;base64," + _make_b64(40, 30)
    frame = decode_frame(b64)
    assert frame.shape == (30, 40, 3)


# ── detect_faces ─────────────────────────────────────────────────────────────

def test_detect_faces_no_faces():
    _repr_mock().represent.return_value = []

    from core.detector import detect_faces

    assert detect_faces(np.zeros((480, 640, 3), dtype=np.uint8)) == []


def test_detect_faces_one_face():
    _repr_mock().represent.return_value = [
        {
            "embedding": [0.1] * EMBED_DIM,
            "facial_area": {"x": 50, "y": 100, "w": 60, "h": 70},
            "face_confidence": 0.98,
        }
    ]

    from core.detector import detect_faces

    result = detect_faces(np.zeros((480, 640, 3), dtype=np.uint8))

    assert len(result) == 1
    box, embedding = result[0]
    assert box == {"top": 100, "right": 110, "bottom": 170, "left": 50}
    assert len(embedding) == EMBED_DIM


def test_detect_faces_exception_returns_empty():
    _repr_mock().represent.side_effect = ValueError("no face")

    from core.detector import detect_faces

    assert detect_faces(np.zeros((480, 640, 3), dtype=np.uint8)) == []

    _repr_mock().represent.side_effect = None
    _repr_mock().represent.return_value = []


def test_detect_faces_returns_correct_types():
    _repr_mock().represent.return_value = [
        {
            "embedding": list(range(EMBED_DIM)),
            "facial_area": {"x": 10, "y": 20, "w": 30, "h": 40},
            "face_confidence": 0.9,
        }
    ]

    from core.detector import detect_faces

    result = detect_faces(np.zeros((200, 200, 3), dtype=np.uint8))
    box, enc = result[0]
    assert all(k in box for k in ("top", "right", "bottom", "left"))
    assert isinstance(enc, list)
