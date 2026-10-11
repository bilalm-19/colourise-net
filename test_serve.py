"""
API tests. They use randomly initialised weights, so they check the system works
(loading, request handling, output sizes, errors), not colour quality.

Run: pytest
"""
import io
import os
import tempfile

import torch
from fastapi.testclient import TestClient
from PIL import Image

from model import ColouriserUNet

# Save random weights to a temp file and point the server at them before importing it,
# so the tests don't depend on the trained checkpoint
_weights = os.path.join(tempfile.mkdtemp(), "test_model.pt")
torch.save(ColouriserUNet().state_dict(), _weights)
os.environ["MODEL_PATH"] = _weights

import serve  # noqa: E402  (must come after MODEL_PATH is set)


def make_png(size=(64, 48)):
    """Create a small grey test image in memory and return its PNG bytes."""
    buf = io.BytesIO()
    Image.new("L", size, color=128).save(buf, format="PNG")
    return buf.getvalue()


def test_health():
    with TestClient(serve.app) as client:
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["parameters"] > 1_000_000


def test_colourise_side_by_side():
    with TestClient(serve.app) as client:
        r = client.post("/colourise", files={"file": ("x.png", make_png(), "image/png")})
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"
        img = Image.open(io.BytesIO(r.content))
        assert img.mode == "RGB"
        assert img.size == (64 * 2, 48)  # stretched + full-res, side by side


def test_colourise_raw_model_output():
    with TestClient(serve.app) as client:
        r = client.post("/colourise?full_res=false", files={"file": ("x.png", make_png(), "image/png")})
        assert r.status_code == 200
        assert Image.open(io.BytesIO(r.content)).size == (32, 32)


def test_rejects_non_image():
    with TestClient(serve.app) as client:
        r = client.post("/colourise", files={"file": ("x.txt", b"not an image", "text/plain")})
        assert r.status_code == 415


def test_rejects_file_too_large():
    with TestClient(serve.app) as client:
        too_big = b"0" * (serve.MAX_UPLOAD_BYTES + 1)  # 1 byte over the limit
        r = client.post("/colourise", files={"file": ("big.png", too_big, "image/png")})
        assert r.status_code == 413


def test_rejects_image_dimensions_too_large():
    with TestClient(serve.app) as client:
        # A plain 5000x5000 image compresses to a tiny file, but has too many pixels
        r = client.post("/colourise", files={"file": ("huge.png", make_png((5000, 5000)), "image/png")})
        assert r.status_code == 413


def test_home_page():
    with TestClient(serve.app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]