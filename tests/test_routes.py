"""Regression tests for /analyze-* HTTP boundary protections."""

from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from apps.api import pipeline as pipeline_mod
from apps.api.main import app
from apps.api.routes import MAX_IMAGE_BYTES, MAX_SEQUENCE_LEN


@pytest.fixture
def client():
    pipeline_mod.reset_pipeline()
    return TestClient(app)


def _tiny_png_b64() -> str:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", frame)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def test_analyze_frame_rejects_empty_payload(client):
    r = client.post("/api/v1/analyze-frame", json={"image_base64": ""})
    assert r.status_code == 400


def test_analyze_frame_rejects_oversize_base64(client):
    huge = "A" * (((MAX_IMAGE_BYTES * 4) // 3) + 100)
    r = client.post("/api/v1/analyze-frame", json={"image_base64": huge})
    assert r.status_code == 413


def test_analyze_frame_rejects_invalid_base64(client):
    r = client.post("/api/v1/analyze-frame", json={"image_base64": "not_a_real_image"})
    assert r.status_code == 400


def test_analyze_sequence_rejects_oversize_sequence(client):
    payload = {"images_base64": [_tiny_png_b64()] * (MAX_SEQUENCE_LEN + 1)}
    r = client.post("/api/v1/analyze-sequence", json=payload)
    assert r.status_code == 413


def test_analyze_frame_accepts_tiny_image(client):
    r = client.post("/api/v1/analyze-frame", json={"image_base64": _tiny_png_b64()})
    # The pipeline runs in mock mode here; it must not crash.
    assert r.status_code == 200
    data = r.json()
    assert "table_state" in data
