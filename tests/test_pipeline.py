"""Tests for end-to-end pipeline."""

import numpy as np
import pytest

from apps.api.pipeline import Pipeline
from libs.common.schemas import FrameAnalysis


class TestPipeline:
    def test_analyze_blank_frame(self):
        """Full pipeline should work on a blank frame."""
        pipeline = Pipeline()
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = pipeline.analyze_frame(frame, frame_idx=0)

        assert isinstance(result, FrameAnalysis)
        assert result.frame_idx == 0
        assert result.processing_time_ms > 0

    def test_analyze_produces_detections(self):
        """Mock detector should produce detections."""
        pipeline = Pipeline()
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = pipeline.analyze_frame(frame)

        assert len(result.detections) > 0, "Mock detector should produce detections"

    def test_analyze_produces_recommendation(self):
        """Pipeline should produce a recommendation."""
        pipeline = Pipeline()
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = pipeline.analyze_frame(frame)

        rec = result.recommendation
        assert rec is not None
        assert rec.best_action is not None

    def test_analyze_sequence(self):
        """Sequence analysis should process multiple frames."""
        pipeline = Pipeline()
        frames = [np.zeros((1080, 1920, 3), dtype=np.uint8) for _ in range(3)]
        results = pipeline.analyze_sequence(frames)

        assert len(results) == 3
        for i, r in enumerate(results):
            assert r.frame_idx == i

    def test_pipeline_returns_valid_json(self):
        """Result should be serializable to JSON."""
        pipeline = Pipeline()
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = pipeline.analyze_frame(frame)

        json_str = result.model_dump_json()
        assert len(json_str) > 100  # Should be substantial JSON

    def test_confidence_report_present(self):
        """Confidence report should be filled in."""
        pipeline = Pipeline()
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = pipeline.analyze_frame(frame)

        conf = result.recommendation.confidence
        assert conf.vision_confidence >= 0
        assert conf.ocr_confidence >= 0
        assert conf.state_confidence >= 0


class TestPipelineAPI:
    """Tests using the FastAPI test client."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from apps.api.main import app
        return TestClient(app)

    def test_health(self, client):
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"

    def test_analyze_synthetic(self, client):
        res = client.post("/api/v1/analyze-synthetic")
        assert res.status_code == 200
        data = res.json()
        assert "recommendation" in data
        assert "table_state" in data
        assert "detections" in data
        assert data["processing_time_ms"] > 0

    def test_analyze_frame_base64(self, client):
        """Test base64 frame upload."""
        import base64

        import cv2

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode(".png", frame)
        b64 = base64.b64encode(buffer).decode("utf-8")

        res = client.post(
            "/api/v1/analyze-frame",
            json={"image_base64": b64, "frame_idx": 0},
        )
        assert res.status_code == 200
        data = res.json()
        assert "recommendation" in data
