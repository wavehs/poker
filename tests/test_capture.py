"""Tests for Capture Agent — Phase 2."""

import sys
import time
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from services.capture_agent.capture import CaptureAgent, CaptureMetrics

# ─── CaptureMetrics ─────────────────────────────────────────────────────────


class TestCaptureMetrics:
    def test_initial_state(self):
        m = CaptureMetrics()
        assert m.frames_captured == 0
        assert m.avg_latency_ms == 0.0
        assert m.max_latency_ms == 0.0
        assert m.fps_actual == 0.0

    def test_record_single(self):
        m = CaptureMetrics()
        m.record(10.0)
        assert m.frames_captured == 1
        assert m.avg_latency_ms == 10.0
        assert m.max_latency_ms == 10.0
        assert m.last_latency_ms == 10.0
        assert m.fps_actual == 100.0

    def test_record_multiple(self):
        m = CaptureMetrics()
        m.record(10.0)
        m.record(20.0)
        m.record(30.0)
        assert m.frames_captured == 3
        assert m.avg_latency_ms == 20.0
        assert m.max_latency_ms == 30.0
        assert m.last_latency_ms == 30.0

    def test_report(self):
        m = CaptureMetrics()
        m.record(15.0)
        r = m.report()
        assert r["frames_captured"] == 1
        assert r["avg_latency_ms"] == 15.0
        assert "fps_actual" in r


# ─── CaptureAgent — File mode ───────────────────────────────────────────────


class TestCaptureAgentFileMode:
    def test_file_backend_returns_blank(self):
        agent = CaptureAgent(source="file")
        assert agent.backend == "file"
        frame, meta = agent.capture_frame()
        assert frame.shape == (1080, 1920, 3)
        assert meta["source"] == "blank"
        assert meta["backend"] == "file"
        assert meta["capture_latency_ms"] >= 0

    def test_custom_region(self):
        region = {"left": 0, "top": 0, "width": 800, "height": 600}
        agent = CaptureAgent(source="file", region=region)
        frame, meta = agent.capture_frame()
        assert frame.shape == (600, 800, 3)

    def test_frame_idx_increments(self):
        agent = CaptureAgent(source="file")
        _, m1 = agent.capture_frame()
        _, m2 = agent.capture_frame()
        assert m1["frame_idx"] == 0
        assert m2["frame_idx"] == 1

    def test_metrics_updated(self):
        agent = CaptureAgent(source="file")
        agent.capture_frame()
        agent.capture_frame()
        assert agent.metrics.frames_captured == 2
        assert agent.metrics.avg_latency_ms > 0

    def test_context_manager(self):
        with CaptureAgent(source="file") as agent:
            frame, meta = agent.capture_frame()
            assert frame is not None

    def test_capture_continuous_max_frames(self):
        agent = CaptureAgent(source="file", fps_target=100)
        frames = list(agent.capture_continuous(max_frames=5))
        assert len(frames) == 5

    def test_capture_continuous_duration(self):
        agent = CaptureAgent(source="file", fps_target=100)
        t0 = time.perf_counter()
        frames = list(agent.capture_continuous(duration_s=0.2))
        elapsed = time.perf_counter() - t0
        assert elapsed >= 0.15  # Allow some slack
        assert len(frames) >= 1


# ─── CaptureAgent — Load from file ──────────────────────────────────────────


class TestCaptureAgentLoadFile:
    def test_load_frame(self, tmp_path):
        # Create a test image
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        path = tmp_path / "test.png"
        cv2.imwrite(str(path), img)

        agent = CaptureAgent(source="file")
        frame, meta = agent.load_frame(path)
        assert frame.shape == (100, 100, 3)
        assert meta["source"] == str(path)

    def test_load_frame_not_found(self):
        agent = CaptureAgent(source="file")
        with pytest.raises(FileNotFoundError):
            agent.load_frame("nonexistent.png")

    def test_load_frames_from_directory(self, tmp_path):
        for i in range(3):
            img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f"frame_{i:03d}.png"), img)

        agent = CaptureAgent(source="file")
        frames = agent.load_frames_from_directory(tmp_path)
        assert len(frames) == 3

    def test_load_frames_empty_directory(self, tmp_path):
        agent = CaptureAgent(source="file")
        frames = agent.load_frames_from_directory(tmp_path)
        assert len(frames) == 0

    def test_load_frames_nonexistent_directory(self):
        agent = CaptureAgent(source="file")
        frames = agent.load_frames_from_directory("nonexistent_dir")
        assert len(frames) == 0


# ─── Backend resolution ─────────────────────────────────────────────────────


class TestBackendResolution:
    def test_file_always_available(self):
        agent = CaptureAgent(source="file")
        assert agent.backend == "file"

    def test_screen_resolves_to_real_backend(self):
        """'screen' backward compat should resolve to best available."""
        agent = CaptureAgent(source="screen")
        assert agent.backend in ("dxcam", "mss", "file")

    def test_auto_resolves(self):
        agent = CaptureAgent(source="auto")
        assert agent.backend in ("dxcam", "mss", "file")

    @pytest.mark.skipif(sys.platform != "win32", reason="DXcam is Windows-only")
    def test_dxcam_fallback_if_unavailable(self):
        """If DXcam import fails, should fallback to MSS."""
        with patch("services.capture_agent.capture._DXCAM_AVAILABLE", False):
            agent = CaptureAgent(source="dxcam")
            assert agent.backend in ("mss", "file")


# ─── MSS backend ────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not CaptureAgent(source="mss").backend == "mss",
    reason="MSS not available",
)
class TestMSSCapture:
    def test_mss_captures_real_frame(self):
        region = {"left": 0, "top": 0, "width": 100, "height": 100}
        agent = CaptureAgent(source="mss", region=region)
        frame, meta = agent.capture_frame()

        assert frame.shape == (100, 100, 3)
        assert meta["source"] == "mss"
        assert meta["backend"] == "mss"
        assert not np.all(frame == 0)  # Should capture something real

    def test_mss_latency_reasonable(self):
        region = {"left": 0, "top": 0, "width": 200, "height": 200}
        agent = CaptureAgent(source="mss", region=region)
        _, meta = agent.capture_frame()
        # MSS should capture within 200ms on any machine
        assert meta["capture_latency_ms"] < 200
