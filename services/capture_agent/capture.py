"""
Capture Agent — Real-time screen capture service.

Phase 2: GPU-accelerated via DXcam (Windows), MSS fallback (cross-platform).
Maintains file-based loading for testing and replay.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ─── Backend availability detection ──────────────────────────────────────────

_DXCAM_AVAILABLE = False
_MSS_AVAILABLE = False

if sys.platform == "win32":
    try:
        import dxcam as _dxcam_mod  # noqa: F401

        _DXCAM_AVAILABLE = True
    except ImportError:
        pass

try:
    import mss as _mss_mod  # noqa: F401

    _MSS_AVAILABLE = True
except ImportError:
    pass


# ─── Capture metadata ───────────────────────────────────────────────────────


class CaptureMetrics:
    """Accumulates capture performance metrics."""

    __slots__ = (
        "frames_captured",
        "_latency_sum",
        "_latency_max",
        "_last_latency",
    )

    def __init__(self) -> None:
        self.frames_captured: int = 0
        self._latency_sum: float = 0.0
        self._latency_max: float = 0.0
        self._last_latency: float = 0.0

    def record(self, latency_ms: float) -> None:
        self.frames_captured += 1
        self._latency_sum += latency_ms
        self._last_latency = latency_ms
        if latency_ms > self._latency_max:
            self._latency_max = latency_ms

    @property
    def avg_latency_ms(self) -> float:
        if self.frames_captured == 0:
            return 0.0
        return self._latency_sum / self.frames_captured

    @property
    def max_latency_ms(self) -> float:
        return self._latency_max

    @property
    def last_latency_ms(self) -> float:
        return self._last_latency

    @property
    def fps_actual(self) -> float:
        if self._last_latency <= 0:
            return 0.0
        return 1000.0 / self._last_latency

    def report(self) -> dict:
        return {
            "frames_captured": self.frames_captured,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
            "last_latency_ms": round(self.last_latency_ms, 2),
            "fps_actual": round(self.fps_actual, 1),
        }


# ─── CaptureAgent ────────────────────────────────────────────────────────────


class CaptureAgent:
    """
    Captures frames from screen regions or loads from disk.

    Backend priority (auto-select):
        1. dxcam  — GPU-accelerated, Windows-only, lowest latency
        2. mss    — cross-platform, CPU-based, ~20-30ms per frame
        3. file   — load from disk (synthetic / replay)

    Usage:
        agent = CaptureAgent(source="auto")
        frame, meta = agent.capture_frame()

        # or as context manager for resource cleanup:
        with CaptureAgent(source="dxcam") as agent:
            for frame, meta in agent.capture_continuous(max_frames=100):
                process(frame)
    """

    BACKENDS = ("dxcam", "mss", "file")

    def __init__(
        self,
        source: str = "auto",
        region: dict[str, int] | None = None,
        fps_target: float = 5.0,
    ) -> None:
        """
        Args:
            source: Backend name or 'auto' for best available.
                    Options: 'auto', 'dxcam', 'mss', 'file'.
            region: Screen region dict with keys: left, top, width, height.
            fps_target: Target frames per second for capture loop.
        """
        self.region = region or {"left": 0, "top": 0, "width": 1920, "height": 1080}
        self.fps_target = fps_target
        self._frame_idx = 0
        self.metrics = CaptureMetrics()

        # Resolve backend
        self._backend = self._resolve_backend(source)
        self._dxcam_camera = None  # lazy init

        logger.info("CaptureAgent initialized: backend=%s, region=%s", self._backend, self.region)

    def _resolve_backend(self, source: str) -> str:
        """Resolve requested backend to available one."""
        if source == "auto":
            if _DXCAM_AVAILABLE:
                return "dxcam"
            if _MSS_AVAILABLE:
                return "mss"
            return "file"

        if source == "dxcam":
            if not _DXCAM_AVAILABLE:
                logger.warning("DXcam unavailable, falling back to mss")
                return "mss" if _MSS_AVAILABLE else "file"
            return "dxcam"

        if source == "mss":
            if not _MSS_AVAILABLE:
                logger.warning("MSS unavailable, falling back to file")
                return "file"
            return "mss"

        if source == "screen":
            # Backward compat: 'screen' → auto-select best screen backend
            if _DXCAM_AVAILABLE:
                return "dxcam"
            if _MSS_AVAILABLE:
                return "mss"
            return "file"

        return "file"

    @property
    def backend(self) -> str:
        """Currently active backend name."""
        return self._backend

    # ─── Context manager ─────────────────────────────────────────────────

    def __enter__(self) -> CaptureAgent:
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def release(self) -> None:
        """Release backend resources."""
        if self._dxcam_camera is not None:
            try:
                self._dxcam_camera.stop()
            except Exception:
                pass
            self._dxcam_camera = None
            logger.info("DXcam camera released")

    # ─── Core capture ────────────────────────────────────────────────────

    def capture_frame(self) -> tuple[np.ndarray, dict]:
        """
        Capture a single frame.

        Returns:
            Tuple of (frame_bgr, metadata_dict).
        """
        t0 = time.perf_counter()

        if self._backend == "dxcam":
            frame, meta = self._capture_dxcam()
        elif self._backend == "mss":
            frame, meta = self._capture_mss()
        else:
            # file mode: return blank frame
            frame, meta = self._capture_blank()

        latency = (time.perf_counter() - t0) * 1000
        meta["capture_latency_ms"] = round(latency, 2)
        meta["backend"] = self._backend
        self.metrics.record(latency)

        return frame, meta

    def capture_continuous(
        self,
        max_frames: int = 0,
        duration_s: float = 0.0,
    ) -> Iterator[tuple[np.ndarray, dict]]:
        """
        Generator for continuous frame capture.

        Args:
            max_frames: Stop after N frames (0 = unlimited).
            duration_s: Stop after N seconds (0 = unlimited).

        Yields:
            (frame_bgr, metadata_dict) tuples at target FPS.
        """
        frame_interval = 1.0 / self.fps_target if self.fps_target > 0 else 0
        start_time = time.perf_counter()
        count = 0

        while True:
            t_frame_start = time.perf_counter()

            frame, meta = self.capture_frame()
            yield frame, meta

            count += 1
            if max_frames > 0 and count >= max_frames:
                break
            if duration_s > 0 and (time.perf_counter() - start_time) >= duration_s:
                break

            # Throttle to target FPS
            elapsed = time.perf_counter() - t_frame_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ─── File-based loading (unchanged from Phase 1) ─────────────────────

    def load_frame(self, path: str | Path) -> tuple[np.ndarray, dict]:
        """
        Load a frame from an image file.

        Args:
            path: Path to image file (PNG, JPG).

        Returns:
            Tuple of (frame_bgr, metadata_dict).

        Raises:
            FileNotFoundError: If the image file doesn't exist.
            ValueError: If the image cannot be decoded.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Frame not found: {path}")

        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Cannot decode image: {path}")

        meta = {
            "frame_idx": self._frame_idx,
            "timestamp_ms": time.time() * 1000,
            "source": str(path),
            "width": frame.shape[1],
            "height": frame.shape[0],
        }
        self._frame_idx += 1
        return frame, meta

    def load_frames_from_directory(
        self,
        directory: str | Path,
        extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    ) -> list[tuple[np.ndarray, dict]]:
        """
        Load all frames from a directory.

        Args:
            directory: Path to directory containing frame images.
            extensions: File extensions to include.

        Returns:
            List of (frame_bgr, metadata_dict) tuples, sorted by filename.
        """
        directory = Path(directory)
        if not directory.exists():
            return []

        files = sorted(
            f for f in directory.iterdir()
            if f.suffix.lower() in extensions
        )

        frames = []
        for f in files:
            try:
                frames.append(self.load_frame(f))
            except (ValueError, FileNotFoundError):
                continue
        return frames

    # ─── DXcam backend ───────────────────────────────────────────────────

    def _get_dxcam_camera(self):
        """Lazy-initialize DXcam camera."""
        if self._dxcam_camera is None:
            import dxcam

            self._dxcam_camera = dxcam.create(output_color="BGR")
            logger.info("DXcam camera created")
        return self._dxcam_camera

    def _capture_dxcam(self) -> tuple[np.ndarray, dict]:
        """GPU-accelerated screen capture via DXcam."""
        try:
            camera = self._get_dxcam_camera()

            left = self.region["left"]
            top = self.region["top"]
            right = left + self.region["width"]
            bottom = top + self.region["height"]

            frame = camera.grab(region=(left, top, right, bottom))

            if frame is None:
                # DXcam sometimes returns None on first call
                time.sleep(0.05)
                frame = camera.grab(region=(left, top, right, bottom))

            if frame is None:
                logger.warning("DXcam grab returned None, falling back to MSS")
                return self._capture_mss_or_blank()

            meta = {
                "frame_idx": self._frame_idx,
                "timestamp_ms": time.time() * 1000,
                "source": "dxcam",
                "width": frame.shape[1],
                "height": frame.shape[0],
            }
            self._frame_idx += 1
            return frame, meta

        except Exception as e:
            logger.warning("DXcam capture failed: %s, falling back", e)
            return self._capture_mss_or_blank()

    # ─── MSS backend ────────────────────────────────────────────────────

    def _capture_mss(self) -> tuple[np.ndarray, dict]:
        """Cross-platform screen capture via MSS."""
        import mss

        with mss.mss() as sct:
            monitor = {
                "left": self.region["left"],
                "top": self.region["top"],
                "width": self.region["width"],
                "height": self.region["height"],
            }
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            # MSS returns BGRA, convert to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        meta = {
            "frame_idx": self._frame_idx,
            "timestamp_ms": time.time() * 1000,
            "source": "mss",
            "width": frame.shape[1],
            "height": frame.shape[0],
        }
        self._frame_idx += 1
        return frame, meta

    # ─── Blank / fallback ────────────────────────────────────────────────

    def _capture_blank(self) -> tuple[np.ndarray, dict]:
        """Return a blank frame (file mode or last-resort fallback)."""
        frame = np.zeros(
            (self.region["height"], self.region["width"], 3), dtype=np.uint8
        )
        meta = {
            "frame_idx": self._frame_idx,
            "timestamp_ms": time.time() * 1000,
            "source": "blank",
            "width": self.region["width"],
            "height": self.region["height"],
        }
        self._frame_idx += 1
        return frame, meta

    def _capture_mss_or_blank(self) -> tuple[np.ndarray, dict]:
        """Try MSS, then blank."""
        if _MSS_AVAILABLE:
            try:
                return self._capture_mss()
            except Exception as e:
                logger.warning("MSS fallback also failed: %s", e)
        return self._capture_blank()
