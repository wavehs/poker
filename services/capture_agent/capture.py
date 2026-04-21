"""
Capture Agent — Screen/window capture service.

Phase 1: Loads frames from disk (synthetic or prerecorded).
Phase 2+: Real-time screen capture via mss/dxcam.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class CaptureAgent:
    """
    Captures frames from screen regions or loads from disk.
    
    Phase 1 implementation uses file-based frame loading.
    Phase 2 will add real-time screen capture via mss.
    """

    def __init__(
        self,
        source: str = "file",
        region: Optional[dict[str, int]] = None,
        fps_target: float = 2.0,
    ) -> None:
        """
        Args:
            source: 'file' for disk-based, 'screen' for live capture (Phase 2).
            region: Screen region dict with keys: left, top, width, height.
            fps_target: Target frames per second for capture loop.
        """
        self.source = source
        self.region = region or {"left": 0, "top": 0, "width": 1920, "height": 1080}
        self.fps_target = fps_target
        self._frame_idx = 0

    def capture_frame(self) -> tuple[np.ndarray, dict]:
        """
        Capture a single frame.
        
        Returns:
            Tuple of (frame_bgr, metadata_dict).
        """
        if self.source == "screen":
            return self._capture_screen()
        # Default: return a blank frame (Phase 1 placeholder)
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
        self, directory: str | Path, extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg")
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

    def _capture_screen(self) -> tuple[np.ndarray, dict]:
        """
        Real-time screen capture (Phase 2 stub).
        
        TODO: Implement via mss or dxcam for GPU-accelerated capture.
        """
        try:
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
                # mss returns BGRA, convert to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                meta = {
                    "frame_idx": self._frame_idx,
                    "timestamp_ms": time.time() * 1000,
                    "source": "screen",
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                }
                self._frame_idx += 1
                return frame, meta
        except ImportError:
            # Fallback to blank frame
            return self.capture_frame()
