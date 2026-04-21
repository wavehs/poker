"""
OCR Backends — Abstract OCR backend protocol and concrete implementations.

Provides:
- OCRBackend (Protocol) — common interface
- PaddleOCRBackend — primary, GPU-first
- EasyOCRBackend — fallback
- MockOCRBackend — testing
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


# ─── Protocol ────────────────────────────────────────────────────────────────


@runtime_checkable
class OCRBackend(Protocol):
    """Common interface for all OCR backends."""

    @property
    def name(self) -> str:
        """Backend identifier."""
        ...

    @property
    def is_available(self) -> bool:
        """Whether the backend is ready for use."""
        ...

    def recognize(self, crop: np.ndarray) -> list[tuple[str, float, list]]:
        """
        Recognize text in a pre-cropped image region.

        Args:
            crop: BGR image crop (numpy array).

        Returns:
            List of (text, confidence, bbox_points) tuples.
            bbox_points is a list of 4 corner points [[x1,y1], ...].
        """
        ...


# ─── PaddleOCR Backend ──────────────────────────────────────────────────────


class PaddleOCRBackend:
    """PaddleOCR-based text recognition. GPU-first, lazy initialization."""

    def __init__(self, use_gpu: bool = False, lang: str = "en") -> None:
        self._use_gpu = use_gpu
        self._lang = lang
        self._engine = None
        self._init_attempted = False
        self._available = False

    @property
    def name(self) -> str:
        return "paddle"

    @property
    def is_available(self) -> bool:
        if not self._init_attempted:
            self._try_init()
        return self._available

    def _try_init(self) -> None:
        """Lazy-initialize PaddleOCR engine."""
        self._init_attempted = True
        try:
            from paddleocr import PaddleOCR

            self._engine = PaddleOCR(
                use_angle_cls=False,
                lang=self._lang,
                use_gpu=self._use_gpu,
                show_log=False,
                # Optimize for speed on small crops
                det_db_thresh=0.3,
                rec_batch_num=1,
            )
            self._available = True
            logger.info("PaddleOCR initialized (gpu=%s, lang=%s)", self._use_gpu, self._lang)
        except Exception as e:
            logger.warning("PaddleOCR init failed: %s", e)
            self._available = False

    def recognize(self, crop: np.ndarray) -> list[tuple[str, float, list]]:
        if not self.is_available or self._engine is None:
            return []

        try:
            results = self._engine.ocr(crop, cls=False)
            if not results or results[0] is None:
                return []

            parsed: list[tuple[str, float, list]] = []
            for line in results[0]:
                bbox_points = line[0]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                text = line[1][0]
                confidence = float(line[1][1])
                parsed.append((text, confidence, bbox_points))

            return parsed

        except Exception as e:
            logger.warning("PaddleOCR recognize failed: %s", e)
            return []


# ─── EasyOCR Backend ─────────────────────────────────────────────────────────


class EasyOCRBackend:
    """EasyOCR-based text recognition. Fallback option."""

    def __init__(self, use_gpu: bool = False, lang: list[str] | None = None) -> None:
        self._use_gpu = use_gpu
        self._lang = lang or ["en"]
        self._reader = None
        self._init_attempted = False
        self._available = False

    @property
    def name(self) -> str:
        return "easyocr"

    @property
    def is_available(self) -> bool:
        if not self._init_attempted:
            self._try_init()
        return self._available

    def _try_init(self) -> None:
        """Lazy-initialize EasyOCR reader."""
        self._init_attempted = True
        try:
            import easyocr

            self._reader = easyocr.Reader(
                self._lang,
                gpu=self._use_gpu,
                verbose=False,
            )
            self._available = True
            logger.info("EasyOCR initialized (gpu=%s)", self._use_gpu)
        except Exception as e:
            logger.warning("EasyOCR init failed: %s", e)
            self._available = False

    def recognize(self, crop: np.ndarray) -> list[tuple[str, float, list]]:
        if not self.is_available or self._reader is None:
            return []

        try:
            results = self._reader.readtext(crop)

            parsed: list[tuple[str, float, list]] = []
            for bbox, text, confidence in results:
                parsed.append((text, float(confidence), bbox))

            return parsed

        except Exception as e:
            logger.warning("EasyOCR recognize failed: %s", e)
            return []


# ─── Mock Backend ────────────────────────────────────────────────────────────


class MockOCRBackend:
    """Mock OCR backend for testing. Returns empty results for real images."""

    def __init__(self) -> None:
        self._available = True

    @property
    def name(self) -> str:
        return "mock"

    @property
    def is_available(self) -> bool:
        return self._available

    def recognize(self, crop: np.ndarray) -> list[tuple[str, float, list]]:
        """Mock always returns empty — OCREngine handles mock logic via labels."""
        return []


# ─── Factory ─────────────────────────────────────────────────────────────────


def create_backend(
    name: str = "auto",
    use_gpu: bool = False,
    lang: str = "en",
) -> OCRBackend:
    """
    Create an OCR backend by name, with automatic fallback.

    Args:
        name: Backend name ('auto', 'paddle', 'easyocr', 'mock').
        use_gpu: Whether to use GPU acceleration.
        lang: Language code.

    Returns:
        An initialized OCRBackend instance.
    """
    if name == "mock":
        return MockOCRBackend()

    if name == "paddle":
        b = PaddleOCRBackend(use_gpu=use_gpu, lang=lang)
        if b.is_available:
            return b
        logger.warning("PaddleOCR unavailable, falling back to EasyOCR")
        name = "easyocr"

    if name == "easyocr":
        b = EasyOCRBackend(use_gpu=use_gpu, lang=[lang])
        if b.is_available:
            return b
        logger.warning("EasyOCR unavailable, falling back to Mock")
        return MockOCRBackend()

    if name == "auto":
        # Try PaddleOCR first, then EasyOCR, then Mock
        paddle = PaddleOCRBackend(use_gpu=use_gpu, lang=lang)
        if paddle.is_available:
            return paddle

        easy = EasyOCRBackend(use_gpu=use_gpu, lang=[lang])
        if easy.is_available:
            return easy

        logger.info("No real OCR backend available, using Mock")
        return MockOCRBackend()

    logger.warning("Unknown backend '%s', using Mock", name)
    return MockOCRBackend()
