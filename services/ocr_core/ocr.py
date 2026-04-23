"""
OCR Core — Text recognition abstraction layer.

Phase 2: Real OCR via PaddleOCR / EasyOCR with automatic fallback.
Mock mode preserved for testing and when no backend is available.
"""

from __future__ import annotations

import logging
import re
import time

import numpy as np

from libs.common.schemas import BoundingBox, Detection, DetectionClass, OCRResult
from services.ocr_core.backends import OCRBackend, create_backend
from services.ocr_core.preprocess import crop_bbox, preprocess_fallback, preprocess_for_ocr

logger = logging.getLogger(__name__)


class OCREngine:
    """
    OCR engine that extracts text from frame regions.

    Backend priority: PaddleOCR → EasyOCR → Mock (auto-fallback).
    Preserves Phase 1 interface: extract(), extract_region().
    """

    # Detection classes that contain readable text
    TEXT_CLASSES = {
        DetectionClass.POT,
        DetectionClass.BET_AMOUNT,
        DetectionClass.PLAYER_PANEL,
        DetectionClass.CHIP_STACK,
    }

    def __init__(
        self,
        backend: str = "mock",
        confidence_boost: float = 0.0,
        use_gpu: bool = False,
    ) -> None:
        """
        Args:
            backend: OCR backend name ('auto', 'paddle', 'easyocr', 'mock').
            confidence_boost: Added to base confidence for tuning.
            use_gpu: Whether to use GPU for OCR inference.
        """
        self.backend = backend
        self.confidence_boost = confidence_boost
        self._backend_impl: OCRBackend = create_backend(backend, use_gpu=use_gpu)
        self._use_real_ocr = self._backend_impl.name != "mock"

        logger.info("OCREngine initialized: backend=%s (resolved=%s)",
                     backend, self._backend_impl.name)

    @property
    def backend_name(self) -> str:
        """Resolved backend name."""
        return self._backend_impl.name

    def extract(
        self,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> list[OCRResult]:
        """
        Run OCR on text-bearing regions found in detections.

        Args:
            frame: BGR frame as numpy array.
            detections: Detections from vision model.

        Returns:
            List of OCRResult for text-bearing detections.
        """
        results: list[OCRResult] = []

        for det in detections:
            if det.detection_class not in self.TEXT_CLASSES:
                continue

            if self._use_real_ocr:
                ocr = self._real_ocr(frame, det)
            else:
                ocr = self._mock_ocr(det)

            if ocr is not None:
                results.append(ocr)

        return results

    def extract_region(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
        field_type: str = "generic",
    ) -> OCRResult | None:
        """
        Run OCR on a specific region.

        Args:
            frame: BGR frame.
            bbox: Region to OCR.
            field_type: Semantic type for the result.

        Returns:
            OCRResult or None if nothing readable.
        """
        if self._use_real_ocr:
            return self._real_ocr_region(frame, bbox, field_type)

        return OCRResult(
            text="0",
            confidence=0.5,
            bbox=bbox,
            field_type=field_type,
        )

    # ─── Real OCR ────────────────────────────────────────────────────────

    def _run_ocr_pipeline(
        self, crop: np.ndarray, field_type: str, bbox: BoundingBox
    ) -> OCRResult | None:
        """Helper to run the OCR pipeline with fallback and validation."""
        processed = preprocess_for_ocr(crop, field_type=field_type)

        t0 = time.perf_counter()
        raw_results = self._backend_impl.recognize(processed)
        latency_ms = (time.perf_counter() - t0) * 1000

        if not raw_results:
            raw_results = self._backend_impl.recognize(crop)

        best_text = ""
        best_conf = 0.0
        if raw_results:
            best_text, best_conf, _ = max(raw_results, key=lambda r: r[1])
            best_text = best_text.strip()

        # Fallback for low confidence
        if best_conf < 0.7:
            fallback_processed = preprocess_fallback(crop)
            fallback_results = self._backend_impl.recognize(fallback_processed)
            if fallback_results:
                fb_text, fb_conf, _ = max(fallback_results, key=lambda r: r[1])
                fb_text = fb_text.strip()
                if fb_conf > best_conf:
                    best_text = fb_text
                    best_conf = fb_conf
                    logger.debug("OCR fallback used for [%s]: conf improved to %.2f", field_type, fb_conf)

        if not best_text:
            return None

        if field_type in ("pot", "stack", "bet", "blind"):
            best_text = self._clean_numeric(best_text)

            # Validation
            try:
                val = float(best_text)
                if not (0 <= val <= 1000000):
                    logger.warning("OCR validation failed: [%s] value %s out of bounds", field_type, best_text)
                    return None
            except ValueError:
                logger.warning("OCR validation failed: [%s] value '%s' is not numeric", field_type, best_text)
                return None

        confidence = min(1.0, best_conf + self.confidence_boost)

        logger.debug("OCR [%s]: '%s' (conf=%.2f, %.1fms)",
                     field_type, best_text, confidence, latency_ms)

        return OCRResult(
            text=best_text,
            confidence=confidence,
            bbox=bbox,
            field_type=field_type,
        )

    def _real_ocr(self, frame: np.ndarray, detection: Detection) -> OCRResult | None:
        """Real OCR on a detection region."""
        bbox = detection.bbox
        crop = crop_bbox(frame, bbox.x, bbox.y, bbox.w, bbox.h, padding=0.1)
        if crop.size < 10:
            return None

        field_type = self._field_type_from_detection(detection)
        return self._run_ocr_pipeline(crop, field_type, bbox)

    def _real_ocr_region(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
        field_type: str,
    ) -> OCRResult | None:
        """Real OCR on an arbitrary region."""
        crop = crop_bbox(frame, bbox.x, bbox.y, bbox.w, bbox.h, padding=0.05)
        if crop.size < 10:
            return None

        return self._run_ocr_pipeline(crop, field_type, bbox)

    # ─── Mock OCR (Phase 1 behavior) ─────────────────────────────────────

    def _mock_ocr(self, detection: Detection) -> OCRResult | None:
        """
        Generate mock OCR result from detection label.
        """
        text = detection.label.strip()
        if not text:
            return None

        field_type = self._field_type_from_detection(detection)

        # Clean up text: extract numeric value
        cleaned = self._clean_numeric(text)

        confidence = min(
            1.0,
            detection.bbox.confidence * 0.95 + self.confidence_boost
        )

        return OCRResult(
            text=cleaned,
            confidence=confidence,
            bbox=detection.bbox,
            field_type=field_type,
        )

    # ─── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _field_type_from_detection(detection: Detection) -> str:
        """Map detection class to OCR field type."""
        field_type_map = {
            DetectionClass.POT: "pot",
            DetectionClass.BET_AMOUNT: "bet",
            DetectionClass.PLAYER_PANEL: "stack",
            DetectionClass.CHIP_STACK: "stack",
        }
        return field_type_map.get(detection.detection_class, "generic")

    @staticmethod
    def _clean_numeric(text: str) -> str:
        """
        Clean text to extract numeric value.
        Handles formats like '$1,500', '1.5K', '500BB', etc.
        """
        text = text.replace(",", "").replace("$", "").replace("€", "").strip()

        # Handle K/M suffixes
        match = re.match(r"^([\d.]+)\s*[kK]$", text)
        if match:
            return str(int(float(match.group(1)) * 1000))

        match = re.match(r"^([\d.]+)\s*[mM]$", text)
        if match:
            return str(int(float(match.group(1)) * 1_000_000))

        # Remove trailing non-numeric (like 'BB')
        match = re.match(r"^([\d.]+)", text)
        if match:
            return match.group(1)

        return text
