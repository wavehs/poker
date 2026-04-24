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
from services.ocr_core.preprocess import (
    contrast_boost,
    crop_bbox,
    preprocess_fallback,
    preprocess_for_ocr,
    upscale_x2,
)

logger = logging.getLogger(__name__)


class OCREngine:
    """
    OCR engine that extracts text from frame regions.

    Backend priority: PaddleOCR → EasyOCR → Mock (auto-fallback).
    Preserves Phase 1 interface: extract(), extract_region().
    """

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
        self.backend = backend
        self.confidence_boost = confidence_boost
        self._backend_impl: OCRBackend = create_backend(backend, use_gpu=use_gpu)
        self._use_real_ocr = self._backend_impl.name != "mock"

        logger.info(
            "OCREngine initialized: backend=%s (resolved=%s)",
            backend,
            self._backend_impl.name,
        )

    @property
    def backend_name(self) -> str:
        return self._backend_impl.name

    def extract(
        self,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> list[OCRResult]:
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
        if self._use_real_ocr:
            return self._real_ocr_region(frame, bbox, field_type)

        return OCRResult(
            text="0",
            confidence=0.5,
            bbox=bbox,
            field_type=field_type,
        )

    # ─── Real OCR ────────────────────────────────────────────────────────

    def _run_fallback_pipeline(self, crop: np.ndarray, field_type: str) -> tuple[str, float]:
        """
        Run OCR with multiple preprocessing fallbacks and return the best result.
        """
        best_conf = 0.0
        best_text = ""

        def evaluate_results(raw_results):
            nonlocal best_conf, best_text
            if raw_results:
                text, conf, _ = max(raw_results, key=lambda r: r[1])
                text = text.strip()
                if conf > best_conf and text:
                    best_conf = conf
                    best_text = text

        processed = preprocess_for_ocr(crop, field_type=field_type)
        evaluate_results(self._backend_impl.recognize(processed))

        if best_conf < 0.75:
            cb_crop = contrast_boost(crop)
            processed = preprocess_for_ocr(cb_crop, field_type=field_type)
            evaluate_results(self._backend_impl.recognize(processed))

        if best_conf < 0.75:
            up_crop = upscale_x2(crop)
            processed = preprocess_for_ocr(up_crop, field_type=field_type)
            evaluate_results(self._backend_impl.recognize(processed))

        if best_conf < 0.75:
            fallback_processed = preprocess_fallback(crop)
            evaluate_results(self._backend_impl.recognize(fallback_processed))

        if best_conf < 0.75:
            evaluate_results(self._backend_impl.recognize(crop))

        return best_text, best_conf

    def _run_ocr_pipeline(
        self, crop: np.ndarray, field_type: str, bbox: BoundingBox
    ) -> OCRResult | None:
        """Run OCR pipeline with fallback preprocessing and validation."""
        t0 = time.perf_counter()

        best_text, best_conf = self._run_fallback_pipeline(crop, field_type)
        latency_ms = (time.perf_counter() - t0) * 1000

        if not best_text:
            return None

        if field_type in ("pot", "stack", "bet", "blind"):
            best_text = self._clean_numeric(best_text)

            try:
                val = float(best_text)
                if not (0 <= val <= 1_000_000):
                    logger.warning(
                        "OCR validation failed: [%s] value %s out of bounds",
                        field_type,
                        best_text,
                    )
                    return None
            except ValueError:
                logger.warning(
                    "OCR validation failed: [%s] value '%s' is not numeric",
                    field_type,
                    best_text,
                )
                return None

        confidence = min(1.0, best_conf + self.confidence_boost)

        low_confidence = confidence < 0.6

        if not low_confidence and field_type in ("pot", "bet"):
            try:
                num_val = float(best_text)
                if num_val < 0 or num_val > 10_000_000:
                    low_confidence = True
                    logger.warning(
                        "Validation failed for %s: %s is out of range [0, 10_000_000]",
                        field_type,
                        num_val,
                    )
            except ValueError:
                low_confidence = True
                logger.warning(
                    "Validation failed for %s: %s is not a valid number",
                    field_type,
                    best_text,
                )

        logger.debug(
            "OCR [%s]: '%s' (conf=%.2f, %.1fms)",
            field_type,
            best_text,
            confidence,
            latency_ms,
        )

        res = OCRResult(
            text=best_text,
            confidence=confidence,
            bbox=bbox,
            field_type=field_type,
        )
        object.__setattr__(res, "low_confidence", low_confidence)
        return res

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
        text = detection.label.strip()
        if not text:
            return None

        field_type = self._field_type_from_detection(detection)
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
        field_type_map = {
            DetectionClass.POT: "pot",
            DetectionClass.BET_AMOUNT: "bet",
            DetectionClass.PLAYER_PANEL: "stack",
            DetectionClass.CHIP_STACK: "stack",
        }
        return field_type_map.get(detection.detection_class, "generic")

    @staticmethod
    def _clean_numeric(text: str) -> str:
        text = text.replace(",", "").replace("$", "").replace("€", "").strip()

        match = re.match(r"^([\d.]+)\s*[kK]$", text)
        if match:
            return str(int(float(match.group(1)) * 1000))

        match = re.match(r"^([\d.]+)\s*[mM]$", text)
        if match:
            return str(int(float(match.group(1)) * 1_000_000))

        match = re.match(r"^([\d.]+)", text)
        if match:
            return match.group(1)

        return text