"""
OCR Core — Text recognition abstraction layer.

Phase 1: Extracts text from detections using label data (mock).
Phase 2+: Real OCR via Tesseract / EasyOCR / PaddleOCR.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np

from libs.common.schemas import BoundingBox, Detection, DetectionClass, OCRResult


class OCREngine:
    """
    OCR engine that extracts text from frame regions.
    
    Phase 1: Uses detection labels as pseudo-OCR output.
    Phase 2: Will integrate real OCR backends.
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
    ) -> None:
        """
        Args:
            backend: OCR backend name ('mock', 'tesseract', 'easyocr', 'paddle').
            confidence_boost: Added to base confidence for tuning.
        """
        self.backend = backend
        self.confidence_boost = confidence_boost

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

            if self.backend == "mock":
                ocr = self._mock_ocr(det)
            else:
                ocr = self._real_ocr(frame, det)

            if ocr is not None:
                results.append(ocr)

        return results

    def extract_region(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
        field_type: str = "generic",
    ) -> Optional[OCRResult]:
        """
        Run OCR on a specific region.
        
        Args:
            frame: BGR frame.
            bbox: Region to OCR.
            field_type: Semantic type for the result.
            
        Returns:
            OCRResult or None if nothing readable.
        """
        if self.backend == "mock":
            return OCRResult(
                text="0",
                confidence=0.5,
                bbox=bbox,
                field_type=field_type,
            )
        return self._real_ocr_region(frame, bbox, field_type)

    def _mock_ocr(self, detection: Detection) -> Optional[OCRResult]:
        """
        Generate mock OCR result from detection label.
        """
        text = detection.label.strip()
        if not text:
            return None

        # Determine field type from detection class
        field_type_map = {
            DetectionClass.POT: "pot",
            DetectionClass.BET_AMOUNT: "bet",
            DetectionClass.PLAYER_PANEL: "stack",
            DetectionClass.CHIP_STACK: "stack",
        }
        field_type = field_type_map.get(detection.detection_class, "generic")

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

    def _real_ocr(self, frame: np.ndarray, detection: Detection) -> Optional[OCRResult]:
        """Real OCR on a detection region (Phase 2 stub)."""
        # TODO: Crop region from frame, preprocess, run OCR backend
        return None

    def _real_ocr_region(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
        field_type: str,
    ) -> Optional[OCRResult]:
        """Real OCR on an arbitrary region (Phase 2 stub)."""
        # TODO: Crop, preprocess, run backend
        return None

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
