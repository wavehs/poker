"""Tests for OCR Core — backends, preprocessing, and engine."""

import cv2
import numpy as np

from libs.common.schemas import BoundingBox, Detection, DetectionClass
from services.ocr_core.backends import (
    MockOCRBackend,
    create_backend,
)
from services.ocr_core.ocr import OCREngine
from services.ocr_core.preprocess import (
    _auto_invert,
    _resize_height,
    _threshold_otsu,
    _to_grayscale,
    crop_bbox,
    preprocess_for_ocr,
)

# ─── MockOCRBackend ──────────────────────────────────────────────────────────


class TestMockOCRBackend:
    def test_name(self):
        b = MockOCRBackend()
        assert b.name == "mock"

    def test_is_available(self):
        b = MockOCRBackend()
        assert b.is_available

    def test_recognize_returns_empty(self):
        b = MockOCRBackend()
        crop = np.zeros((32, 100, 3), dtype=np.uint8)
        assert b.recognize(crop) == []


# ─── create_backend ──────────────────────────────────────────────────────────


class TestCreateBackend:
    def test_mock_backend(self):
        b = create_backend("mock")
        assert b.name == "mock"

    def test_auto_returns_something(self):
        b = create_backend("auto")
        assert b.name in ("paddle", "easyocr", "mock")

    def test_unknown_falls_back_to_mock(self):
        b = create_backend("nonexistent")
        assert b.name == "mock"


# ─── Preprocessing ───────────────────────────────────────────────────────────


from services.ocr_core.preprocess import contrast_boost, upscale_x2

class TestPreprocessing:
    def test_contrast_boost(self):
        img = np.full((10, 10, 3), 100, dtype=np.uint8)
        boosted = contrast_boost(img, alpha=1.5, beta=0)
        assert np.array_equal(boosted, np.full((10, 10, 3), 150, dtype=np.uint8))

    def test_upscale_x2(self):
        img = np.zeros((10, 20, 3), dtype=np.uint8)
        upscaled = upscale_x2(img)
        assert upscaled.shape == (20, 40)

    def test_to_grayscale_bgr(self):
        img = np.random.randint(0, 255, (50, 100, 3), dtype=np.uint8)
        gray = _to_grayscale(img)
        assert gray.ndim == 2
        assert gray.shape == (50, 100)

    def test_to_grayscale_already_gray(self):
        gray = np.random.randint(0, 255, (50, 100), dtype=np.uint8)
        result = _to_grayscale(gray)
        assert np.array_equal(result, gray)

    def test_auto_invert_dark_background(self):
        # Dark image (mean < 128) should be inverted
        dark = np.full((50, 100), 30, dtype=np.uint8)
        inverted = _auto_invert(dark)
        assert np.mean(inverted) > 128

    def test_auto_invert_light_background(self):
        # Light image (mean >= 128) should stay
        light = np.full((50, 100), 200, dtype=np.uint8)
        result = _auto_invert(light)
        assert np.mean(result) >= 128

    def test_threshold_otsu(self):
        gray = np.random.randint(0, 255, (50, 100), dtype=np.uint8)
        binary = _threshold_otsu(gray)
        unique = set(binary.flatten())
        assert unique <= {0, 255}

    def test_resize_height(self):
        gray = np.zeros((100, 200), dtype=np.uint8)
        resized = _resize_height(gray, 64)
        assert resized.shape[0] == 64
        # Aspect ratio preserved: 200 * (64/100) = 128
        assert resized.shape[1] == 128

    def test_preprocess_numeric(self):
        img = np.random.randint(0, 255, (40, 80, 3), dtype=np.uint8)
        result = preprocess_for_ocr(img, field_type="pot", target_height=64)
        assert result.ndim == 2  # grayscale
        assert result.shape[0] >= 64  # height + padding

    def test_preprocess_empty_crop(self):
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        result = preprocess_for_ocr(empty, target_height=64)
        assert result.shape == (64, 64)


class TestCropBbox:
    def test_crop_valid_region(self):
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        crop = crop_bbox(frame, 100, 200, 300, 150, padding=0.0)
        assert crop.shape == (150, 300, 3)

    def test_crop_with_padding(self):
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        crop = crop_bbox(frame, 100, 200, 100, 50, padding=0.1)
        # With 10% padding: w=100+20=120, h=50+10=60
        assert crop.shape[0] >= 50
        assert crop.shape[1] >= 100

    def test_crop_clamps_to_frame(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = crop_bbox(frame, 80, 80, 50, 50, padding=0.0)
        # Should clamp to frame boundary
        assert crop.shape[0] <= 100
        assert crop.shape[1] <= 100


# ─── OCREngine — Mock mode ──────────────────────────────────────────────────


class TestOCREngineMock:
    def test_init_mock(self):
        engine = OCREngine(backend="mock")
        assert engine.backend_name == "mock"

    def test_extract_from_mock_detections(self):
        engine = OCREngine(backend="mock")
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        detections = [
            Detection(
                detection_class=DetectionClass.POT,
                bbox=BoundingBox(x=400, y=300, w=200, h=50, confidence=0.9),
                label="500",
            ),
            Detection(
                detection_class=DetectionClass.PLAYER_PANEL,
                bbox=BoundingBox(x=100, y=800, w=150, h=60, confidence=0.85),
                label="2000",
            ),
            Detection(
                detection_class=DetectionClass.CARD,
                bbox=BoundingBox(x=500, y=700, w=40, h=70, confidence=0.92),
                label="Ah",
            ),
        ]

        results = engine.extract(frame, detections)
        # Cards should be skipped, only text classes processed
        assert len(results) == 2
        assert results[0].field_type == "pot"
        assert results[0].text == "500"
        assert results[1].field_type == "stack"

    def test_extract_region_mock(self):
        engine = OCREngine(backend="mock")
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = BoundingBox(x=10, y=10, w=50, h=20, confidence=0.8)

        result = engine.extract_region(frame, bbox, field_type="pot")
        assert result is not None
        assert result.text == "0"
        assert result.field_type == "pot"

    def test_extract_empty_label(self):
        engine = OCREngine(backend="mock")
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = [
            Detection(
                detection_class=DetectionClass.POT,
                bbox=BoundingBox(x=10, y=10, w=50, h=20, confidence=0.9),
                label="",
            ),
        ]
        results = engine.extract(frame, detections)
        assert len(results) == 0


# ─── OCREngine — Real OCR Edge Cases ────────────────────────────────────────

from unittest.mock import MagicMock, patch

class TestOCREngineEdgeCases:
    def test_ocr_fallback_low_confidence(self):
        engine = OCREngine(backend="auto")
        # Ensure it acts like a real backend so fallback logic triggers
        engine._use_real_ocr = True

        # Replace actual backend with a mock
        engine._backend_impl = MagicMock()

        # Simulate recognize returning low confidence first, then high confidence on fallback
        engine._backend_impl.recognize.side_effect = [
            [("50", 0.6, None)],     # First try: low confidence (0.6 < 0.7)
            [("500", 0.9, None)]     # Fallback try: higher confidence (0.9)
        ]

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = BoundingBox(x=10, y=10, w=50, h=20, confidence=0.8)

        with patch("services.ocr_core.ocr.preprocess_fallback", return_value=np.zeros((100, 100), dtype=np.uint8)) as mock_fallback:
            result = engine.extract_region(frame, bbox, field_type="pot")

            assert mock_fallback.called
            assert result is not None
            assert result.text == "500"
            assert result.confidence == 0.9

    def test_ocr_validation_out_of_range(self):
        engine = OCREngine(backend="auto")
        engine._use_real_ocr = True
        engine._backend_impl = MagicMock()

        # Simulate recognize returning a number > 1000000
        engine._backend_impl.recognize.side_effect = [
            [("2000000", 0.9, None)] # First try: confidence is good, but value is out of range
        ]

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = BoundingBox(x=10, y=10, w=50, h=20, confidence=0.8)

        result = engine.extract_region(frame, bbox, field_type="pot")

        # Should return None because 2000000 > 1000000
        assert result is None

    def test_ocr_validation_not_numeric(self):
        engine = OCREngine(backend="auto")
        engine._use_real_ocr = True
        engine._backend_impl = MagicMock()

        # Simulate recognize returning non-numeric text for a numeric field
        engine._backend_impl.recognize.side_effect = [
            [("hello", 0.9, None)]
        ]

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = BoundingBox(x=10, y=10, w=50, h=20, confidence=0.8)

        result = engine.extract_region(frame, bbox, field_type="pot")

        # Should return None because 'hello' cannot be parsed to float
        assert result is None


# ─── OCREngine — Clean numeric ──────────────────────────────────────────────


class TestCleanNumeric:
    def test_plain_number(self):
        assert OCREngine._clean_numeric("500") == "500"

    def test_dollar_sign(self):
        assert OCREngine._clean_numeric("$1500") == "1500"

    def test_comma_separator(self):
        assert OCREngine._clean_numeric("$1,500") == "1500"

    def test_k_suffix(self):
        assert OCREngine._clean_numeric("1.5K") == "1500"

    def test_m_suffix(self):
        assert OCREngine._clean_numeric("2M") == "2000000"

    def test_bb_suffix(self):
        assert OCREngine._clean_numeric("500BB") == "500"

    def test_euro_sign(self):
        assert OCREngine._clean_numeric("€250") == "250"

    def test_whitespace(self):
        assert OCREngine._clean_numeric("  1000  ") == "1000"

    def test_non_numeric(self):
        assert OCREngine._clean_numeric("abc") == "abc"


# ─── OCREngine — Synthetic image OCR ────────────────────────────────────────


class TestOCRSyntheticImage:
    """Tests using cv2.putText to create images with known text."""

    @staticmethod
    def _make_text_image(text: str, size: tuple = (200, 80)) -> np.ndarray:
        """Create a BGR image with white text on dark background."""
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        img[:] = (30, 30, 30)  # dark background
        cv2.putText(
            img, text, (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2
        )
        return img

    def test_preprocess_synthetic_text(self):
        """Verify preprocessing produces clean binary output."""
        img = self._make_text_image("1500")
        processed = preprocess_for_ocr(img, field_type="pot")
        assert processed.ndim == 2
        # Should be mostly binary
        unique = set(processed.flatten())
        assert 0 in unique or 255 in unique

# ─── OCREngine — Fallbacks and Validation ────────────────────────────────────

class MockRealBackend:
    def __init__(self, responses):
        self.name = "mock_real"
        self.is_available = True
        self.responses = responses
        self.call_count = 0

    def recognize(self, crop):
        if self.call_count < len(self.responses):
            res = self.responses[self.call_count]
            self.call_count += 1
            return res
        return []

class TestOCRFallbackAndValidation:
    def test_ocr_fallback_pipeline(self, monkeypatch):
        engine = OCREngine(backend="mock")
        # override private backend with our mock sequence
        mock_backend = MockRealBackend([
            [], # 1. standard fails
            [], # 2. contrast boost fails
            [("1000", 0.8, [])], # 3. upscale succeeds
        ])
        engine._backend_impl = mock_backend
        engine._use_real_ocr = True

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = BoundingBox(x=10, y=10, w=50, h=50, confidence=0.9)
        det = Detection(detection_class=DetectionClass.POT, bbox=bbox, label="")

        result = engine._real_ocr(frame, det)
        assert result is not None
        assert result.text == "1000"
        assert result.confidence == 0.8
        assert not getattr(result, "low_confidence", False)

    def test_low_confidence_flag(self):
        engine = OCREngine(backend="mock")
        mock_backend = MockRealBackend([
            [("500", 0.5, [])], # standard succeeds but low conf
            [], # contrast fails
            [], # upscale fails
            [], # original fails
        ])
        engine._backend_impl = mock_backend
        engine._use_real_ocr = True

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = BoundingBox(x=10, y=10, w=50, h=50, confidence=0.9)
        det = Detection(detection_class=DetectionClass.POT, bbox=bbox, label="")

        result = engine._real_ocr(frame, det)
        assert result is not None
        assert result.text == "500"
        assert result.confidence == 0.5
        assert getattr(result, "low_confidence", False) is True

    def test_pot_bet_validation_out_of_range(self):
        engine = OCREngine(backend="mock")
        mock_backend = MockRealBackend([
            [("15000000", 0.9, [])], # high conf but out of range
        ])
        engine._backend_impl = mock_backend
        engine._use_real_ocr = True

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = BoundingBox(x=10, y=10, w=50, h=50, confidence=0.9)
        det = Detection(detection_class=DetectionClass.POT, bbox=bbox, label="")

        result = engine._real_ocr(frame, det)
        assert result is not None
        assert result.text == "15000000"
        assert getattr(result, "low_confidence", False) is True

    def test_pot_bet_validation_invalid_number(self):
        engine = OCREngine(backend="mock")
        mock_backend = MockRealBackend([
            [("abc", 0.9, [])], # high conf but not a number
        ])
        engine._backend_impl = mock_backend
        engine._use_real_ocr = True

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = BoundingBox(x=10, y=10, w=50, h=50, confidence=0.9)
        det = Detection(detection_class=DetectionClass.POT, bbox=bbox, label="")

        result = engine._real_ocr(frame, det)
        assert result is not None
        assert result.text == "abc"
        assert getattr(result, "low_confidence", False) is True
