"""Tests for Vision Core — detector, mock and YOLO adapter."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from libs.common.schemas import BoundingBox, Detection, DetectionClass
from services.vision_core.detector import VisionDetector, _YOLO_INDEX_TO_CLASS


# ─── Mock mode ───────────────────────────────────────────────────────────────


class TestVisionDetectorMock:
    def test_mock_mode_by_default(self):
        detector = VisionDetector()
        assert detector.is_mock

    def test_mock_detect_produces_detections(self):
        detector = VisionDetector()
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        detections = detector.detect(frame, frame_idx=0)
        assert len(detections) > 0

    def test_mock_detect_has_cards(self):
        detector = VisionDetector()
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        card_dets = [d for d in detections if d.detection_class == DetectionClass.CARD]
        assert len(card_dets) >= 2  # At least hero cards

    def test_mock_detect_has_pot(self):
        detector = VisionDetector()
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        pot_dets = [d for d in detections if d.detection_class == DetectionClass.POT]
        assert len(pot_dets) == 1

    def test_mock_detect_confidence_threshold(self):
        detector = VisionDetector(confidence_threshold=0.99)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        for d in detections:
            assert d.bbox.confidence >= 0.99

    def test_detect_batch_mock(self):
        detector = VisionDetector()
        frames = [np.zeros((1080, 1920, 3), dtype=np.uint8) for _ in range(3)]
        results = detector.detect_batch(frames)
        assert len(results) == 3
        for batch_dets in results:
            assert len(batch_dets) > 0

    def test_inference_latency_tracked(self):
        detector = VisionDetector()
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        detector.detect(frame)
        assert detector.last_inference_ms > 0


# ─── YOLO class mapping ─────────────────────────────────────────────────────


class TestYOLOClassMapping:
    def test_all_detection_classes_mapped(self):
        mapped_classes = set(_YOLO_INDEX_TO_CLASS.values())
        expected = {
            DetectionClass.CARD,
            DetectionClass.CHIP_STACK,
            DetectionClass.POT,
            DetectionClass.DEALER_BUTTON,
            DetectionClass.PLAYER_PANEL,
            DetectionClass.BET_AMOUNT,
            DetectionClass.ACTION_BUTTON,
            DetectionClass.BOARD_AREA,
        }
        assert mapped_classes == expected

    def test_indices_sequential(self):
        indices = sorted(_YOLO_INDEX_TO_CLASS.keys())
        assert indices == list(range(8))


# ─── _parse_results ─────────────────────────────────────────────────────────


class TestParseResults:
    def test_parse_empty_results(self):
        detector = VisionDetector()
        mock_result = MagicMock()
        mock_result.boxes = None
        detections = detector._parse_results([mock_result], frame_idx=0)
        assert detections == []

    def test_parse_single_detection(self):
        detector = VisionDetector()

        # Mock a result with one detection
        mock_boxes = MagicMock()
        mock_boxes.xyxy = [np.array([100, 200, 200, 350])]
        mock_boxes.conf = [np.float32(0.92)]
        mock_boxes.cls = [np.float32(0)]  # card
        mock_boxes.__len__ = lambda self: 1

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes

        detections = detector._parse_results([mock_result], frame_idx=5)

        assert len(detections) == 1
        d = detections[0]
        assert d.detection_class == DetectionClass.CARD
        assert abs(d.bbox.x - 100) < 1
        assert abs(d.bbox.y - 200) < 1
        assert abs(d.bbox.w - 100) < 1
        assert abs(d.bbox.h - 150) < 1
        assert abs(d.bbox.confidence - 0.92) < 0.01
        assert d.frame_idx == 5

    def test_parse_multiple_detections(self):
        detector = VisionDetector()

        mock_boxes = MagicMock()
        mock_boxes.xyxy = [
            np.array([100, 200, 200, 350]),
            np.array([300, 400, 500, 450]),
        ]
        mock_boxes.conf = [np.float32(0.9), np.float32(0.8)]
        mock_boxes.cls = [np.float32(0), np.float32(2)]  # card, pot
        mock_boxes.__len__ = lambda self: 2

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes

        detections = detector._parse_results([mock_result], frame_idx=0)

        assert len(detections) == 2
        assert detections[0].detection_class == DetectionClass.CARD
        assert detections[1].detection_class == DetectionClass.POT


# ─── Model loading fallback ─────────────────────────────────────────────────


class TestModelLoading:
    def test_invalid_model_path_falls_back_to_mock(self):
        detector = VisionDetector(model_path="nonexistent_model.pt")
        assert detector.is_mock

    def test_model_path_none_is_mock(self):
        detector = VisionDetector(model_path=None)
        assert detector.is_mock
