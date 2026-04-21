"""Tests for State Engine."""

import numpy as np

from libs.common.schemas import DetectionClass, Street
from services.state_engine.engine import StateEngine, parse_card
from services.vision_core.detector import VisionDetector
from services.ocr_core.ocr import OCREngine


class TestParseCard:
    def test_parse_ace_hearts(self):
        card = parse_card("Ah")
        assert card.code == "Ah"
        assert card.is_known

    def test_parse_ten_clubs(self):
        card = parse_card("Tc")
        assert card.code == "Tc"

    def test_parse_invalid(self):
        card = parse_card("X")
        assert not card.is_known

    def test_parse_empty(self):
        card = parse_card("")
        assert not card.is_known


class TestStateEngine:
    def test_update_produces_table_state(self, blank_frame):
        detector = VisionDetector()
        ocr = OCREngine()
        engine = StateEngine()

        detections = detector.detect(blank_frame, frame_idx=0)
        ocr_results = ocr.extract(blank_frame, detections)
        state, tracked = engine.update(detections, ocr_results, frame_idx=0)

        assert state is not None
        assert isinstance(state.pot, float)
        assert state.street in Street

    def test_street_detection(self):
        engine = StateEngine()

        assert engine._determine_street(0) == Street.PREFLOP
        assert engine._determine_street(3) == Street.FLOP
        assert engine._determine_street(4) == Street.TURN
        assert engine._determine_street(5) == Street.RIVER
        assert engine._determine_street(2) == Street.UNKNOWN

    def test_state_confidence(self, sample_table_state):
        engine = StateEngine()
        conf = engine.get_state_confidence(sample_table_state)
        assert 0.0 <= conf <= 1.0
        # With hero cards, pot, players, and valid street → should be decent
        assert conf > 0.5

    def test_multiple_frames_build_history(self, blank_frame):
        detector = VisionDetector()
        ocr = OCREngine()
        engine = StateEngine(smoothing_window=3)

        for i in range(5):
            detections = detector.detect(blank_frame, frame_idx=i)
            ocr_results = ocr.extract(blank_frame, detections)
            engine.update(detections, ocr_results, frame_idx=i)

        # History should be trimmed to window size
        assert len(engine._frame_history) == 3
