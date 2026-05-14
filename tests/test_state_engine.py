"""Tests for State Engine."""


from libs.common.schemas import (
    BoundingBox,
    Detection,
    DetectionClass,
    Street,
)
from services.ocr_core.ocr import OCREngine
from services.state_engine.engine import StateEngine, parse_card
from services.vision_core.detector import VisionDetector


def _panel(seat_idx: int, x: float, y: float, w: float = 200, h: float = 80) -> Detection:
    return Detection(
        bbox=BoundingBox(x=x, y=y, w=w, h=h, confidence=0.95),
        detection_class=DetectionClass.PLAYER_PANEL,
        label=f"panel_{seat_idx}",
        frame_idx=0,
    )


def _button(x: float, y: float) -> Detection:
    return Detection(
        bbox=BoundingBox(x=x, y=y, w=24, h=24, confidence=0.9),
        detection_class=DetectionClass.DEALER_BUTTON,
        label="dealer_button",
        frame_idx=0,
    )


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


class TestHeroAndDealerSpatialResolution:
    """Regression tests for hero seat identification and dealer-button mapping.

    Before the fix:
      - `_extract_dealer` always returned 0 (hardcoded placeholder).
      - hero seat defaulted to index 0 regardless of visual layout.
    The new implementation identifies hero by panel Y-position (bottom band)
    and the dealer by Euclidean distance from the button detection.
    """

    def test_hero_identified_from_bottom_panel(self):
        engine = StateEngine()
        # 1920x1080 frame; hero panel is the bottom one.
        detections = [
            _panel(0, x=860, y=80),     # top
            _panel(1, x=200, y=480),    # left
            _panel(2, x=1500, y=480),   # right
            _panel(3, x=860, y=900),    # bottom (hero)
        ]
        state, _ = engine.update(
            detections, [], frame_idx=0, frame_shape=(1080, 1920)
        )
        heroes = [p for p in state.players if p.is_hero]
        assert len(heroes) == 1
        assert heroes[0].seat == 3

    def test_dealer_button_maps_to_nearest_panel(self):
        engine = StateEngine()
        detections = [
            _panel(0, x=860, y=80),     # top — has button
            _panel(1, x=200, y=480),
            _panel(2, x=1500, y=480),
            _panel(3, x=860, y=900),
            _button(x=900, y=120),      # directly inside top panel area
        ]
        state, _ = engine.update(
            detections, [], frame_idx=0, frame_shape=(1080, 1920)
        )
        dealers = [p for p in state.players if p.is_dealer]
        assert len(dealers) == 1
        # The top panel is seat 0 — the button should land there, not always
        # on seat 0 by coincidence of the hardcoded return value, so move it
        # to a different position to make sure:
        detections[-1] = _button(x=210, y=520)
        state2, _ = engine.update(
            detections, [], frame_idx=1, frame_shape=(1080, 1920)
        )
        dealers2 = [p for p in state2.players if p.is_dealer]
        assert len(dealers2) == 1
        assert dealers2[0].seat == 1

    def test_no_dealer_button_means_no_dealer(self):
        engine = StateEngine()
        detections = [
            _panel(0, x=860, y=80),
            _panel(1, x=860, y=900),
        ]
        state, _ = engine.update(
            detections, [], frame_idx=0, frame_shape=(1080, 1920)
        )
        assert all(not p.is_dealer for p in state.players)
