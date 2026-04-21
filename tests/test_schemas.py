"""Tests for shared Pydantic schemas."""

from libs.common.schemas import (
    Action,
    ActionType,
    BoundingBox,
    Card,
    ConfidenceReport,
    FrameAnalysis,
    Rank,
    Recommendation,
    Suit,
    TableState,
)


class TestCard:
    def test_card_code(self):
        card = Card(rank=Rank.ACE, suit=Suit.HEARTS)
        assert card.code == "Ah"

    def test_card_unknown(self):
        card = Card()
        assert card.code == "??"
        assert not card.is_known

    def test_card_is_known(self):
        card = Card(rank=Rank.TEN, suit=Suit.CLUBS)
        assert card.is_known

    def test_card_str(self):
        card = Card(rank=Rank.KING, suit=Suit.SPADES)
        assert str(card) == "Ks"


class TestBoundingBox:
    def test_valid_bbox(self):
        bbox = BoundingBox(x=100, y=200, w=50, h=80, confidence=0.95)
        assert bbox.confidence == 0.95

    def test_confidence_bounds(self):
        import pytest
        with pytest.raises(Exception):
            BoundingBox(x=0, y=0, w=10, h=10, confidence=1.5)


class TestTableState:
    def test_hero_property(self, sample_table_state):
        hero = sample_table_state.hero
        assert hero is not None
        assert hero.is_hero
        assert hero.name == "Hero"

    def test_effective_stack(self, sample_table_state):
        eff = sample_table_state.effective_stack
        # Hero has 2000, max opponent has 3000, so effective = min(2000, 3000) = 2000
        assert eff == 2000.0

    def test_spr(self, sample_table_state):
        spr = sample_table_state.spr
        # eff_stack=2000, pot=500 → spr=4.0
        assert spr == 4.0

    def test_empty_state(self):
        state = TableState()
        assert state.hero is None
        assert state.effective_stack == 0.0


class TestConfidenceReport:
    def test_overall_confidence(self):
        conf = ConfidenceReport(
            vision_confidence=0.9,
            ocr_confidence=0.8,
            state_confidence=0.85,
            recommendation_confidence=0.75,
        )
        assert 0.0 < conf.overall <= 1.0

    def test_dangerous_low_overall(self):
        conf = ConfidenceReport(
            vision_confidence=0.3,
            ocr_confidence=0.3,
            state_confidence=0.3,
            recommendation_confidence=0.3,
        )
        assert conf.is_dangerous

    def test_dangerous_single_low(self):
        conf = ConfidenceReport(
            vision_confidence=0.2,
            ocr_confidence=0.9,
            state_confidence=0.9,
            recommendation_confidence=0.9,
        )
        assert conf.is_dangerous  # vision < 0.3

    def test_not_dangerous(self):
        conf = ConfidenceReport(
            vision_confidence=0.9,
            ocr_confidence=0.9,
            state_confidence=0.9,
            recommendation_confidence=0.9,
        )
        assert not conf.is_dangerous


class TestRecommendation:
    def test_default_recommendation(self):
        rec = Recommendation(
            best_action=Action(action_type=ActionType.CHECK)
        )
        assert rec.best_action.action_type == ActionType.CHECK
        assert rec.play_style.value == "balanced"

    def test_uncertain_recommendation(self):
        rec = Recommendation(
            best_action=Action(action_type=ActionType.UNCERTAIN),
            is_uncertain=True,
        )
        assert rec.is_uncertain


class TestFrameAnalysis:
    def test_default_frame_analysis(self):
        fa = FrameAnalysis()
        assert fa.frame_idx == 0
        assert fa.recommendation.best_action.action_type == ActionType.UNCERTAIN
