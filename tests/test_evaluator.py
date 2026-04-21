"""Tests for Hand Evaluator abstraction and backends."""

import itertools
import pytest

from services.solver_core.evaluator import (
    BuiltinEvaluator,
    HandEvaluator,
    card_to_int,
    int_to_card,
    card_rank,
    card_suit,
    get_best_evaluator,
    get_evaluator_by_name,
    _evaluate_five_int,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

_R_MAP = {"2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "7": 5, "8": 6, "9": 7,
          "T": 8, "J": 9, "Q": 10, "K": 11, "A": 12}
_S_MAP = {"c": 0, "d": 1, "h": 2, "s": 3}


def _c(code: str) -> int:
    """Convert 'Ah' -> int."""
    return _R_MAP[code[0]] * 4 + _S_MAP[code[1]]


def _hand(codes: list[str]) -> list[int]:
    """Convert list of card codes to list of ints."""
    return [_c(c) for c in codes]


# ─── card_to_int / int_to_card ───────────────────────────────────────────────


class TestCardEncoding:
    def test_card_to_int_range(self):
        for r in _R_MAP:
            for s in _S_MAP:
                i = card_to_int(r, s)
                assert 0 <= i <= 51

    def test_roundtrip(self):
        for i in range(52):
            r, s = int_to_card(i)
            assert card_to_int(r, s) == i

    def test_rank_extraction(self):
        assert card_rank(_c("Ah")) == 12  # Ace
        assert card_rank(_c("2c")) == 0   # Two

    def test_suit_extraction(self):
        assert card_suit(_c("Ah")) == 2   # hearts
        assert card_suit(_c("Ks")) == 3   # spades

    def test_all_unique(self):
        all_ints = [card_to_int(r, s) for r in _R_MAP for s in _S_MAP]
        assert len(set(all_ints)) == 52


# ─── Builtin Evaluator ──────────────────────────────────────────────────────


class TestBuiltinEvaluator:
    @pytest.fixture
    def ev(self) -> BuiltinEvaluator:
        return BuiltinEvaluator()

    def test_name(self, ev):
        assert ev.name == "builtin"

    def test_royal_flush(self, ev):
        h = _hand(["Ah", "Kh", "Qh", "Jh", "Th"])
        rank = ev.evaluate(h)
        assert rank > 0

    def test_straight_flush_lower_than_royal(self, ev):
        royal = ev.evaluate(_hand(["Ah", "Kh", "Qh", "Jh", "Th"]))
        sf = ev.evaluate(_hand(["9h", "8h", "7h", "6h", "5h"]))
        assert royal > sf

    def test_hand_ranking_order(self, ev):
        """Each hand category should rank higher than the one below."""
        hands = [
            _hand(["Ah", "Kh", "Qh", "Jh", "Th"]),  # Royal flush
            _hand(["9s", "8s", "7s", "6s", "5s"]),    # Straight flush
            _hand(["As", "Ah", "Ad", "Ac", "Ks"]),    # Four of a kind
            _hand(["Ks", "Kh", "Kd", "Qs", "Qh"]),    # Full house
            _hand(["Ah", "Jh", "9h", "7h", "3h"]),    # Flush
            _hand(["9c", "8d", "7h", "6s", "5c"]),    # Straight
            _hand(["Js", "Jh", "Jd", "Ac", "Kc"]),    # Three of a kind
            _hand(["As", "Ah", "Ks", "Kh", "Qc"]),    # Two pair
            _hand(["As", "Ah", "Kc", "Qd", "Js"]),    # One pair
            _hand(["As", "Kd", "Qc", "Jh", "9s"]),    # High card
        ]

        ranks = [ev.evaluate(h) for h in hands]
        for i in range(len(ranks) - 1):
            assert ranks[i] > ranks[i + 1], (
                f"Hand {i} (rank={ranks[i]}) should be > hand {i+1} (rank={ranks[i+1]})"
            )

    def test_wheel_straight(self, ev):
        wheel = ev.evaluate(_hand(["Ah", "2d", "3c", "4s", "5h"]))
        high_card = ev.evaluate(_hand(["Ah", "Kd", "Qc", "Jh", "9s"]))
        assert wheel > high_card

    def test_seven_cards(self, ev):
        """Should find best 5 from 7."""
        # Has a flush in hearts (Ah, Kh, 9h, 7h, 3h)
        h = _hand(["Ah", "Kh", "9h", "7h", "3h", "2d", "4c"])
        rank = ev.evaluate(h)
        # Should be at least a flush
        pure_flush = ev.evaluate(_hand(["Ah", "Kh", "9h", "7h", "3h"]))
        assert rank >= pure_flush

    def test_six_cards(self, ev):
        """Should work with 6 cards too."""
        h = _hand(["Ah", "Ad", "Kc", "Ks", "Qh", "2d"])
        rank = ev.evaluate(h)
        assert rank > 0

    def test_less_than_five(self, ev):
        """Should handle <5 cards gracefully."""
        h = _hand(["Ah", "Kd"])
        rank = ev.evaluate(h)
        assert rank >= 0


# ─── Evaluator Protocol ─────────────────────────────────────────────────────


class TestEvaluatorProtocol:
    def test_builtin_is_hand_evaluator(self):
        assert isinstance(BuiltinEvaluator(), HandEvaluator)

    def test_get_best_evaluator(self):
        ev = get_best_evaluator()
        assert isinstance(ev, HandEvaluator)
        assert ev.name in ("builtin", "eval7", "treys")

    def test_get_evaluator_by_name_builtin(self):
        ev = get_evaluator_by_name("builtin")
        assert ev.name == "builtin"

    def test_get_evaluator_by_name_unknown(self):
        with pytest.raises(ValueError, match="Unknown evaluator"):
            get_evaluator_by_name("nonexistent")


# ─── Cross-validation (if eval7/treys available) ────────────────────────────


class TestCrossValidation:
    """Ensure all available evaluators produce consistent ordering."""

    def _get_all_evaluators(self) -> list[HandEvaluator]:
        evs = [BuiltinEvaluator()]
        try:
            evs.append(get_evaluator_by_name("eval7"))
        except (ImportError, Exception):
            pass
        try:
            evs.append(get_evaluator_by_name("treys"))
        except (ImportError, Exception):
            pass
        return evs

    def test_all_evaluators_agree_on_order(self):
        """All evaluators should agree on hand ranking order."""
        evs = self._get_all_evaluators()
        if len(evs) < 2:
            pytest.skip("Need at least 2 evaluators for cross-validation")

        hands = [
            _hand(["Ah", "Kh", "Qh", "Jh", "Th"]),  # Royal flush
            _hand(["9s", "8s", "7s", "6s", "5s"]),    # Straight flush
            _hand(["As", "Ah", "Ad", "Ac", "Ks"]),    # Four of a kind
            _hand(["Ah", "Jh", "9h", "7h", "3h"]),    # Flush
            _hand(["9c", "8d", "7h", "6s", "5c"]),    # Straight
            _hand(["As", "Ah", "Kc", "Qd", "Js"]),    # One pair
            _hand(["As", "Kd", "Qc", "Jh", "9s"]),    # High card
        ]

        for ev in evs:
            ranks = [ev.evaluate(h) for h in hands]
            for i in range(len(ranks) - 1):
                assert ranks[i] > ranks[i + 1], (
                    f"Evaluator {ev.name}: hand {i} should be > hand {i+1}"
                )
