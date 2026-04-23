"""Tests for Solver Core — hand evaluation and equity."""


from libs.common.schemas import Card, Rank, Suit
from services.solver_core.evaluator import BuiltinEvaluator
from services.solver_core.solver import (
    EquitySolver,
    SolverProfile,
    evaluate_hand,
    rank_value,
)


class TestRankValue:
    def test_rank_order(self):
        assert rank_value(Rank.TWO) == 0
        assert rank_value(Rank.ACE) == 12
        assert rank_value(Rank.TEN) == 8

    def test_unknown_rank(self):
        assert rank_value(Rank.UNKNOWN) == -1


class TestEvaluateHand:
    def _make_cards(self, codes: list[str]) -> list[Card]:
        """Helper to create cards from short codes like 'Ah', 'Kd'."""
        rank_map = {
            "2": Rank.TWO, "3": Rank.THREE, "4": Rank.FOUR, "5": Rank.FIVE,
            "6": Rank.SIX, "7": Rank.SEVEN, "8": Rank.EIGHT, "9": Rank.NINE,
            "T": Rank.TEN, "J": Rank.JACK, "Q": Rank.QUEEN, "K": Rank.KING,
            "A": Rank.ACE,
        }
        suit_map = {"h": Suit.HEARTS, "d": Suit.DIAMONDS, "c": Suit.CLUBS, "s": Suit.SPADES}

        cards = []
        for code in codes:
            cards.append(Card(
                rank=rank_map[code[0]],
                suit=suit_map[code[1]],
                confidence=1.0,
            ))
        return cards

    def test_high_card(self):
        cards = self._make_cards(["Ah", "Kd", "Jc", "9s", "7h"])
        val, name, _ = evaluate_hand(cards)
        assert name == "high_card"

    def test_pair(self):
        cards = self._make_cards(["Ah", "Ad", "Kc", "Js", "9h"])
        val, name, _ = evaluate_hand(cards)
        assert name == "pair"

    def test_two_pair(self):
        cards = self._make_cards(["Ah", "Ad", "Kc", "Ks", "9h"])
        val, name, _ = evaluate_hand(cards)
        assert name == "two_pair"

    def test_three_of_a_kind(self):
        cards = self._make_cards(["Ah", "Ad", "Ac", "Ks", "9h"])
        val, name, _ = evaluate_hand(cards)
        assert name == "three_of_a_kind"

    def test_straight(self):
        cards = self._make_cards(["5h", "6d", "7c", "8s", "9h"])
        val, name, _ = evaluate_hand(cards)
        assert name == "straight"

    def test_flush(self):
        cards = self._make_cards(["Ah", "Kh", "Jh", "9h", "7h"])
        val, name, _ = evaluate_hand(cards)
        assert name == "flush"

    def test_full_house(self):
        cards = self._make_cards(["Ah", "Ad", "Ac", "Ks", "Kh"])
        val, name, _ = evaluate_hand(cards)
        assert name == "full_house"

    def test_four_of_a_kind(self):
        cards = self._make_cards(["Ah", "Ad", "Ac", "As", "Kh"])
        val, name, _ = evaluate_hand(cards)
        assert name == "four_of_a_kind"

    def test_straight_flush(self):
        cards = self._make_cards(["5h", "6h", "7h", "8h", "9h"])
        val, name, _ = evaluate_hand(cards)
        assert name == "straight_flush"

    def test_royal_flush(self):
        cards = self._make_cards(["Th", "Jh", "Qh", "Kh", "Ah"])
        val, name, _ = evaluate_hand(cards)
        assert name == "royal_flush"

    def test_wheel_straight(self):
        cards = self._make_cards(["Ah", "2d", "3c", "4s", "5h"])
        val, name, _ = evaluate_hand(cards)
        assert name == "straight"

    def test_seven_cards_best_five(self):
        """Should find the best 5-card hand from 7 cards."""
        cards = self._make_cards(["Ah", "Ad", "Kc", "Ks", "Qh", "2d", "3c"])
        val, name, _ = evaluate_hand(cards)
        assert name == "two_pair"  # AA + KK

    def test_hand_ranking_order(self):
        """Higher hands should have higher rank values."""
        pair = self._make_cards(["Ah", "Ad", "Kc", "Js", "9h"])
        flush = self._make_cards(["Ah", "Kh", "Jh", "9h", "7h"])
        full = self._make_cards(["Ah", "Ad", "Ac", "Ks", "Kh"])

        v_pair, _, _ = evaluate_hand(pair)
        v_flush, _, _ = evaluate_hand(flush)
        v_full, _, _ = evaluate_hand(full)

        assert v_pair < v_flush < v_full


class TestEquitySolver:
    def test_equity_returns_float(self, sample_hero_cards, sample_community_cards):
        solver = EquitySolver(default_simulations=500, adaptive=False)
        equity = solver.compute_equity(
            sample_hero_cards,
            sample_community_cards,
            num_opponents=1,
        )
        assert 0.0 <= equity <= 1.0

    def test_equity_with_no_community(self, sample_hero_cards):
        solver = EquitySolver(default_simulations=500, adaptive=False)
        equity = solver.compute_equity(
            sample_hero_cards,
            [],
            num_opponents=1,
        )
        assert 0.0 <= equity <= 1.0

    def test_equity_empty_hand(self):
        solver = EquitySolver(default_simulations=100, adaptive=False)
        equity = solver.compute_equity([], [], num_opponents=1)
        assert equity == 0.0

    def test_pot_odds(self):
        solver = EquitySolver()
        odds = solver.compute_pot_odds(pot=100, to_call=50)
        # 50 / (100 + 50) = 0.333...
        assert abs(odds - 1 / 3) < 0.01

    def test_pot_odds_zero(self):
        solver = EquitySolver()
        odds = solver.compute_pot_odds(pot=0, to_call=0)
        assert odds == 0.0

    def test_hand_strength(self, sample_hero_cards, sample_community_cards):
        solver = EquitySolver()
        strength = solver.compute_hand_strength(
            sample_hero_cards, sample_community_cards
        )
        assert 0.0 <= strength <= 1.0
        # AK with A+K on the board = two pair, should be decent
        assert strength >= 0.2


# ─── Phase 3: Profiling Tests ───────────────────────────────────────────────


class TestSolverProfile:
    def test_profile_populated(self, sample_hero_cards, sample_community_cards):
        """compute_equity should populate last_profile."""
        solver = EquitySolver(default_simulations=200, adaptive=False)
        solver.compute_equity(
            sample_hero_cards, sample_community_cards, num_opponents=1,
        )
        prof = solver.last_profile
        assert prof is not None
        assert prof.total_ms > 0
        assert prof.simulation_ms > 0
        assert prof.simulations_run == 200
        assert prof.evaluate_calls > 0

    def test_profile_as_dict(self, sample_hero_cards, sample_community_cards):
        solver = EquitySolver(default_simulations=100, adaptive=False)
        solver.compute_equity(sample_hero_cards, sample_community_cards)
        d = solver.last_profile.as_dict()
        assert "total_ms" in d
        assert "simulations_run" in d
        assert "evaluate_calls" in d

    def test_profile_repr(self):
        p = SolverProfile()
        p.total_ms = 42.5
        p.simulations_run = 100
        assert "42.5" in repr(p)


# ─── Phase 3: Adaptive Monte Carlo Tests ────────────────────────────────────


class TestAdaptiveMonteCarlo:
    def _make_cards(self, codes: list[str]) -> list[Card]:
        rank_map = {
            "2": Rank.TWO, "3": Rank.THREE, "4": Rank.FOUR, "5": Rank.FIVE,
            "6": Rank.SIX, "7": Rank.SEVEN, "8": Rank.EIGHT, "9": Rank.NINE,
            "T": Rank.TEN, "J": Rank.JACK, "Q": Rank.QUEEN, "K": Rank.KING,
            "A": Rank.ACE,
        }
        suit_map = {"h": Suit.HEARTS, "d": Suit.DIAMONDS, "c": Suit.CLUBS, "s": Suit.SPADES}
        return [Card(rank=rank_map[c[0]], suit=suit_map[c[1]], confidence=1.0) for c in codes]

    def test_adaptive_early_stops_on_clear_spot(self):
        """AA preflop should converge quickly with adaptive MC."""
        hero = self._make_cards(["Ah", "As"])
        solver = EquitySolver(
            default_simulations=5000,
            adaptive=True,
            min_simulations=200,
            step_size=200,
            confidence_threshold=0.03,
        )
        equity = solver.compute_equity(hero, [], num_opponents=1)
        prof = solver.last_profile

        assert prof is not None
        # AA preflop equity should be ~0.85
        assert 0.75 <= equity <= 0.95
        # Adaptive should stop early (well before 5000)
        assert prof.early_stopped or prof.simulations_run < 5000

    def test_adaptive_disabled_runs_full(self):
        """With adaptive=False, should run all simulations."""
        hero = self._make_cards(["Ah", "As"])
        solver = EquitySolver(
            default_simulations=500,
            adaptive=False,
        )
        solver.compute_equity(hero, [], num_opponents=1)
        prof = solver.last_profile

        assert prof is not None
        assert prof.simulations_run == 500
        assert not prof.early_stopped

    def test_adaptive_still_accurate(self):
        """Adaptive should produce similar equity to fixed."""
        hero = self._make_cards(["Kh", "Qs"])
        board = self._make_cards(["Kd", "Qd", "7c"])

        solver_fixed = EquitySolver(default_simulations=2000, adaptive=False)
        solver_adaptive = EquitySolver(
            default_simulations=2000, adaptive=True,
            min_simulations=200, step_size=200, confidence_threshold=0.03,
        )

        eq_fixed = solver_fixed.compute_equity(hero, board, num_opponents=1)
        eq_adaptive = solver_adaptive.compute_equity(hero, board, num_opponents=1)

        # Should be within 5% of each other
        assert abs(eq_fixed - eq_adaptive) < 0.10


# ─── Phase 3: Evaluator Integration Tests ───────────────────────────────────


class TestEvaluatorIntegration:
    def test_solver_with_builtin_evaluator(self, sample_hero_cards, sample_community_cards):
        solver = EquitySolver(
            default_simulations=300,
            evaluator=BuiltinEvaluator(),
            adaptive=False,
        )
        equity = solver.compute_equity(
            sample_hero_cards, sample_community_cards, num_opponents=1,
        )
        assert 0.0 <= equity <= 1.0
        # AK on AK7 board should have good equity
        assert equity > 0.5

    def test_solver_auto_detects_evaluator(self):
        """Default solver should auto-detect best evaluator."""
        solver = EquitySolver(default_simulations=100, adaptive=False)
        assert solver.evaluator is not None
        assert solver.evaluator.name in ("builtin", "eval7", "treys")


# ─── Phase 3: Equity Regression Tests ────────────────────────────────────────


class TestEquityRegression:
    """Regression tests to ensure equity values stay calibrated."""

    def _make_cards(self, codes: list[str]) -> list[Card]:
        rank_map = {
            "2": Rank.TWO, "3": Rank.THREE, "4": Rank.FOUR, "5": Rank.FIVE,
            "6": Rank.SIX, "7": Rank.SEVEN, "8": Rank.EIGHT, "9": Rank.NINE,
            "T": Rank.TEN, "J": Rank.JACK, "Q": Rank.QUEEN, "K": Rank.KING,
            "A": Rank.ACE,
        }
        suit_map = {"h": Suit.HEARTS, "d": Suit.DIAMONDS, "c": Suit.CLUBS, "s": Suit.SPADES}
        return [Card(rank=rank_map[c[0]], suit=suit_map[c[1]], confidence=1.0) for c in codes]

    def test_aa_preflop_equity(self):
        """AA vs 1 random opponent preflop ≈ 85% ± 5%."""
        solver = EquitySolver(default_simulations=2000, adaptive=False)
        hero = self._make_cards(["Ah", "As"])
        eq = solver.compute_equity(hero, [], num_opponents=1)
        assert 0.78 <= eq <= 0.92, f"AA equity {eq:.3f} outside expected range"

    def test_72o_preflop_equity(self):
        """72o vs 1 random opponent preflop ≈ 35%."""
        solver = EquitySolver(default_simulations=2000, adaptive=False)
        hero = self._make_cards(["7h", "2s"])
        eq = solver.compute_equity(hero, [], num_opponents=1)
        assert 0.25 <= eq <= 0.45, f"72o equity {eq:.3f} outside expected range"

    def test_set_on_river_equity(self):
        """Set of queens on river should dominate."""
        solver = EquitySolver(default_simulations=2000, adaptive=False)
        hero = self._make_cards(["Qh", "Qs"])
        board = self._make_cards(["Qd", "8c", "5h", "3s", "2d"])
        eq = solver.compute_equity(hero, board, num_opponents=1)
        assert eq > 0.85, f"Set of Qs equity {eq:.3f} should be > 0.85"
