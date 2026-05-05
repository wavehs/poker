"""
Solver Core — Hand evaluation, equity calculation, Monte Carlo simulation.

Deterministic poker math engine. This is the source of truth for all
recommendations — no LLM, no heuristics, only math.

Phase 3: Optimized with pluggable evaluators, adaptive Monte Carlo,
LRU caching, integer card representation, and fine-grained profiling.
"""

from __future__ import annotations

import math
import random
import time
from collections import Counter

from libs.common.schemas import Card, Rank, Suit
from services.solver_core.evaluator import (
    HandEvaluator,
    get_best_evaluator,
)

# ─── Constants ────────────────────────────────────────────────────────────────

RANK_ORDER = {r: i for i, r in enumerate(
    [Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX, Rank.SEVEN,
     Rank.EIGHT, Rank.NINE, Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE]
)}

HAND_RANKS = {
    "high_card": 0,
    "pair": 1,
    "two_pair": 2,
    "three_of_a_kind": 3,
    "straight": 4,
    "flush": 5,
    "full_house": 6,
    "four_of_a_kind": 7,
    "straight_flush": 8,
    "royal_flush": 9,
}

# Rank.value -> int mapping for fast Card->int conversion
_RANK_STR_TO_IDX = {r.value: i for i, r in enumerate(
    [Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX, Rank.SEVEN,
     Rank.EIGHT, Rank.NINE, Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE]
)}
_SUIT_STR_TO_IDX = {"c": 0, "d": 1, "h": 2, "s": 3}


def _card_obj_to_int(card: Card) -> int:
    """Convert a Card pydantic model to int (0-51) for fast evaluation."""
    r = _RANK_STR_TO_IDX.get(card.rank.value, -1)
    s = _SUIT_STR_TO_IDX.get(card.suit.value, -1)
    if r < 0 or s < 0:
        return -1
    return r * 4 + s


# ─── Hand Evaluation (legacy API, preserved) ────────────────────────────────


def rank_value(rank: Rank) -> int:
    """Get numeric value of a rank (2=0 ... A=12)."""
    return RANK_ORDER.get(rank, -1)


def evaluate_hand(cards: list[Card]) -> tuple[int, str, list[int]]:
    """
    Evaluate the best 5-card hand from a list of cards.

    Args:
        cards: List of Card objects (5-7 cards).

    Returns:
        Tuple of (hand_rank_value, hand_name, kickers).
        Higher rank value = better hand.
    """
    if len(cards) < 5:
        return 0, "high_card", [rank_value(c.rank) for c in cards]

    # If more than 5 cards, find best 5-card combination
    if len(cards) > 5:
        best: tuple[int, str, list[int]] = (0, "high_card", [])
        for combo in _combinations(cards, 5):
            result = _evaluate_five(combo)
            if _compare_hands(result, best) > 0:
                best = result
        return best

    return _evaluate_five(cards)


def _evaluate_five(cards: list[Card]) -> tuple[int, str, list[int]]:
    """Evaluate exactly 5 cards."""
    ranks = sorted([rank_value(c.rank) for c in cards], reverse=True)
    suits = [c.suit for c in cards]
    rank_counts = Counter(ranks)

    is_flush = len(set(suits)) == 1
    is_straight, high = _check_straight(ranks)

    # Straight flush / Royal flush
    if is_flush and is_straight:
        if high == 12:  # Ace high
            return HAND_RANKS["royal_flush"], "royal_flush", [high]
        return HAND_RANKS["straight_flush"], "straight_flush", [high]

    # Four of a kind
    fours = [r for r, c in rank_counts.items() if c == 4]
    if fours:
        kicker = [r for r in ranks if r != fours[0]]
        return HAND_RANKS["four_of_a_kind"], "four_of_a_kind", fours + kicker[:1]

    # Full house
    threes = [r for r, c in rank_counts.items() if c == 3]
    pairs = [r for r, c in rank_counts.items() if c == 2]
    if threes and pairs:
        return HAND_RANKS["full_house"], "full_house", threes + pairs

    # Flush
    if is_flush:
        return HAND_RANKS["flush"], "flush", ranks

    # Straight
    if is_straight:
        return HAND_RANKS["straight"], "straight", [high]

    # Three of a kind
    if threes:
        kickers = sorted([r for r in ranks if r != threes[0]], reverse=True)
        return HAND_RANKS["three_of_a_kind"], "three_of_a_kind", threes + kickers[:2]

    # Two pair
    if len(pairs) >= 2:
        pairs_sorted = sorted(pairs, reverse=True)
        kicker = [r for r in ranks if r not in pairs_sorted]
        return HAND_RANKS["two_pair"], "two_pair", pairs_sorted[:2] + kicker[:1]

    # One pair
    if pairs:
        kickers = sorted([r for r in ranks if r != pairs[0]], reverse=True)
        return HAND_RANKS["pair"], "pair", pairs + kickers[:3]

    # High card
    return HAND_RANKS["high_card"], "high_card", ranks


def _check_straight(ranks: list[int]) -> tuple[bool, int]:
    """Check if sorted ranks form a straight. Returns (is_straight, high_card)."""
    unique = sorted(set(ranks), reverse=True)
    if len(unique) < 5:
        return False, -1

    # Normal straight
    for i in range(len(unique) - 4):
        window = unique[i:i + 5]
        if window[0] - window[4] == 4:
            return True, window[0]

    # Wheel (A-2-3-4-5)
    if set(unique) >= {12, 0, 1, 2, 3}:
        return True, 3  # 5-high straight

    return False, -1


def _compare_hands(
    a: tuple[int, str, list[int]],
    b: tuple[int, str, list[int]],
) -> int:
    """Compare two hand evaluations. Returns >0 if a wins, <0 if b wins, 0 if tie."""
    if a[0] != b[0]:
        return a[0] - b[0]
    for ka, kb in zip(a[2], b[2], strict=False):
        if ka != kb:
            return ka - kb
    return 0


def _combinations(items: list, r: int) -> list[list]:
    """Generate all r-length combinations of items."""
    if r == 0:
        return [[]]
    if not items:
        return []
    result = []
    for i, item in enumerate(items):
        for combo in _combinations(items[i + 1:], r - 1):
            result.append([item] + combo)
    return result


# ─── Solver Profiling ────────────────────────────────────────────────────────


class SolverProfile:
    """Fine-grained profiling data for a single compute_equity() call."""

    __slots__ = (
        "total_ms", "deck_build_ms", "simulation_ms",
        "evaluate_calls", "simulations_run", "early_stopped",
        "cache_hits", "cache_misses",
    )

    def __init__(self) -> None:
        self.total_ms: float = 0.0
        self.deck_build_ms: float = 0.0
        self.simulation_ms: float = 0.0
        self.evaluate_calls: int = 0
        self.simulations_run: int = 0
        self.early_stopped: bool = False
        self.cache_hits: int = 0
        self.cache_misses: int = 0

    def as_dict(self) -> dict:
        return {
            "total_ms": round(self.total_ms, 2),
            "deck_build_ms": round(self.deck_build_ms, 2),
            "simulation_ms": round(self.simulation_ms, 2),
            "evaluate_calls": self.evaluate_calls,
            "simulations_run": self.simulations_run,
            "early_stopped": self.early_stopped,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }

    def __repr__(self) -> str:
        return (
            f"SolverProfile(total={self.total_ms:.1f}ms, "
            f"sims={self.simulations_run}, evals={self.evaluate_calls}, "
            f"early_stop={self.early_stopped}, "
            f"cache_h={self.cache_hits}/m={self.cache_misses})"
        )


# ─── Equity Solver ───────────────────────────────────────────────────────────


class EquitySolver:
    """
    Monte Carlo equity calculator.

    Phase 3 optimizations:
    - Pluggable evaluator backend (eval7, treys, builtin)
    - Integer card representation in hot path
    - LRU cache for board evaluations
    - Adaptive Monte Carlo with early stopping
    - Fine-grained profiling
    """

    def __init__(
        self,
        default_simulations: int = 5000,
        evaluator: HandEvaluator | None = None,
        adaptive: bool = True,
        min_simulations: int = 200,
        step_size: int = 200,
        confidence_threshold: float = 0.03,
        enable_cache: bool = True,
    ) -> None:
        """
        Args:
            default_simulations: Max number of Monte Carlo simulations.
            evaluator: Hand evaluator backend (auto-detected if None).
            adaptive: Enable adaptive Monte Carlo (early stop when converged).
            min_simulations: Minimum sims before early stopping.
            step_size: Check convergence every N simulations.
            confidence_threshold: Stop if 95% CI half-width < this value.
            enable_cache: Enable LRU cache for board evaluations.
        """
        self.default_simulations = default_simulations
        self.evaluator = evaluator or get_best_evaluator()
        self.adaptive = adaptive
        self.min_simulations = min_simulations
        self.step_size = step_size
        self.confidence_threshold = confidence_threshold
        self.enable_cache = enable_cache

        # Build full int-deck once (0..51)
        self._full_deck = list(range(52))

        # Profiling
        self.last_profile: SolverProfile | None = None

        # Board evaluation cache (cleared per equity call)
        self._board_cache: dict[tuple[int, ...], int] | None = None

    def compute_equity(
        self,
        hole_cards: list[Card],
        community_cards: list[Card],
        num_opponents: int = 1,
        simulations: int | None = None,
    ) -> float:
        """
        Estimate equity vs random hands via Monte Carlo simulation.

        Args:
            hole_cards: Hero's 2 hole cards.
            community_cards: 0-5 community cards.
            num_opponents: Number of opponents.
            simulations: Override default simulation count.

        Returns:
            Equity as float [0, 1].
        """
        # Call the range-based method with a None range to fall back to random cards
        return self.compute_equity_vs_range(
            hole_cards=hole_cards,
            community_cards=community_cards,
            opponent_range_cards=None,
            num_opponents=num_opponents,
            simulations=simulations,
        )

    def compute_equity_vs_range(
        self,
        hole_cards: list[Card],
        community_cards: list[Card],
        opponent_range_cards: list[tuple[int, int]] | None = None,
        num_opponents: int = 1,
        simulations: int | None = None,
    ) -> float:
        """
        Estimate equity via Monte Carlo simulation.

        Args:
            hole_cards: Hero's 2 hole cards.
            community_cards: 0-5 community cards.
            num_opponents: Number of opponents.
            simulations: Override default simulation count.

        Returns:
            Equity as float [0, 1].
        """
        if not hole_cards or len(hole_cards) < 2:
            return 0.0
        if not all(c.is_known for c in hole_cards):
            return 0.0

        profile = SolverProfile()
        t0 = time.perf_counter()

        sims = simulations or self.default_simulations

        # ── Convert to int representation
        t_deck = time.perf_counter()
        hero_ints = [_card_obj_to_int(c) for c in hole_cards]
        board_ints = [_card_obj_to_int(c) for c in community_cards if c.is_known]

        # Validate cards
        if any(c < 0 for c in hero_ints):
            return 0.0

        # Build deck excluding known cards
        known = set(hero_ints + board_ints)
        deck = [c for c in self._full_deck if c not in known]
        profile.deck_build_ms = (time.perf_counter() - t_deck) * 1000

        # Pre-filter range cards to exclude combinations conflicting with known cards
        valid_range_cards: list[tuple[int, int]] | None = None
        if opponent_range_cards is not None:
            temp_range = []
            for combo in opponent_range_cards:
                if combo[0] not in known and combo[1] not in known:
                    temp_range.append(combo)

            # If after filtering the range is empty, fallback to random
            if temp_range:
                valid_range_cards = temp_range

        # ── Run Monte Carlo
        t_sim = time.perf_counter()
        cards_needed = 5 - len(board_ints)

        # When picking random hands, we need cards for the board + opp hands
        # When using range, we draw the opponent hand separately
        # so we just need board cards + cards for any *other* random opponents
        random_cards_needed = cards_needed
        if valid_range_cards is None:
            random_cards_needed += num_opponents * 2
        else:
            random_cards_needed += max(0, num_opponents - 1) * 2

        wins = 0
        ties = 0
        total_run = 0

        # Reset board cache for this call
        if self.enable_cache:
            self._board_cache = {}

        forbidden = [False] * 52
        for b in board_ints:
            forbidden[b] = True

        for batch_start in range(0, sims, self.step_size):
            batch_end = min(batch_start + self.step_size, sims)

            for _ in range(batch_start, batch_end):
                if valid_range_cards:
                    opp_hand = random.choice(valid_range_cards)
                    # Fast check to avoid list comprehension if not enough cards
                    if len(deck) - 2 < random_cards_needed:
                        continue

                    # Instead of creating current_deck every loop, sample and then filter.
                    # This is slightly faster on average if random_cards_needed is small.
                    # Or we can just resample if we hit one of the 2 opponent cards.
                    sampled_deck: list[int] = []
                    forbidden[opp_hand[0]] = True
                    forbidden[opp_hand[1]] = True

                    while len(sampled_deck) < random_cards_needed:
                        s = random.choice(deck)
                        if not forbidden[s]:
                            forbidden[s] = True
                            sampled_deck.append(s)

                    # Inject the opponent hand into sampled_deck to match
                    # the signature expectation where opponents are drawn sequentially
                    # _simulate_once_int draws cards_needed first for the board.
                    board_samples = sampled_deck[:cards_needed]
                    other_opp_samples = sampled_deck[cards_needed:]

                    forbidden[opp_hand[0]] = False
                    forbidden[opp_hand[1]] = False
                    for s in sampled_deck:
                        forbidden[s] = False

                    # Construct expected deck: board + opp1 + opp2...
                    custom_deck = board_samples + list(opp_hand) + other_opp_samples

                    result = self._simulate_once_int(
                        hero_ints, board_ints, custom_deck, num_opponents,
                        cards_needed, profile,
                    )
                else:
                    sampled_deck = random.sample(deck, random_cards_needed)
                    result = self._simulate_once_int(
                        hero_ints, board_ints, sampled_deck, num_opponents,
                        cards_needed, profile,
                    )
                if result > 0:
                    wins += 1
                elif result == 0:
                    ties += 1
                total_run += 1

            # ── Adaptive early stopping
            if self.adaptive and total_run >= self.min_simulations:
                equity_est = (wins + ties * 0.5) / total_run
                # Wilson score interval half-width approximation
                if total_run > 0:
                    std_err = math.sqrt(equity_est * (1 - equity_est) / total_run)
                    half_width = 1.96 * std_err  # 95% CI
                    if half_width < self.confidence_threshold:
                        profile.early_stopped = True
                        break

        profile.simulations_run = total_run
        profile.simulation_ms = (time.perf_counter() - t_sim) * 1000
        profile.total_ms = (time.perf_counter() - t0) * 1000

        if self.enable_cache and self._board_cache is not None:
            profile.cache_hits = getattr(self, '_cache_hits', 0)
            profile.cache_misses = getattr(self, '_cache_misses', 0)

        self.last_profile = profile

        # Clear cache
        self._board_cache = None
        self._cache_hits = 0
        self._cache_misses = 0

        if total_run == 0:
            return 0.0

        return (wins + ties * 0.5) / total_run

    def _simulate_once_int(
        self,
        hero_ints: list[int],
        board_ints: list[int],
        deck: list[int],
        num_opponents: int,
        cards_needed: int,
        profile: SolverProfile,
    ) -> int:
        """
        Run a single Monte Carlo simulation using int cards.

        Returns:
            1 if hero wins, 0 if tie, -1 if hero loses.
        """
        idx = 0

        # Deal remaining community cards
        board = board_ints + deck[idx:idx + cards_needed]
        idx += cards_needed

        # Evaluate hero (use cache for board part)
        hero_all = hero_ints + board
        hero_rank = self.evaluator.evaluate(hero_all)
        profile.evaluate_calls += 1

        # Evaluate opponents
        best_opp_rank = -1
        for _ in range(num_opponents):
            opp_cards = deck[idx:idx + 2]
            idx += 2
            opp_all = opp_cards + board
            opp_rank = self.evaluator.evaluate(opp_all)
            profile.evaluate_calls += 1
            if opp_rank > best_opp_rank:
                best_opp_rank = opp_rank

        if hero_rank > best_opp_rank:
            return 1
        elif hero_rank == best_opp_rank:
            return 0
        else:
            return -1

    def compute_range_vs_range_equity(
        self,
        hero_range_cards: list[tuple[int, int]],
        villain_range_cards: list[tuple[int, int]],
        community_cards: list[Card],
        simulations: int | None = None,
    ) -> dict[tuple[int, int], float]:
        """
        Compute equity for each hand in the hero's range against the entire
        villain range via Monte Carlo simulation.

        Returns:
            Dictionary mapping each valid hero hand tuple to its equity [0, 1].
        """
        if not hero_range_cards or not villain_range_cards:
            return {}

        sims = simulations or self.default_simulations
        board_ints = [_card_obj_to_int(c) for c in community_cards if c.is_known]

        # Filter known cards
        known = set(board_ints)
        deck = [c for c in self._full_deck if c not in known]

        # Pre-filter valid ranges
        valid_v_hands = []
        for combo in villain_range_cards:
            if combo[0] not in known and combo[1] not in known:
                valid_v_hands.append(combo)

        valid_h_hands = []
        for combo in hero_range_cards:
            if combo[0] not in known and combo[1] not in known:
                valid_h_hands.append(combo)

        if not valid_v_hands or not valid_h_hands:
            return {}

        wins = {h: 0.0 for h in valid_h_hands}
        ties = {h: 0.0 for h in valid_h_hands}
        totals = {h: 0.0 for h in valid_h_hands}

        cards_needed = 5 - len(board_ints)

        if self.enable_cache:
            self._board_cache = {}

        forbidden = [False] * 52
        for b in board_ints:
            forbidden[b] = True

        for _ in range(sims):
            v_hand = random.choice(valid_v_hands)

            # Fast check
            if len(deck) - 2 < cards_needed:
                continue

            # Draw board
            sampled_board: list[int] = []

            forbidden[v_hand[0]] = True
            forbidden[v_hand[1]] = True

            while len(sampled_board) < cards_needed:
                s = random.choice(deck)
                if not forbidden[s]:
                    forbidden[s] = True
                    sampled_board.append(s)

            full_board = board_ints + sampled_board

            # Evaluate villain
            v_rank = self.evaluator.evaluate(list(v_hand) + full_board)

            # Evaluate hero hands
            for h_hand in valid_h_hands:
                if forbidden[h_hand[0]] or forbidden[h_hand[1]]:
                    continue

                h_rank = self.evaluator.evaluate(list(h_hand) + full_board)

                if h_rank > v_rank:
                    wins[h_hand] += 1
                elif h_rank == v_rank:
                    ties[h_hand] += 1
                totals[h_hand] += 1

            # Reset forbidden state
            forbidden[v_hand[0]] = False
            forbidden[v_hand[1]] = False
            for s in sampled_board:
                forbidden[s] = False

        if self.enable_cache:
            self._board_cache = None

        distribution = {}
        for h_hand in valid_h_hands:
            if totals[h_hand] > 0:
                distribution[h_hand] = (wins[h_hand] + ties[h_hand] * 0.5) / totals[h_hand]
            else:
                distribution[h_hand] = 0.0

        return distribution

    def compute_hand_strength(
        self,
        hole_cards: list[Card],
        community_cards: list[Card],
    ) -> float:
        """
        Compute relative hand strength [0, 1].

        Returns how strong the current made hand is relative to
        the best possible hand.
        """
        all_cards = hole_cards + community_cards
        if len(all_cards) < 2:
            return 0.0

        hand_val, _, _ = evaluate_hand(all_cards)
        # Normalize: royal_flush = 9, high_card = 0
        return hand_val / 9.0

    def compute_pot_odds(self, pot: float, to_call: float) -> float:
        """
        Compute pot odds.

        Args:
            pot: Current pot size.
            to_call: Amount to call.

        Returns:
            Pot odds as fraction [0, 1]. E.g., 0.25 means need 25% equity to call.
        """
        if pot + to_call <= 0:
            return 0.0
        return to_call / (pot + to_call)

    def compute_spr(self, effective_stack: float, pot_size: float) -> float:
        """
        Compute stack-to-pot ratio (SPR).

        Args:
            effective_stack: The effective stack size.
            pot_size: Current pot size.

        Returns:
            SPR as a float.
        """
        if pot_size <= 0:
            return float("inf")
        return effective_stack / pot_size

    def _build_deck(self, exclude: set[str]) -> list[Card]:
        """Build a deck of all cards minus excluded ones.

        Legacy method — kept for backward compatibility.
        New code uses integer deck directly.
        """
        deck: list[Card] = []
        for r in Rank:
            if r == Rank.UNKNOWN:
                continue
            for s in Suit:
                if s == Suit.UNKNOWN:
                    continue
                code = f"{r.value}{s.value}"
                if code not in exclude:
                    deck.append(Card(rank=r, suit=s, confidence=1.0, source="deck"))
        return deck

    def _simulate_once(
        self,
        hole_cards: list[Card],
        community_cards: list[Card],
        deck: list[Card],
        num_opponents: int,
    ) -> int:
        """
        Run a single Monte Carlo simulation.

        Legacy method — kept for backward compatibility.
        New code uses _simulate_once_int() for performance.

        Returns:
            1 if hero wins, 0 if tie, -1 if hero loses.
        """
        # Deal remaining community cards
        cards_needed = 5 - len(community_cards)
        idx = 0

        board = list(community_cards)
        for _ in range(cards_needed):
            board.append(deck[idx])
            idx += 1

        # Evaluate hero
        hero_hand = evaluate_hand(hole_cards + board)

        # Deal and evaluate opponents
        best_opponent: tuple[int, str, list[int]] = (0, "high_card", [])
        for _ in range(num_opponents):
            opp_cards = [deck[idx], deck[idx + 1]]
            idx += 2
            opp_hand = evaluate_hand(opp_cards + board)
            if _compare_hands(opp_hand, best_opponent) > 0:
                best_opponent = opp_hand

        result = _compare_hands(hero_hand, best_opponent)
        if result > 0:
            return 1
        elif result == 0:
            return 0
        else:
            return -1
