"""
Hand Evaluator Abstraction — Pluggable backends for hand evaluation.

Supports three backends:
  1. BuiltinEvaluator  — current pure-Python evaluator (baseline)
  2. Eval7Evaluator    — eval7 C-extension (~100× faster)
  3. TreysEvaluator    — treys lookup-table (~5-10× faster)

All evaluators implement the same Protocol and return comparable
integer ranks (higher = better hand).

Usage:
    evaluator = get_best_evaluator()
    rank = evaluator.evaluate([0, 1, 2, 3, 4])  # 5-7 int-coded cards
"""

from __future__ import annotations

import itertools
import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ─── Integer Card Encoding ──────────────────────────────────────────────────
# Card as int: 0..51
#   rank_index = card // 4   (0=2, 1=3, ..., 12=A)
#   suit_index = card % 4    (0=clubs, 1=diamonds, 2=hearts, 3=spades)

RANK_NAMES = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SUIT_NAMES = ["c", "d", "h", "s"]


def card_to_int(rank_str: str, suit_str: str) -> int:
    """Convert rank+suit strings to int (0-51).

    Args:
        rank_str: One of '2'..'9', 'T', 'J', 'Q', 'K', 'A'
        suit_str: One of 'c', 'd', 'h', 's'
    """
    r = RANK_NAMES.index(rank_str)
    s = SUIT_NAMES.index(suit_str)
    return r * 4 + s


def int_to_card(card_int: int) -> tuple[str, str]:
    """Convert int (0-51) to (rank_str, suit_str)."""
    return RANK_NAMES[card_int // 4], SUIT_NAMES[card_int % 4]


def card_rank(card_int: int) -> int:
    """Get rank index (0=2 .. 12=A) from card int."""
    return card_int // 4


def card_suit(card_int: int) -> int:
    """Get suit index from card int."""
    return card_int % 4


# ─── Evaluator Protocol ─────────────────────────────────────────────────────


@runtime_checkable
class HandEvaluator(Protocol):
    """Protocol for hand evaluators. Higher rank = better hand."""

    @property
    def name(self) -> str:
        """Human-readable evaluator name."""
        ...

    def evaluate(self, cards: list[int]) -> int:
        """Evaluate 5-7 int-coded cards. Returns comparable int rank (higher = better)."""
        ...


# ─── Builtin Evaluator ──────────────────────────────────────────────────────


# Hand category base values (higher = better)
_HIGH_CARD = 0
_PAIR = 1_000_000
_TWO_PAIR = 2_000_000
_THREE_KIND = 3_000_000
_STRAIGHT = 4_000_000
_FLUSH = 5_000_000
_FULL_HOUSE = 6_000_000
_FOUR_KIND = 7_000_000
_STRAIGHT_FLUSH = 8_000_000


def _evaluate_five_int(c0: int, c1: int, c2: int, c3: int, c4: int) -> int:
    """Evaluate exactly 5 cards (as ints). Returns int rank (higher = better).

    Optimized: completely allocation-free. Uses an inline sorting network
    and logic instead of lists/dicts/sets.
    """
    # Extract ranks and suits
    r0, r1, r2, r3, r4 = c0 // 4, c1 // 4, c2 // 4, c3 // 4, c4 // 4
    s0, s1, s2, s3, s4 = c0 % 4, c1 % 4, c2 % 4, c3 % 4, c4 % 4

    # Inline sorting network for 5 elements (descending)
    if r0 < r1: r0, r1 = r1, r0
    if r1 < r2: r1, r2 = r2, r1
    if r2 < r3: r2, r3 = r3, r2
    if r3 < r4: r3, r4 = r4, r3
    if r0 < r1: r0, r1 = r1, r0
    if r1 < r2: r1, r2 = r2, r1
    if r2 < r3: r2, r3 = r3, r2
    if r0 < r1: r0, r1 = r1, r0
    if r1 < r2: r1, r2 = r2, r1
    if r0 < r1: r0, r1 = r1, r0

    is_flush = s0 == s1 == s2 == s3 == s4

    # Check straight
    is_straight = False
    straight_high = -1
    unique_5 = r0 != r1 and r1 != r2 and r2 != r3 and r3 != r4
    if r0 - r4 == 4 and unique_5:
        is_straight = True
        straight_high = r0
    elif r0 == 12 and r1 == 3 and r2 == 2 and r3 == 1 and r4 == 0:
        is_straight = True
        straight_high = 3

    if is_flush and is_straight:
        return _STRAIGHT_FLUSH + straight_high

    # Count duplicates based on sorted order.
    # Since it's sorted, duplicates must be adjacent.

    # 4 of a kind: A A A A B (r0==r3) or A B B B B (r1==r4)
    if r0 == r3:
        return _FOUR_KIND + r0 * 15 + r4
    if r1 == r4:
        return _FOUR_KIND + r1 * 15 + r0

    # Full house: A A A B B (r0==r2 and r3==r4) or A A B B B (r0==r1 and r2==r4)
    if r0 == r2 and r3 == r4:
        return _FULL_HOUSE + r0 * 15 + r3
    if r0 == r1 and r2 == r4:
        return _FULL_HOUSE + r2 * 15 + r0

    if is_flush:
        return _FLUSH + r0 * 15**4 + r1 * 15**3 + r2 * 15**2 + r3 * 15 + r4

    if is_straight:
        return _STRAIGHT + straight_high

    # 3 of a kind: A A A B C (r0==r2), A B B B C (r1==r3), A B C C C (r2==r4)
    if r0 == r2:
        return _THREE_KIND + r0 * 15**2 + r3 * 15 + r4
    if r1 == r3:
        return _THREE_KIND + r1 * 15**2 + r0 * 15 + r4
    if r2 == r4:
        return _THREE_KIND + r2 * 15**2 + r0 * 15 + r1

    # Two pair: A A B B C (r0==r1, r2==r3), A A B C C (r0==r1, r3==r4), A B B C C (r1==r2, r3==r4)
    if r0 == r1 and r2 == r3:
        return _TWO_PAIR + r0 * 15**2 + r2 * 15 + r4
    if r0 == r1 and r3 == r4:
        return _TWO_PAIR + r0 * 15**2 + r3 * 15 + r2
    if r1 == r2 and r3 == r4:
        return _TWO_PAIR + r1 * 15**2 + r3 * 15 + r0

    # Pair: A A B C D (r0==r1), A B B C D (r1==r2), A B C C D (r2==r3), A B C D D (r3==r4)
    if r0 == r1:
        return _PAIR + r0 * 15**3 + r2 * 15**2 + r3 * 15 + r4
    if r1 == r2:
        return _PAIR + r1 * 15**3 + r0 * 15**2 + r3 * 15 + r4
    if r2 == r3:
        return _PAIR + r2 * 15**3 + r0 * 15**2 + r1 * 15 + r4
    if r3 == r4:
        return _PAIR + r3 * 15**3 + r0 * 15**2 + r1 * 15 + r2

    # High card
    return _HIGH_CARD + r0 * 15**4 + r1 * 15**3 + r2 * 15**2 + r3 * 15 + r4


class BuiltinEvaluator:
    """Pure-Python hand evaluator using integer card encoding.

    This is the optimized version of the original evaluator — uses int
    cards instead of Card objects, avoids Counter/sorted overhead.
    """

    @property
    def name(self) -> str:
        return "builtin"

    def evaluate(self, cards: list[int]) -> int:
        """Evaluate 5-7 int-coded cards. Returns int rank (higher = better)."""
        n = len(cards)
        if n < 5:
            # Not enough cards — just sum rank values as tiebreaker
            return sum(c // 4 for c in cards)

        if n == 5:
            return _evaluate_five_int(cards[0], cards[1], cards[2], cards[3], cards[4])

        # 6-7 cards: find best 5-card combination
        best = -1
        for combo in itertools.combinations(cards, 5):
            val = _evaluate_five_int(combo[0], combo[1], combo[2], combo[3], combo[4])
            if val > best:
                best = val
        return best


# ─── Eval7 Backend ───────────────────────────────────────────────────────────


class Eval7Evaluator:
    """eval7 C-extension backend. ~100× faster than pure Python.

    Requires: pip install eval7
    """

    def __init__(self) -> None:
        import eval7 as _eval7
        self._eval7 = _eval7
        # Pre-build card lookup table: int(0-51) -> eval7.Card
        self._card_lut: list = []
        for i in range(52):
            r, s = int_to_card(i)
            self._card_lut.append(_eval7.Card(r + s))

    @property
    def name(self) -> str:
        return "eval7"

    def evaluate(self, cards: list[int]) -> int:
        """Evaluate using eval7. Returns int rank (higher = better)."""
        hand = [self._card_lut[c] for c in cards]
        # eval7.evaluate returns lower = better, so we negate
        return -self._eval7.evaluate(hand)


# ─── Treys Backend ───────────────────────────────────────────────────────────


class TreysEvaluator:
    """treys lookup-table backend. ~5-10× faster than pure Python.

    Requires: pip install treys
    """

    def __init__(self) -> None:
        from treys import Card as _TreysCard
        from treys import Evaluator as _TreysEval
        self._evaluator = _TreysEval()
        self._treys_card = _TreysCard
        # Pre-build card lookup table: int(0-51) -> treys int card
        self._card_lut: list[int] = []
        for i in range(52):
            r, s = int_to_card(i)
            self._card_lut.append(_TreysCard.new(r + s))

    @property
    def name(self) -> str:
        return "treys"

    def evaluate(self, cards: list[int]) -> int:
        """Evaluate using treys. Returns int rank (higher = better).

        treys uses 5-card evaluation with separate board/hand.
        For 7 cards, we find the best 5-card combination.
        For 5 cards, we split as 2 hand + 3 board (arbitrary, treys needs this split).
        """
        treys_cards = [self._card_lut[c] for c in cards]

        if len(treys_cards) == 5:
            # treys.evaluate needs hand(2) + board(3-5)
            score = self._evaluator.evaluate(treys_cards[2:], treys_cards[:2])
        elif len(treys_cards) >= 6:
            # Find best 5-card combination
            best = 999_999
            for combo in itertools.combinations(treys_cards, 5):
                score = self._evaluator.evaluate(list(combo[2:]), list(combo[:2]))
                if score < best:
                    best = score
            score = best
        else:
            return 0

        # treys: lower score = better hand, max rank = 7462
        # Invert so higher = better (consistent with our protocol)
        return 7463 - score


# ─── Auto-detection ──────────────────────────────────────────────────────────


def get_best_evaluator() -> HandEvaluator:
    """Auto-detect and return the best available evaluator.

    Priority: eval7 > treys > builtin
    """
    # Try eval7 first (fastest)
    try:
        ev = Eval7Evaluator()
        logger.info("Using eval7 evaluator (C-extension)")
        return ev
    except (ImportError, Exception) as e:
        logger.debug("eval7 not available: %s", e)

    # Try treys
    try:
        ev = TreysEvaluator()
        logger.info("Using treys evaluator (lookup-table)")
        return ev
    except (ImportError, Exception) as e:
        logger.debug("treys not available: %s", e)

    # Fallback to builtin
    logger.info("Using builtin evaluator (pure Python)")
    return BuiltinEvaluator()


def get_evaluator_by_name(name: str) -> HandEvaluator:
    """Get a specific evaluator by name.

    Args:
        name: 'eval7', 'treys', or 'builtin'

    Raises:
        ValueError: If the evaluator is not available.
    """
    if name == "eval7":
        return Eval7Evaluator()
    elif name == "treys":
        return TreysEvaluator()
    elif name == "builtin":
        return BuiltinEvaluator()
    else:
        raise ValueError(f"Unknown evaluator: {name!r}. Choose from: eval7, treys, builtin")
