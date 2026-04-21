"""
Solver Core — Hand evaluation, equity calculation, Monte Carlo simulation.

Deterministic poker math engine. This is the source of truth for all
recommendations — no LLM, no heuristics, only math.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Optional

from libs.common.schemas import Card, Rank, Suit, TableState


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


# ─── Hand Evaluation ────────────────────────────────────────────────────────


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
    for ka, kb in zip(a[2], b[2]):
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


# ─── Equity Solver ───────────────────────────────────────────────────────────


class EquitySolver:
    """
    Monte Carlo equity calculator.
    
    Runs simulations to estimate hand equity against random opponent ranges.
    """

    def __init__(self, default_simulations: int = 5000) -> None:
        """
        Args:
            default_simulations: Number of Monte Carlo simulations.
        """
        self.default_simulations = default_simulations

    def compute_equity(
        self,
        hole_cards: list[Card],
        community_cards: list[Card],
        num_opponents: int = 1,
        simulations: Optional[int] = None,
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

        sims = simulations or self.default_simulations
        wins = 0
        ties = 0

        # Build deck minus known cards
        known = set()
        for c in hole_cards + community_cards:
            if c.is_known:
                known.add(c.code)

        deck = self._build_deck(known)

        for _ in range(sims):
            random.shuffle(deck)
            result = self._simulate_once(
                hole_cards, community_cards, deck, num_opponents
            )
            if result > 0:
                wins += 1
            elif result == 0:
                ties += 1

        return (wins + ties * 0.5) / sims

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

    def _build_deck(self, exclude: set[str]) -> list[Card]:
        """Build a deck of all cards minus excluded ones."""
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
