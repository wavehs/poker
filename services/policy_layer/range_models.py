"""
Opponent range modelling module.
Provides default preflop ranges based on opponent play style and
methods to estimate and narrow range based on streets and actions.
"""
from typing import List, Set, Tuple
import itertools

from libs.common.schemas import PlayStyle, Street, TableState
from services.solver_core.evaluator import RANK_NAMES, SUIT_NAMES, card_to_int

# Default preflop ranges defined as sets of hand string representations
# Format: "AA" for pairs, "AKs" for suited, "AKo" for offsuit
RANGES: dict[PlayStyle, Set[str]] = {
    PlayStyle.CONSERVATIVE: {
        "AA", "KK", "QQ", "JJ", "TT", "99",
        "AKs", "AQs", "AJs",
        "AKo", "AQo"
    },
    PlayStyle.BALANCED: {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77",
        "AKs", "AQs", "AJs", "ATs", "KQs", "KJs",
        "AKo", "AQo", "AJo", "KQo"
    },
    PlayStyle.AGGRESSIVE: {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55",
        "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s",
        "KQs", "KJs", "KTs", "QJs", "QTs", "JTs", "T9s", "98s", "87s", "76s",
        "AKo", "AQo", "AJo", "ATo", "KQo", "KJo", "QJo"
    },
}

def estimate_opponent_range(state: TableState, play_style: PlayStyle) -> Set[str]:
    """
    Estimates the opponent's range based on their play style and the current street.
    Narrows the range as the hand progresses (e.g., folding weaker hands).
    """
    base_range = RANGES.get(play_style, RANGES[PlayStyle.BALANCED]).copy()

    # Range narrowing logic by street
    if state.street in (Street.FLOP, Street.TURN, Street.RIVER):
        # Remove weaker preflop holdings assuming they folded to preflop/flop action
        weak_hands = {
            "55", "66", "77",
            "A5s", "A6s", "A7s", "A8s", "A9s",
            "87s", "76s", "98s",
            "ATo", "KJo", "QJo"
        }
        base_range = base_range - weak_hands

    return base_range

def range_to_cards(hand_range: Set[str]) -> List[Tuple[int, int]]:
    """
    Converts a set of hand strings to a list of integer card pairs.
    Each returned tuple contains two integers (0-51) representing specific cards.
    """
    cards_in_range = []

    for hand_str in hand_range:
        if len(hand_str) == 2:
            # Pair: e.g. "AA"
            r = hand_str[0]
            if r not in RANK_NAMES:
                continue
            cards = [card_to_int(r, s) for s in SUIT_NAMES]
            cards_in_range.extend(list(itertools.combinations(cards, 2)))

        elif len(hand_str) == 3:
            # Non-pair: e.g. "AKs", "AKo"
            r1, r2, type_ = hand_str
            if r1 not in RANK_NAMES or r2 not in RANK_NAMES:
                continue

            if type_ == "s":
                # Suited
                for s in SUIT_NAMES:
                    cards_in_range.append((card_to_int(r1, s), card_to_int(r2, s)))
            elif type_ == "o":
                # Offsuit
                for s1 in SUIT_NAMES:
                    for s2 in SUIT_NAMES:
                        if s1 != s2:
                            cards_in_range.append((card_to_int(r1, s1), card_to_int(r2, s2)))

    return cards_in_range
