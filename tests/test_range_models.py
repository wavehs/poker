import pytest
from libs.common.schemas import TableState, Street, PlayStyle
from services.policy_layer.range_models import estimate_opponent_range, range_to_cards, RANGES

def test_estimate_opponent_range_preflop():
    state = TableState(street=Street.PREFLOP)

    # Check default balanced range
    balanced_range = estimate_opponent_range(state, PlayStyle.BALANCED)
    assert "AA" in balanced_range
    assert "77" in balanced_range
    assert "AKs" in balanced_range
    assert len(balanced_range) == len(RANGES[PlayStyle.BALANCED])

    # Check aggressive range
    aggro_range = estimate_opponent_range(state, PlayStyle.AGGRESSIVE)
    assert len(aggro_range) > len(balanced_range)
    assert "55" in aggro_range

    # Check conservative range
    tight_range = estimate_opponent_range(state, PlayStyle.CONSERVATIVE)
    assert len(tight_range) < len(balanced_range)
    assert "77" not in tight_range

def test_estimate_opponent_range_postflop_narrowing():
    state_preflop = TableState(street=Street.PREFLOP)
    state_flop = TableState(street=Street.FLOP)

    range_preflop = estimate_opponent_range(state_preflop, PlayStyle.BALANCED)
    range_flop = estimate_opponent_range(state_flop, PlayStyle.BALANCED)

    # Some weak hands should be folded
    assert "77" in range_preflop
    assert "77" not in range_flop

    assert len(range_flop) < len(range_preflop)

def test_range_to_cards():
    # Pair should yield 6 combinations
    cards_pair = range_to_cards({"AA"})
    assert len(cards_pair) == 6
    for c1, c2 in cards_pair:
        assert c1 != c2

    # Suited should yield 4 combinations
    cards_suited = range_to_cards({"AKs"})
    assert len(cards_suited) == 4

    # Offsuit should yield 12 combinations
    cards_offsuit = range_to_cards({"AKo"})
    assert len(cards_offsuit) == 12

def test_range_to_cards_invalid_str():
    # Should ignore invalid strings and return empty list
    cards = range_to_cards({"XX", "12s", "A"})
    assert len(cards) == 0
