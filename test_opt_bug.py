from services.solver_core.solver import EquitySolver
from libs.common.schemas import Card, Rank, Suit
import pytest

def test_no_name_error():
    solver = EquitySolver(enable_cache=False)

    # 2 hearts vs 3 spades
    hero_range = [(0, 1), (2, 3)]
    villain_range = [(4, 5), (6, 7)]
    community_cards = []

    res = solver.compute_range_vs_range_equity(hero_range, villain_range, community_cards, simulations=10)
    print(res)

test_no_name_error()
