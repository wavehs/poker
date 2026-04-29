import time
import sys

sys.path.insert(0, ".")

from libs.common.schemas import Card, Rank, Suit
from services.solver_core.evaluator import TreysEvaluator
from services.solver_core.solver import EquitySolver

def _card(rank_str: str, suit_str: str) -> Card:
    _ranks = {"2": Rank.TWO, "3": Rank.THREE, "4": Rank.FOUR, "5": Rank.FIVE,
              "6": Rank.SIX, "7": Rank.SEVEN, "8": Rank.EIGHT, "9": Rank.NINE,
              "T": Rank.TEN, "J": Rank.JACK, "Q": Rank.QUEEN, "K": Rank.KING, "A": Rank.ACE}
    _suits = {"c": Suit.CLUBS, "d": Suit.DIAMONDS, "h": Suit.HEARTS, "s": Suit.SPADES}
    return Card(rank=_ranks[rank_str], suit=_suits[suit_str], confidence=1.0, source="bench")

def bench_range():
    solver = EquitySolver(default_simulations=1000, evaluator=TreysEvaluator(), adaptive=False)

    # Generate larger ranges for more impact
    hero_range = [(i, j) for i in range(20) for j in range(i+1, 20)]
    villain_range = [(i, j) for i in range(20, 40) for j in range(i+1, 40)]

    board = [_card("A", "d"), _card("K", "d"), _card("7", "c")]

    t0 = time.time()
    solver.compute_range_vs_range_equity(hero_range, villain_range, board, simulations=1000)
    print("Time taken for 1000 sims:", time.time() - t0)

bench_range()
