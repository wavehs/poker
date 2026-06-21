import time
import sys
import itertools
sys.path.insert(0, ".")
from services.solver_core.evaluator import TreysEvaluator

class FastTreysEvaluator(TreysEvaluator):
    def evaluate(self, cards: list[int]) -> int:
        treys_cards = [self._card_lut[c] for c in cards]

        if len(treys_cards) == 5:
            score = self._evaluator.evaluate(treys_cards[2:], treys_cards[:2])
        elif len(treys_cards) >= 6:
            best = 999_999
            for combo in itertools.combinations(treys_cards, 5):
                score = self._evaluator.evaluate(combo[2:], combo[:2])
                if score < best:
                    best = score
            score = best
        else:
            return 0
        return 7463 - score

def bench_treys():
    ev = FastTreysEvaluator()
    cards_7 = [0, 1, 2, 3, 4, 5, 6]

    t0 = time.time()
    for _ in range(100000):
        ev.evaluate(cards_7)
    t1 = time.time()
    print("Time for 100000 evals (7 cards, optimized):", t1 - t0)

bench_treys()
