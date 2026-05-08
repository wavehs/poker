import sys
sys.path.insert(0, ".")
from services.solver_core.solver import EquitySolver
from services.solver_core.evaluator import TreysEvaluator
from libs.common.schemas import Card, Rank, Suit

class FastSolver(EquitySolver):
    def compute_range_vs_range_equity(self, hero_range_cards, villain_range_cards, community_cards, simulations=None):
        import random
        if not hero_range_cards or not villain_range_cards:
            return {}

        sims = simulations or self.default_simulations

        from services.solver_core.solver import _card_obj_to_int

        board_ints = [_card_obj_to_int(c) for c in community_cards if c.is_known]

        known = set(board_ints)
        deck = [c for c in self._full_deck if c not in known]

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

        for _ in range(sims):
            v_hand = random.choice(valid_v_hands)

            if len(deck) - 2 < cards_needed:
                continue

            # Optimize board sampling as mentioned in .jules/bolt.md
            # "random.sample is much faster for drawing hands since it avoids shuffling the entire deck array."
            # Actually the bolt.md talks about Deck Sampling optimization, but the current code uses while loop.
            # "Contrary to intuition, in performance-critical Monte Carlo simulations drawing a small number of cards, using a while loop with random.choice() and linear membership checks is roughly 2x faster than using random.sample() over a dynamically filtered deck (via list comprehension)."
            # Let's keep the while loop for board sampling.
            sampled_board: list[int] = []
            while len(sampled_board) < cards_needed:
                s = random.choice(deck)
                if s not in v_hand and s not in sampled_board:
                    sampled_board.append(s)

            full_board = board_ints + sampled_board

            v_rank = self.evaluator.evaluate(list(v_hand) + full_board)

            # Using a set is what memory says is ~4x faster than repeated list/tuple membership checks inside the hot loop.
            forbidden = set(v_hand)
            forbidden.update(sampled_board)

            evaluate = self.evaluator.evaluate

            for h_hand in valid_h_hands:
                if h_hand[0] in forbidden or h_hand[1] in forbidden:
                    continue

                h_rank = evaluate(list(h_hand) + full_board)

                if h_rank > v_rank:
                    wins[h_hand] += 1
                elif h_rank == v_rank:
                    ties[h_hand] += 1
                totals[h_hand] += 1

        if self.enable_cache:
            self._board_cache = None

        distribution = {}
        for h_hand in valid_h_hands:
            if totals[h_hand] > 0:
                distribution[h_hand] = (wins[h_hand] + ties[h_hand] * 0.5) / totals[h_hand]
            else:
                distribution[h_hand] = 0.0

        return distribution

def _card(rank_str: str, suit_str: str) -> Card:
    _ranks = {"2": Rank.TWO, "3": Rank.THREE, "4": Rank.FOUR, "5": Rank.FIVE,
              "6": Rank.SIX, "7": Rank.SEVEN, "8": Rank.EIGHT, "9": Rank.NINE,
              "T": Rank.TEN, "J": Rank.JACK, "Q": Rank.QUEEN, "K": Rank.KING, "A": Rank.ACE}
    _suits = {"c": Suit.CLUBS, "d": Suit.DIAMONDS, "h": Suit.HEARTS, "s": Suit.SPADES}
    return Card(rank=_ranks[rank_str], suit=_suits[suit_str], confidence=1.0, source="bench")

def bench_range():
    solver = FastSolver(default_simulations=1000, evaluator=TreysEvaluator(), adaptive=False)

    hero_range = [(i, j) for i in range(20) for j in range(i+1, 20)]
    villain_range = [(i, j) for i in range(20, 40) for j in range(i+1, 40)]

    board = [_card("A", "d"), _card("K", "d"), _card("7", "c")]

    import time
    t0 = time.time()
    solver.compute_range_vs_range_equity(hero_range, villain_range, board, simulations=1000)
    print("Time taken for 1000 sims (set update):", time.time() - t0)

bench_range()
