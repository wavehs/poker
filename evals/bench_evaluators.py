"""
Benchmark Suite — Evaluator comparison and solver performance benchmarks.

Compares all available evaluator backends and measures equity solver
performance across different scenarios.

Usage:
    python evals/bench_evaluators.py
"""

from __future__ import annotations

import random
import sys
import time

# Add project root to path
sys.path.insert(0, ".")

from services.solver_core.evaluator import (
    BuiltinEvaluator,
    HandEvaluator,
    int_to_card,
)


def _get_available_evaluators() -> list[HandEvaluator]:
    """Get all available evaluator backends."""
    evaluators: list[HandEvaluator] = [BuiltinEvaluator()]

    try:
        from services.solver_core.evaluator import Eval7Evaluator
        evaluators.append(Eval7Evaluator())
    except (ImportError, Exception):
        pass

    try:
        from services.solver_core.evaluator import TreysEvaluator
        evaluators.append(TreysEvaluator())
    except (ImportError, Exception):
        pass

    return evaluators


def _generate_random_hands(n: int, cards_per_hand: int = 7) -> list[list[int]]:
    """Generate N random hands of given size."""
    all_cards = list(range(52))
    hands = []
    for _ in range(n):
        hand = random.sample(all_cards, cards_per_hand)
        hands.append(hand)
    return hands


def _format_cards(cards: list[int]) -> str:
    """Format int cards as human-readable string."""
    return " ".join(f"{int_to_card(c)[0]}{int_to_card(c)[1]}" for c in cards)


def bench_evaluate(evaluators: list[HandEvaluator], num_hands: int = 10000) -> dict:
    """Benchmark evaluate() for each evaluator on random 7-card hands."""
    print(f"\n{'='*70}")
    print(f"  BENCHMARK: evaluate() — {num_hands} random 7-card hands")
    print(f"{'='*70}\n")

    hands_7 = _generate_random_hands(num_hands, 7)
    hands_5 = _generate_random_hands(num_hands, 5)

    results = {}

    for ev in evaluators:
        # Warmup
        for h in hands_7[:100]:
            ev.evaluate(h)

        # 7-card benchmark
        t0 = time.perf_counter()
        ranks_7 = [ev.evaluate(h) for h in hands_7]
        elapsed_7 = (time.perf_counter() - t0) * 1000

        # 5-card benchmark
        t0 = time.perf_counter()
        ranks_5 = [ev.evaluate(h) for h in hands_5]
        elapsed_5 = (time.perf_counter() - t0) * 1000

        per_eval_7 = elapsed_7 / num_hands
        per_eval_5 = elapsed_5 / num_hands
        throughput_7 = num_hands / (elapsed_7 / 1000)
        throughput_5 = num_hands / (elapsed_5 / 1000)

        results[ev.name] = {
            "7card_total_ms": round(elapsed_7, 2),
            "7card_per_eval_us": round(per_eval_7 * 1000, 2),
            "7card_throughput": int(throughput_7),
            "5card_total_ms": round(elapsed_5, 2),
            "5card_per_eval_us": round(per_eval_5 * 1000, 2),
            "5card_throughput": int(throughput_5),
        }

        print(f"  {ev.name:12s} │ 7-card: {elapsed_7:>8.1f}ms ({per_eval_7*1000:>6.1f}µs/eval, {throughput_7:>10,.0f}/s)")
        print(f"  {'':12s} │ 5-card: {elapsed_5:>8.1f}ms ({per_eval_5*1000:>6.1f}µs/eval, {throughput_5:>10,.0f}/s)")

    return results


def bench_correctness(evaluators: list[HandEvaluator], num_hands: int = 5000) -> bool:
    """Cross-validate: all evaluators must produce the same hand ranking order."""
    print(f"\n{'='*70}")
    print(f"  CORRECTNESS: Cross-validation — {num_hands} hands")
    print(f"{'='*70}\n")

    if len(evaluators) < 2:
        print("  ⚠️ Need at least 2 evaluators for cross-validation")
        return True

    hands = _generate_random_hands(num_hands, 7)
    reference = evaluators[0]
    ref_name = reference.name

    # Get reference rankings
    ref_ranks = [reference.evaluate(h) for h in hands]

    all_passed = True
    for ev in evaluators[1:]:
        ev_ranks = [ev.evaluate(h) for h in hands]

        # Check that the ORDERING is the same (not absolute values)
        mismatches = 0
        for i in range(num_hands):
            for j in range(i + 1, min(i + 10, num_hands)):
                ref_cmp = (ref_ranks[i] > ref_ranks[j]) - (ref_ranks[i] < ref_ranks[j])
                ev_cmp = (ev_ranks[i] > ev_ranks[j]) - (ev_ranks[i] < ev_ranks[j])
                if ref_cmp != ev_cmp:
                    mismatches += 1

        total_pairs = sum(min(10, num_hands - i - 1) for i in range(num_hands))
        match_pct = (1 - mismatches / max(total_pairs, 1)) * 100

        status = "✅" if match_pct > 99.0 else "❌"
        print(f"  {status} {ref_name} vs {ev.name}: {match_pct:.2f}% pair agreement ({mismatches} mismatches)")

        if match_pct < 99.0:
            all_passed = False

    return all_passed


def bench_specific_hands(evaluators: list[HandEvaluator]) -> None:
    """Evaluate specific known hands and compare results."""
    print(f"\n{'='*70}")
    print("  SPECIFIC HANDS — Known hand rankings")
    print(f"{'='*70}\n")

    # Define test hands: (name, cards_as_str)
    test_hands = [
        ("Royal Flush", ["Ah", "Kh", "Qh", "Jh", "Th"]),
        ("Straight Flush", ["9s", "8s", "7s", "6s", "5s"]),
        ("Four of a Kind", ["As", "Ah", "Ad", "Ac", "Ks"]),
        ("Full House", ["Ks", "Kh", "Kd", "Qs", "Qh"]),
        ("Flush", ["Ah", "Jh", "9h", "7h", "3h"]),
        ("Straight", ["9c", "8d", "7h", "6s", "5c"]),
        ("Three of Kind", ["Js", "Jh", "Jd", "Ac", "Kc"]),
        ("Two Pair", ["As", "Ah", "Ks", "Kh", "Qc"]),
        ("One Pair", ["As", "Ah", "Kc", "Qd", "Js"]),
        ("High Card", ["As", "Kd", "Qc", "Jh", "9s"]),
    ]

    # Convert to int cards
    _r_map = {"2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "7": 5, "8": 6, "9": 7,
              "T": 8, "J": 9, "Q": 10, "K": 11, "A": 12}
    _s_map = {"c": 0, "d": 1, "h": 2, "s": 3}

    def _str_to_int(s: str) -> int:
        return _r_map[s[0]] * 4 + _s_map[s[1]]

    int_hands = [(name, [_str_to_int(c) for c in cards]) for name, cards in test_hands]

    # Header
    ev_names = [ev.name for ev in evaluators]
    header = f"  {'Hand':<20s} │ " + " │ ".join(f"{n:>12s}" for n in ev_names)
    print(header)
    print("  " + "─" * len(header))

    for name, cards in int_hands:
        ranks = [str(ev.evaluate(cards)) for ev in evaluators]
        row = f"  {name:<20s} │ " + " │ ".join(f"{r:>12s}" for r in ranks)
        print(row)

    # Verify ordering: each hand should rank strictly higher than the next
    print()
    for ev in evaluators:
        prev_rank = None
        order_ok = True
        for name, cards in int_hands:
            rank = ev.evaluate(cards)
            if prev_rank is not None and rank >= prev_rank:
                order_ok = False
                print(f"  ❌ {ev.name}: {name} ({rank}) should be < previous ({prev_rank})")
            prev_rank = rank
        if order_ok:
            print(f"  ✅ {ev.name}: All hands correctly ordered (descending)")


if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("  POKER HELPER — Evaluator Benchmark Suite")
    print("█" * 70)

    evaluators = _get_available_evaluators()
    print(f"\n  Available evaluators: {', '.join(ev.name for ev in evaluators)}")

    bench_specific_hands(evaluators)
    bench_evaluate(evaluators, num_hands=10000)
    bench_correctness(evaluators, num_hands=2000)

    print(f"\n{'='*70}")
    print("  DONE")
    print(f"{'='*70}\n")
