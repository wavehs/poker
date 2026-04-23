"""
Benchmark Suite — Solver (equity computation) performance benchmarks.

Tests full compute_equity() across different scenarios:
  - Preflop, flop, turn, river
  - Fixed vs adaptive Monte Carlo
  - Different evaluator backends
  - Different opponent counts

Usage:
    python evals/bench_solver.py
"""

from __future__ import annotations

import sys
import time

# Add project root to path
sys.path.insert(0, ".")

from libs.common.schemas import Card, Rank, Suit
from services.solver_core.evaluator import BuiltinEvaluator, HandEvaluator
from services.solver_core.solver import EquitySolver

# ─── Test Data ───────────────────────────────────────────────────────────────


def _card(rank_str: str, suit_str: str) -> Card:
    _ranks = {"2": Rank.TWO, "3": Rank.THREE, "4": Rank.FOUR, "5": Rank.FIVE,
              "6": Rank.SIX, "7": Rank.SEVEN, "8": Rank.EIGHT, "9": Rank.NINE,
              "T": Rank.TEN, "J": Rank.JACK, "Q": Rank.QUEEN, "K": Rank.KING, "A": Rank.ACE}
    _suits = {"c": Suit.CLUBS, "d": Suit.DIAMONDS, "h": Suit.HEARTS, "s": Suit.SPADES}
    return Card(rank=_ranks[rank_str], suit=_suits[suit_str], confidence=1.0, source="bench")


SCENARIOS = {
    "preflop_AA": {
        "hero": [_card("A", "h"), _card("A", "s")],
        "board": [],
        "expected_equity_range": (0.78, 0.92),
    },
    "preflop_72o": {
        "hero": [_card("7", "h"), _card("2", "s")],
        "board": [],
        "expected_equity_range": (0.25, 0.45),
    },
    "flop_TPTK": {
        "hero": [_card("A", "h"), _card("K", "s")],
        "board": [_card("A", "d"), _card("K", "d"), _card("7", "c")],
        "expected_equity_range": (0.75, 0.95),
    },
    "turn_flush_draw": {
        "hero": [_card("A", "h"), _card("J", "h")],
        "board": [_card("K", "h"), _card("9", "h"), _card("5", "c"), _card("2", "d")],
        "expected_equity_range": (0.30, 0.60),
    },
    "river_set": {
        "hero": [_card("Q", "h"), _card("Q", "s")],
        "board": [_card("Q", "d"), _card("8", "c"), _card("5", "h"), _card("3", "s"), _card("2", "d")],
        "expected_equity_range": (0.85, 0.99),
    },
}


# ─── Benchmarks ──────────────────────────────────────────────────────────────


def _get_evaluators() -> list[HandEvaluator]:
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


def bench_equity_by_evaluator(simulations: int = 2000) -> None:
    """Benchmark compute_equity for each evaluator across all scenarios."""
    print(f"\n{'='*70}")
    print(f"  BENCHMARK: compute_equity() — {simulations} sims, fixed MC")
    print(f"{'='*70}")

    evaluators = _get_evaluators()

    for ev in evaluators:
        print(f"\n  ── Evaluator: {ev.name} ──")

        solver = EquitySolver(
            default_simulations=simulations,
            evaluator=ev,
            adaptive=False,
        )

        for name, scenario in SCENARIOS.items():
            t0 = time.perf_counter()
            equity = solver.compute_equity(
                scenario["hero"], scenario["board"],
                num_opponents=1, simulations=simulations,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            profile = solver.last_profile

            lo, hi = scenario["expected_equity_range"]
            ok = "✅" if lo <= equity <= hi else "❌"

            print(f"    {ok} {name:20s} │ eq={equity:.3f} [{lo:.2f}-{hi:.2f}] │ "
                  f"{elapsed:>7.1f}ms │ evals={profile.evaluate_calls if profile else '?'}")


def bench_adaptive_vs_fixed() -> None:
    """Compare adaptive vs fixed Monte Carlo."""
    print(f"\n{'='*70}")
    print("  BENCHMARK: Adaptive vs Fixed Monte Carlo")
    print(f"{'='*70}")

    evaluators = _get_evaluators()
    best_ev = evaluators[-1]  # Use the fastest available

    for sims in [1000, 2000, 5000]:
        print(f"\n  ── Max simulations: {sims}, evaluator: {best_ev.name} ──")

        solver_fixed = EquitySolver(
            default_simulations=sims, evaluator=best_ev, adaptive=False,
        )
        solver_adaptive = EquitySolver(
            default_simulations=sims, evaluator=best_ev, adaptive=True,
            min_simulations=200, step_size=200, confidence_threshold=0.03,
        )

        for name, scenario in SCENARIOS.items():
            # Fixed
            t0 = time.perf_counter()
            eq_fixed = solver_fixed.compute_equity(
                scenario["hero"], scenario["board"], num_opponents=1,
            )
            ms_fixed = (time.perf_counter() - t0) * 1000
            sims_fixed = solver_fixed.last_profile.simulations_run if solver_fixed.last_profile else sims

            # Adaptive
            t0 = time.perf_counter()
            eq_adaptive = solver_adaptive.compute_equity(
                scenario["hero"], scenario["board"], num_opponents=1,
            )
            ms_adaptive = (time.perf_counter() - t0) * 1000
            sims_adaptive = solver_adaptive.last_profile.simulations_run if solver_adaptive.last_profile else sims
            early = solver_adaptive.last_profile.early_stopped if solver_adaptive.last_profile else False

            speedup = ms_fixed / ms_adaptive if ms_adaptive > 0 else 0

            print(f"    {name:20s} │ fixed: {ms_fixed:>6.1f}ms ({sims_fixed:>5d} sims, eq={eq_fixed:.3f}) │ "
                  f"adaptive: {ms_adaptive:>6.1f}ms ({sims_adaptive:>5d} sims, eq={eq_adaptive:.3f}) │ "
                  f"{'early' if early else 'full ':5s} │ {speedup:.1f}×")


def bench_multi_opponent() -> None:
    """Benchmark with different opponent counts."""
    print(f"\n{'='*70}")
    print("  BENCHMARK: Multi-opponent (1-5 opponents)")
    print(f"{'='*70}")

    evaluators = _get_evaluators()
    best_ev = evaluators[-1]

    solver = EquitySolver(
        default_simulations=1000, evaluator=best_ev, adaptive=True,
    )

    hero = SCENARIOS["flop_TPTK"]["hero"]
    board = SCENARIOS["flop_TPTK"]["board"]

    for n_opp in [1, 2, 3, 5]:
        t0 = time.perf_counter()
        equity = solver.compute_equity(hero, board, num_opponents=n_opp)
        elapsed = (time.perf_counter() - t0) * 1000
        profile = solver.last_profile

        print(f"    {n_opp} opponent(s) │ eq={equity:.3f} │ {elapsed:>7.1f}ms │ "
              f"sims={profile.simulations_run if profile else '?'} │ "
              f"evals={profile.evaluate_calls if profile else '?'}")


def bench_latency_target() -> None:
    """Test: can we hit <500ms for all scenarios with best evaluator?"""
    print(f"\n{'='*70}")
    print("  LATENCY TARGET: <500ms per recommendation")
    print(f"{'='*70}")

    evaluators = _get_evaluators()
    best_ev = evaluators[-1]

    solver = EquitySolver(
        default_simulations=5000, evaluator=best_ev, adaptive=True,
        min_simulations=200, step_size=200, confidence_threshold=0.03,
    )

    all_ok = True
    for name, scenario in SCENARIOS.items():
        t0 = time.perf_counter()
        equity = solver.compute_equity(
            scenario["hero"], scenario["board"], num_opponents=2,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        profile = solver.last_profile

        ok = "✅" if elapsed < 500 else "❌"
        if elapsed >= 500:
            all_ok = False

        print(f"    {ok} {name:20s} │ {elapsed:>7.1f}ms │ eq={equity:.3f} │ "
              f"sims={profile.simulations_run if profile else '?'} │ "
              f"early={profile.early_stopped if profile else '?'}")

    print()
    if all_ok:
        print("    ✅ ALL scenarios under 500ms target!")
    else:
        print("    ❌ Some scenarios exceed 500ms — need further optimization")


if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("  POKER HELPER — Solver Benchmark Suite")
    print("█" * 70)

    bench_equity_by_evaluator(simulations=2000)
    bench_adaptive_vs_fixed()
    bench_multi_opponent()
    bench_latency_target()

    print(f"\n{'='*70}")
    print("  DONE")
    print(f"{'='*70}\n")
