"""Solver Core — equity calculation and Monte Carlo simulation."""

from services.solver_core.evaluator import (
    BuiltinEvaluator,
    Eval7Evaluator,
    HandEvaluator,
    TreysEvaluator,
    get_best_evaluator,
    get_evaluator_by_name,
)
from services.solver_core.solver import EquitySolver, SolverProfile

from services.solver_core.calculator import calculate_spr, get_spr_advice

__all__ = [
    "EquitySolver",
    "SolverProfile",
    "HandEvaluator",
    "calculate_spr",
    "get_spr_advice",
    "BuiltinEvaluator",
    "Eval7Evaluator",
    "TreysEvaluator",
    "get_best_evaluator",
    "get_evaluator_by_name",
]
