"""Tests for SPR calculator and advice."""

import math
from services.solver_core.calculator import calculate_spr, get_spr_advice


def test_calculate_spr():
    # Regular SPR calculation
    assert calculate_spr(100.0, 10.0) == 10.0
    assert calculate_spr(50.0, 100.0) == 0.5

    # Edge case: zero pot size
    assert math.isinf(calculate_spr(100.0, 0.0))

    # Edge case: negative pot size
    assert math.isinf(calculate_spr(100.0, -10.0))


def test_get_spr_advice():
    # SPR < 1
    assert "SPR < 1" in get_spr_advice(0.5)

    # 1 <= SPR < 4
    assert "SPR 1-4" in get_spr_advice(1.0)
    assert "SPR 1-4" in get_spr_advice(3.9)

    # 4 <= SPR < 10
    assert "SPR 4-10" in get_spr_advice(4.0)
    assert "SPR 4-10" in get_spr_advice(9.9)

    # 10 <= SPR < 20
    assert "SPR 10-20" in get_spr_advice(10.0)
    assert "SPR 10-20" in get_spr_advice(19.9)

    # SPR >= 20
    assert "SPR 20+" in get_spr_advice(20.0)
    assert "SPR 20+" in get_spr_advice(100.0)
