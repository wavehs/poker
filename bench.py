import sys
import time
from unittest.mock import MagicMock

sys.modules['pydantic'] = MagicMock()
sys.modules['pydantic.BaseModel'] = MagicMock

from services.solver_core.solver import EquitySolver

solver = EquitySolver(default_simulations=1000)
c5 = MagicMock()
c5.is_known = True
c5.rank.value = "2"
c5.suit.value = "h"
c6 = MagicMock()
c6.is_known = True
c6.rank.value = "3"
c6.suit.value = "c"
c7 = MagicMock()
c7.is_known = True
c7.rank.value = "4"
c7.suit.value = "d"

hero_range = [(i, j) for i in range(20) for j in range(i+1, 20)]
villain_range = [(i, j) for i in range(20, 40) for j in range(i+1, 40)]

t0 = time.time()
solver.compute_range_vs_range_equity(hero_range, villain_range, [c5, c6, c7])
print(f"Time: {time.time() - t0}")
