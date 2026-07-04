import time
from services.solver_core.solver import EquitySolver
from libs.common.schemas import Card, Rank, Suit
solver = EquitySolver(enable_cache=False)
h_range = [(i, j) for i in range(52) for j in range(i+1, 52)][:50]
v_range = [(i, j) for i in range(52) for j in range(i+1, 52)][50:100]
board = []
t0 = time.time()
try:
    res = solver.compute_range_vs_range_equity(h_range, v_range, board, simulations=100)
    print("Success, took", time.time() - t0)
except Exception as e:
    import traceback
    traceback.print_exc()
