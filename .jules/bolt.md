## 2024-05-18 - Monte Carlo Deck Sampling Optimization
**Learning:** `random.shuffle` over the entire array inside a hot loop is a common but extremely inefficient pattern when only a small portion of the array is needed. In `EquitySolver`, shuffling a 45-card deck for each simulation took significantly longer than just sampling the needed cards. `random.sample` is much faster for drawing hands since it avoids shuffling the entire deck array.
**Action:** When picking a small subset of elements randomly in a performance-critical loop, use `random.sample` instead of shuffling the whole array and taking a slice.

## 2024-05-04 - [Optimize pure Python Hand Evaluator]
**Learning:** In Python performance-critical hot paths like Monte Carlo inner loops (e.g., `_evaluate_five_int`), removing data structures like `dict`, `set`, and `Counter` in favor of simple boolean logic and manual loop unrolling over pre-sorted array elements yields significant speedups (measured at ~3x faster purely allocation-free logic).
**Action:** Use manual sorting networks (e.g. 10 if/swaps for 5 elements) and simple adjacent element checks for performance-critical integer pipelines instead of generic Python collections and high-level builtin functions.
