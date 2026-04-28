## 2024-05-18 - Monte Carlo Hot Loop Validation Optimization
**Learning:** In performance-critical Monte Carlo inner loops (such as `compute_range_vs_range_equity`), constructing a single O(1) `forbidden` set outside the loop for card validation is up to ~4x faster than performing repeated list or tuple membership checks inside the hot loop.
**Action:** Always pre-calculate sets for exclusion checks outside hot loops where possible, but avoid allocating new collections if the operations to construct them are more expensive than the inner logic itself.

## 2024-05-18 - Monte Carlo Hand Simulation Loop Overheads
**Learning:** Contrary to intuition, in performance-critical Monte Carlo simulations drawing a small number of cards, using a `while` loop with `random.choice()` and linear membership checks is roughly 2x faster than using `random.sample()` over a dynamically filtered deck (via list comprehension). The cost of iterating the entire 50-card deck to construct the filtered list dominates the small number of collision retries in a `while` loop.
**Action:** When picking a small subset of items from a list randomly and excluding a few known values, rely on `while` + `random.choice()` with retry logic instead of `random.sample` on a filtered list, especially in tight loops.
