## 2024-05-18 - Monte Carlo Deck Sampling Optimization
**Learning:** `random.shuffle` over the entire array inside a hot loop is a common but extremely inefficient pattern when only a small portion of the array is needed. In `EquitySolver`, shuffling a 45-card deck for each simulation took significantly longer than just sampling the needed cards. `random.sample` is much faster for drawing hands since it avoids shuffling the entire deck array.
**Action:** When picking a small subset of elements randomly in a performance-critical loop, use `random.sample` instead of shuffling the whole array and taking a slice.

## 2024-05-18 - Fast Monte Carlo Validation Set
**Learning:** In performance-critical Monte Carlo inner loops (such as `compute_range_vs_range_equity`), constructing a single O(1) `forbidden` set outside the loop for card validation is up to ~4x faster than performing repeated list or tuple membership checks inside the hot loop.
**Action:** When validating hands against sampled boards in Monte Carlo loops, use a combined set (e.g. `forbidden = {v_hand[0], v_hand[1], *sampled_board}`) instead of inline boolean logic with `in`.
