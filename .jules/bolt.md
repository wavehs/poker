## 2024-05-18 - Monte Carlo Deck Sampling Optimization
**Learning:** `random.shuffle` over the entire array inside a hot loop is a common but extremely inefficient pattern when only a small portion of the array is needed. In `EquitySolver`, shuffling a 45-card deck for each simulation took significantly longer than just sampling the needed cards. `random.sample` is much faster for drawing hands since it avoids shuffling the entire deck array.
**Action:** When picking a small subset of elements randomly in a performance-critical loop, use `random.sample` instead of shuffling the whole array and taking a slice.

## 2024-10-25 - Boolean Array Tracking for Monte Carlo
**Learning:** In highly optimized Monte Carlo simulation hot loops, tracking drawn cards using list or tuple `in` operations creates significant linear lookup overhead, even for small arrays. Allocating a simple fixed-size boolean array (`forbidden = [False] * 52`) outside the loop and flipping indices dramatically outperforms sequence scanning because it reduces object iteration and relies entirely on fast O(1) index lookups.
**Action:** When drawing or filtering multiple items repeatedly from a fixed set (like a 52-card deck) in performance critical paths, use a pre-allocated boolean tracking array to record state rather than list checking or dynamic `set` construction.
