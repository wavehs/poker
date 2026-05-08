## 2024-05-18 - Monte Carlo Deck Sampling Optimization
**Learning:** `random.shuffle` over the entire array inside a hot loop is a common but extremely inefficient pattern when only a small portion of the array is needed. In `EquitySolver`, shuffling a 45-card deck for each simulation took significantly longer than just sampling the needed cards. `random.sample` is much faster for drawing hands since it avoids shuffling the entire deck array.
**Action:** When picking a small subset of elements randomly in a performance-critical loop, use `random.sample` instead of shuffling the whole array and taking a slice.
## 2026-05-08 - _evaluate_five_int Manual Sort and Check
**Learning:** In the poker hand evaluation hotspot `_evaluate_five_int`, manual loop unrolling for sorting and matching values provided a 3x speedup over using `dict` (for counting), `sorted()`, and array slice assignments.
**Action:** Replace collection allocations and generic functions with explicitly hardcoded comparisons and linear flow when sequence lengths are statically known to be very small (e.g. 5 cards).
