## 2024-05-18 - Monte Carlo Deck Sampling Optimization
**Learning:** `random.shuffle` over the entire array inside a hot loop is a common but extremely inefficient pattern when only a small portion of the array is needed. In `EquitySolver`, shuffling a 45-card deck for each simulation took significantly longer than just sampling the needed cards. `random.sample` is much faster for drawing hands since it avoids shuffling the entire deck array.
**Action:** When picking a small subset of elements randomly in a performance-critical loop, use `random.sample` instead of shuffling the whole array and taking a slice.

## 2024-05-18 - Dictionary Allocations in Hot Paths
**Learning:** In highly repetitive Monte Carlo simulations (like `_evaluate_five_int`), allocating a `dict` for counting instances and using `sorted()` significantly slows down execution. By taking advantage of an already sorted array of values, replacing dictionaries with boolean checks on adjacent elements (`r0 == r3` for four-of-a-kind, etc.) speeds up the evaluation logic by roughly 2.5x to 3x with pure python.
**Action:** When working in hot paths, avoid all dynamic object allocations like dictionaries or sets. Use direct indexing and boolean logic on pre-sorted arrays where possible.
