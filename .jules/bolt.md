## 2024-05-18 - Monte Carlo Deck Sampling Optimization
**Learning:** `random.shuffle` over the entire array inside a hot loop is a common but extremely inefficient pattern when only a small portion of the array is needed. In `EquitySolver`, shuffling a 45-card deck for each simulation took significantly longer than just sampling the needed cards. `random.sample` is much faster for drawing hands since it avoids shuffling the entire deck array.
**Action:** When picking a small subset of elements randomly in a performance-critical loop, use `random.sample` instead of shuffling the whole array and taking a slice.
## 2024-05-18 - Hand Evaluator Memory Allocations in Hot Paths
**Learning:** `_evaluate_five_int` was utilizing `dict` to count elements and calling `sorted()` to identify duplicate frequencies. Since this function is called inside the innermost loops of Monte Carlo simulations millions of times, avoiding object allocations entirely and just relying on comparisons over pre-sorted array elements (`sr0 == sr1`) gave a massive ~2.7x speedup for pure evaluation and significantly dropped full simulation latencies.
**Action:** In Python performance-critical hot paths, try to remove data structures like dicts, sets, and Counter in favor of simple boolean logic and manual loop unrolling, particularly for fixed-size inputs.

## 2024-05-18 - Array Reassignment in Inner Loop
**Learning:** Reassigning a variable in Python to a new type (e.g. `forbidden = set(...)`) breaks existing code below that expects it to be a pre-allocated list (`[False] * 52`), resulting in `TypeError("'set' object does not support item assignment")`. Furthermore, doing `forbidden = set(...)` creates memory allocations in the hot loop, which defeats the purpose of the O(1) array.
**Action:** For pre-allocated O(1) arrays inside hot loops, always mutate the existing structure in place (`forbidden[idx] = True / False`) rather than replacing it with a new list or set.
