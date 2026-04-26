## 2024-05-18 - Monte Carlo Deck Sampling Optimization
**Learning:** `random.shuffle` over the entire array inside a hot loop is a common but extremely inefficient pattern when only a small portion of the array is needed. In `EquitySolver`, shuffling a 45-card deck for each simulation took significantly longer than just sampling the needed cards. `random.sample` is much faster for drawing hands since it avoids shuffling the entire deck array.
**Action:** When picking a small subset of elements randomly in a performance-critical loop, use `random.sample` instead of shuffling the whole array and taking a slice.

## 2024-06-25 - Boolean Logic for Poker Hand Evaluation
**Learning:** `_evaluate_five_int` in `services/solver_core/evaluator.py` is called millions of times. The old version used `dict` and `sorted()` to count card occurrences (e.g. for pairs, three of a kind, etc). Since the 5 ranks are already sorted, we can avoid expensive dictionary and set allocations entirely and just check adjacent elements (e.g., `v0 == v3` or `v1 == v4` means four-of-a-kind). This pattern significantly reduces garbage collection and increases speed.
**Action:** In inner simulation loops, replace dictionary frequency maps with direct boolean logic if the input array is small and already sorted.
