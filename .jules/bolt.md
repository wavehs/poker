## 2024-04-22 - [Hand Evaluator Dictionary Bottleneck]
**Learning:** Dictionary instantiation, multiple list comprehensions, and sorting `dict.values()` inside the inner loop (`_evaluate_five_int`) of a combinatorial solver creates a major bottleneck due to constant object allocation and garbage collection.
**Action:** Replace dynamic collection processing with direct scalar index comparisons and constant multipliers for performance-critical inner loops where inputs are bounded (e.g. exactly 5 cards).
