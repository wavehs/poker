import time
import random

class MockDetection:
    def __init__(self, id_val):
        self.id_val = id_val

def _match_detections_current(detections):
    # Simulate some matching logic
    matched = []
    for i, det in enumerate(detections):
        if random.random() > 0.5:
            matched.append((i, det))
    return matched

def _match_detections_new(detections):
    matched = []
    used_indices = set()
    for i, det in enumerate(detections):
        if random.random() > 0.5:
            matched.append((i, det))
            used_indices.add(i)
    unmatched = [det for i, det in enumerate(detections) if i not in used_indices]
    return matched, unmatched

def current_full_logic(detections):
    matched = _match_detections_current(detections)
    # matched tracks update (omitted)

    matched_dets = {id(det) for _, det in matched}
    unmatched = []
    for det in detections:
        if id(det) not in matched_dets:
            unmatched.append(det)
    return matched, unmatched

def optimized_full_logic(detections):
    matched, unmatched = _match_detections_new(detections)
    # matched tracks update (omitted)
    return matched, unmatched

def benchmark(num_detections, iterations=10000):
    detections = [MockDetection(i) for i in range(num_detections)]

    # Warmup
    current_full_logic(detections)
    optimized_full_logic(detections)

    t0 = time.perf_counter()
    for _ in range(iterations):
        current_full_logic(detections)
    t1 = time.perf_counter()
    current_time = (t1 - t0) * 1000

    t0 = time.perf_counter()
    for _ in range(iterations):
        optimized_full_logic(detections)
    t1 = time.perf_counter()
    optimized_time = (t1 - t0) * 1000

    print(f"Num Detections: {num_detections}")
    print(f"  Current:   {current_time:.4f} ms")
    print(f"  Optimized: {optimized_time:.4f} ms")
    if optimized_time > 0:
        print(f"  Speedup:   {current_time / optimized_time:.2f}x")
    print()

if __name__ == "__main__":
    benchmark(10)
    benchmark(50)
    benchmark(100)
    benchmark(1000)
