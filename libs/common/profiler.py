"""
Profiler — Pipeline stage timing and latency budget tracking.

Provides decorators and utilities for measuring per-stage performance.
"""

from __future__ import annotations

import functools
import logging
import statistics
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class PipelineProfiler:
    """
    Collects and reports per-stage pipeline timing metrics.

    Usage:
        profiler = PipelineProfiler()

        with profiler.measure("vision"):
            detections = detector.detect(frame)

        with profiler.measure("ocr"):
            ocr_results = ocr.extract(frame, detections)

        print(profiler.report())
    """

    def __init__(self) -> None:
        self._timings: dict[str, list[float]] = defaultdict(list)
        self._frame_count: int = 0

    class _MeasureContext:
        """Context manager for measuring a stage."""

        def __init__(self, profiler: PipelineProfiler, stage: str) -> None:
            self._profiler = profiler
            self._stage = stage
            self._start: float = 0.0

        def __enter__(self) -> _MeasureContext:
            self._start = time.perf_counter()
            return self

        def __exit__(self, *exc) -> None:
            elapsed_ms = (time.perf_counter() - self._start) * 1000
            self._profiler._timings[self._stage].append(elapsed_ms)

        @property
        def elapsed_ms(self) -> float:
            return (time.perf_counter() - self._start) * 1000

    def measure(self, stage: str) -> _MeasureContext:
        """Context manager to measure a pipeline stage."""
        return self._MeasureContext(self, stage)

    def record(self, stage: str, latency_ms: float) -> None:
        """Manually record a stage timing."""
        self._timings[stage].append(latency_ms)

    def end_frame(self) -> None:
        """Mark end of a frame (for frame counting)."""
        self._frame_count += 1

    def get_stage_stats(self, stage: str) -> dict[str, float]:
        """Get statistics for a specific stage."""
        timings = self._timings.get(stage, [])
        if not timings:
            return {"count": 0, "avg_ms": 0, "median_ms": 0, "p95_ms": 0, "p99_ms": 0}

        sorted_t = sorted(timings)
        n = len(sorted_t)

        return {
            "count": n,
            "avg_ms": round(statistics.mean(sorted_t), 2),
            "median_ms": round(statistics.median(sorted_t), 2),
            "min_ms": round(sorted_t[0], 2),
            "max_ms": round(sorted_t[-1], 2),
            "p95_ms": round(sorted_t[int(n * 0.95)] if n >= 20 else sorted_t[-1], 2),
            "p99_ms": round(sorted_t[int(n * 0.99)] if n >= 100 else sorted_t[-1], 2),
        }

    def report(self) -> dict[str, Any]:
        """Generate a full profiling report."""
        stages = {}
        total_avg = 0.0

        for stage in sorted(self._timings.keys()):
            stats = self.get_stage_stats(stage)
            stages[stage] = stats
            total_avg += stats["avg_ms"]

        return {
            "frames_profiled": self._frame_count,
            "total_avg_ms": round(total_avg, 2),
            "stages": stages,
        }

    def report_text(self) -> str:
        """Generate a human-readable report string."""
        r = self.report()
        lines = [
            "=" * 60,
            "  PIPELINE PROFILER REPORT",
            f"  Frames: {r['frames_profiled']}",
            f"  Total avg: {r['total_avg_ms']:.2f} ms",
            "=" * 60,
        ]

        for stage, stats in r["stages"].items():
            lines.append(f"\n  {stage}:")
            lines.append(f"    Samples: {stats['count']}")
            lines.append(f"    Avg:     {stats['avg_ms']:>8.2f} ms")
            lines.append(f"    Median:  {stats['median_ms']:>8.2f} ms")
            lines.append(f"    P95:     {stats['p95_ms']:>8.2f} ms")
            lines.append(f"    Min:     {stats['min_ms']:>8.2f} ms")
            lines.append(f"    Max:     {stats['max_ms']:>8.2f} ms")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all collected data."""
        self._timings.clear()
        self._frame_count = 0


# ─── Latency Budget ──────────────────────────────────────────────────────────


class LatencyBudget:
    """
    Checks if pipeline meets latency targets.

    Usage:
        budget = LatencyBudget(total_ms=200)
        budget.set_stage_budget("vision", 80)
        budget.set_stage_budget("ocr", 50)

        violations = budget.check(profiler)
    """

    def __init__(self, total_ms: float = 200.0) -> None:
        self.total_ms = total_ms
        self._stage_budgets: dict[str, float] = {}

    def set_stage_budget(self, stage: str, max_ms: float) -> None:
        """Set maximum latency for a stage."""
        self._stage_budgets[stage] = max_ms

    def check(self, profiler: PipelineProfiler) -> list[str]:
        """
        Check profiler against budgets.

        Returns:
            List of violation messages (empty = all good).
        """
        violations: list[str] = []
        report = profiler.report()

        # Check total
        if report["total_avg_ms"] > self.total_ms:
            violations.append(
                f"Total avg {report['total_avg_ms']:.1f}ms exceeds budget {self.total_ms:.1f}ms"
            )

        # Check per-stage
        for stage, max_ms in self._stage_budgets.items():
            stats = report["stages"].get(stage, {})
            avg = stats.get("avg_ms", 0)
            if avg > max_ms:
                violations.append(
                    f"Stage '{stage}' avg {avg:.1f}ms exceeds budget {max_ms:.1f}ms"
                )

        return violations


# ─── Decorator ───────────────────────────────────────────────────────────────


def profile_stage(
    stage_name: str,
    profiler: PipelineProfiler | None = None,
) -> Callable:
    """
    Decorator to auto-profile a function as a pipeline stage.

    Args:
        stage_name: Name of the pipeline stage.
        profiler: PipelineProfiler instance (if None, timing is logged).

    Usage:
        @profile_stage("vision")
        def detect(frame):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if profiler is not None:
                profiler.record(stage_name, elapsed_ms)
            else:
                logger.debug("Stage '%s': %.2f ms", stage_name, elapsed_ms)

            return result

        return wrapper

    return decorator
