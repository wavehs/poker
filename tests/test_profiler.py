"""Tests for Pipeline Profiler."""

import time

import pytest

from libs.common.profiler import LatencyBudget, PipelineProfiler, profile_stage


class TestPipelineProfiler:
    def test_empty_profiler(self):
        p = PipelineProfiler()
        r = p.report()
        assert r["frames_profiled"] == 0
        assert r["total_avg_ms"] == 0

    def test_measure_context(self):
        p = PipelineProfiler()

        with p.measure("test_stage"):
            time.sleep(0.01)

        stats = p.get_stage_stats("test_stage")
        assert stats["count"] == 1
        assert stats["avg_ms"] >= 5  # At least 5ms (sleep was 10ms)

    def test_manual_record(self):
        p = PipelineProfiler()
        p.record("vision", 15.5)
        p.record("vision", 20.3)

        stats = p.get_stage_stats("vision")
        assert stats["count"] == 2
        assert abs(stats["avg_ms"] - 17.9) < 0.1

    def test_multiple_stages(self):
        p = PipelineProfiler()
        p.record("vision", 10.0)
        p.record("ocr", 20.0)
        p.record("state", 5.0)

        r = p.report()
        assert len(r["stages"]) == 3
        assert r["total_avg_ms"] == 35.0

    def test_frame_count(self):
        p = PipelineProfiler()
        p.end_frame()
        p.end_frame()
        assert p.report()["frames_profiled"] == 2

    def test_report_text(self):
        p = PipelineProfiler()
        p.record("vision", 15.0)
        p.record("ocr", 8.0)

        text = p.report_text()
        assert "vision" in text
        assert "ocr" in text
        assert "PIPELINE PROFILER" in text

    def test_reset(self):
        p = PipelineProfiler()
        p.record("vision", 15.0)
        p.end_frame()
        p.reset()

        r = p.report()
        assert r["frames_profiled"] == 0
        assert len(r["stages"]) == 0

    def test_unknown_stage_stats(self):
        p = PipelineProfiler()
        stats = p.get_stage_stats("nonexistent")
        assert stats["count"] == 0


class TestLatencyBudget:
    def test_within_budget(self):
        p = PipelineProfiler()
        p.record("vision", 30.0)
        p.record("ocr", 20.0)

        budget = LatencyBudget(total_ms=100)
        budget.set_stage_budget("vision", 50)

        violations = budget.check(p)
        assert len(violations) == 0

    def test_total_exceeded(self):
        p = PipelineProfiler()
        p.record("vision", 150.0)
        p.record("ocr", 100.0)

        budget = LatencyBudget(total_ms=200)
        violations = budget.check(p)
        assert len(violations) == 1
        assert "Total" in violations[0]

    def test_stage_exceeded(self):
        p = PipelineProfiler()
        p.record("vision", 100.0)

        budget = LatencyBudget(total_ms=500)
        budget.set_stage_budget("vision", 50)

        violations = budget.check(p)
        assert len(violations) == 1
        assert "vision" in violations[0]


class TestProfileDecorator:
    def test_decorator_records_timing(self):
        profiler = PipelineProfiler()

        @profile_stage("test_func", profiler=profiler)
        def slow_function():
            time.sleep(0.01)
            return 42

        result = slow_function()
        assert result == 42

        stats = profiler.get_stage_stats("test_func")
        assert stats["count"] == 1
        assert stats["avg_ms"] >= 5

    def test_decorator_without_profiler(self):
        """Should work even without a profiler (just logs)."""

        @profile_stage("orphan")
        def func():
            return "ok"

        assert func() == "ok"
