from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from router_benchmark.harness import EvaluationHarness
from router_benchmark.interfaces import Benchmark, Candidate, RouteDecision, Router, Task, TaskDomain
from router_benchmark.metrics import assign_difficulty_bands, compute_router_benchmark_metrics
from router_benchmark.protocol.pareto import ParetoValidationError, pareto_membership_with_witness
from router_benchmark.routers import build_all_routers


class _FixedRouter(Router):
    def __init__(self, name: str, candidate: str) -> None:
        self.name = name
        self.candidate = candidate

    def route(self, task: Task, context: dict, rng) -> RouteDecision:
        return RouteDecision(self.candidate, confidence=1.0, fallback_used=False)


class _RecordingBenchmark(Benchmark):
    name = "recording"

    def __init__(self) -> None:
        self.score_calls = 0

    def generate_tasks(self, rng) -> list[Task]:
        return [
            Task(
                task_id="task-1",
                benchmark_name=self.name,
                domain=TaskDomain.QA_REASONING,
                difficulty=0.5,
                requires_tool_call=False,
                candidates=(
                    Candidate("cheap", "cheap", 0.1, 0.5, 1.0),
                    Candidate("strong", "strong", 1.0, 0.9, 2.0),
                ),
            )
        ]

    def score(self, task: Task, decision: RouteDecision, rng) -> dict:
        self.score_calls += 1
        return {
            "success": bool(rng.integers(0, 2)),
            "cost_usd": float(rng.random()),
            "latency_ms": float(rng.random()),
            "tool_call_correct": None,
        }


class _LiveRecordingBenchmark(_RecordingBenchmark):
    reusable_score = False


def test_harness_does_not_reuse_live_score_outcomes() -> None:
    benchmark = _LiveRecordingBenchmark()
    results = EvaluationHarness(seed=7, n_trials=3).evaluate(
        [_FixedRouter("one", "cheap"), _FixedRouter("two", "cheap")], [benchmark]
    )

    assert len(results) == 6
    assert benchmark.score_calls == 6


def test_harness_reuses_each_candidate_outcome_across_routers_and_trials() -> None:
    benchmark = _RecordingBenchmark()
    results = EvaluationHarness(seed=7, n_trials=3).evaluate(
        [_FixedRouter("one", "cheap"), _FixedRouter("two", "cheap")], [benchmark]
    )

    assert benchmark.score_calls == 4
    assert results.groupby("selected_candidate")[["success", "cost_usd", "latency_ms"]].nunique().eq(1).all().all()


def test_router_factory_exposes_backup_calibration_variants_and_mid_baseline() -> None:
    names = {router.name for router in build_all_routers()}
    assert len(names) == 18
    assert "RouteLLM (Out-of-the-box)" in names
    assert "RouteLLM (Calibrated)" in names
    assert "Baseline: Always Mid" in names


def test_difficulty_bands_are_task_count_balanced_and_singleton_stability_is_undefined() -> None:
    rows = []
    for task_index in range(10):
        rows.append(
            {
                "router_name": "router", "benchmark_name": "benchmark", "task_id": f"task-{task_index}",
                "difficulty": task_index / 10, "trial": 0, "selected_candidate": "cheap",
                "confidence": 0.5, "fallback_used": False, "success": task_index % 2 == 0,
                "tool_call_required": False, "tool_call_correct": None, "cost_usd": 1.0,
                "latency_ms": 1.0,
            }
        )
    frame = pd.DataFrame(rows)
    bands = assign_difficulty_bands(frame)
    counts = bands["difficulty_band"].value_counts()
    metric = compute_router_benchmark_metrics(frame).iloc[0]

    assert counts.max() - counts.min() <= 1
    assert math.isnan(metric["route_stability"])


def test_pareto_reports_a_witness_and_rejects_invalid_metrics() -> None:
    results = pareto_membership_with_witness(
        [
            {"router": "cheap", "cost": 0.1, "success": 0.6},
            {"router": "strong", "cost": 1.0, "success": 0.9},
            {"router": "dominated", "cost": 1.1, "success": 0.8},
        ],
        id_key="router",
        cost_key="cost",
        success_key="success",
    )
    by_id = {row["point_id"]: row for row in results}
    assert by_id["dominated"]["dominated_by"] == "strong"
    with pytest.raises(ParetoValidationError, match="out-of-bounds"):
        pareto_membership_with_witness(
            [{"router": "invalid", "cost": -0.1, "success": 0.5}],
            id_key="router",
            cost_key="cost",
            success_key="success",
        )
