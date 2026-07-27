from __future__ import annotations

import math

import pandas as pd

from router_benchmark.metrics import assign_difficulty_bands, compute_router_benchmark_metrics


def _rows(trials: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "router_name": "router", "benchmark_name": "benchmark", "task_id": "task",
                "difficulty": 0.5, "trial": trial, "selected_candidate": "cheap-small",
                "confidence": 0.5, "fallback_used": False, "success": True,
                "tool_call_required": False, "tool_call_correct": False, "cost_usd": 2.0,
                "latency_ms": 999999.0,
            }
            for trial in trials
        ]
    )


def test_singleton_route_stability_is_undefined() -> None:
    metric = compute_router_benchmark_metrics(_rows([0])).iloc[0]
    assert math.isnan(metric["route_stability"])


def test_cost_is_measured_model_api_cost_not_latency_pricing() -> None:
    metric = compute_router_benchmark_metrics(_rows([0, 1])).iloc[0]
    assert metric["cost_per_task_usd"] == 2.0
    assert metric["cost_per_success_usd"] == 2.0
    assert metric["route_stability"] == 1.0


def test_difficulty_bands_are_task_count_balanced_and_reconcile_success() -> None:
    rows = []
    for task_index in range(10):
        for trial in range(2):
            rows.append({
                "router_name": "router", "benchmark_name": "benchmark", "task_id": f"task-{task_index}",
                "difficulty": 0.5 if task_index in {3, 4, 5} else task_index / 10,
                "trial": trial, "selected_candidate": "cheap-small", "confidence": 0.5,
                "fallback_used": False, "success": task_index % 2 == 0, "tool_call_required": False,
                "tool_call_correct": False, "cost_usd": 1.0, "latency_ms": 1.0,
            })
    frame = pd.DataFrame(rows)
    bands = assign_difficulty_bands(frame)
    counts = bands["difficulty_band"].value_counts()
    assert counts.max() - counts.min() <= 1
    metrics = compute_router_benchmark_metrics(frame).iloc[0]
    weighted_band_rate = sum(counts[band] * metrics[f"{band}_success_rate"] for band in counts.index) / counts.sum()
    assert weighted_band_rate == metrics["success_rate"]

