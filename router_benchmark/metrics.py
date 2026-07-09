"""Deliverable (2): comparison metrics computed from harness results.

compute_router_benchmark_metrics(df) -> one row per (router, benchmark):
    success_rate, cost_per_task_usd, cost_per_success_usd,
    latency_p50_ms, latency_p95_ms, tool_call_accuracy, fallback_rate,
    route_stability, difficulty_band_{easy,medium,hard}_success_rate,
    robustness_std (std of success rate across difficulty bands; lower=more
    robust), mean_confidence

compute_router_overall_metrics(df) -> one row per router, aggregated across
all benchmarks, plus is_pareto_optimal on the cost-vs-quality frontier.

compute_pareto_frontier(df) -> the subset of (router) points that are not
dominated on (cost_per_task_usd, success_rate): no other router has both
lower-or-equal cost AND higher-or-equal success rate (with at least one
strictly better).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DIFFICULTY_BANDS = [(0.0, 1 / 3, "easy"), (1 / 3, 2 / 3, "medium"), (2 / 3, 1.0 + 1e-9, "hard")]


def _band_label(difficulty: float) -> str:
    for lo, hi, label in DIFFICULTY_BANDS:
        if lo <= difficulty < hi:
            return label
    return "hard"


def _route_stability(group: pd.DataFrame) -> float:
    """Fraction of tasks whose selected_candidate is identical across every
    trial for that task. 1.0 = perfectly deterministic router."""
    per_task_nunique = group.groupby("task_id")["selected_candidate"].nunique()
    return float((per_task_nunique == 1).mean())


def compute_router_benchmark_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["difficulty_band"] = df["difficulty"].apply(_band_label)

    rows = []
    for (router, benchmark), g in df.groupby(["router_name", "benchmark_name"]):
        tool_rows = g[g["tool_call_required"]]
        tool_acc = tool_rows["tool_call_correct"].mean() if len(tool_rows) else np.nan

        band_success = g.groupby("difficulty_band")["success"].mean()
        band_success = band_success.reindex(["easy", "medium", "hard"])

        n_success = g["success"].sum()
        cost_per_success = g["cost_usd"].sum() / n_success if n_success > 0 else np.nan

        rows.append(
            {
                "router_name": router,
                "benchmark_name": benchmark,
                "n_tasks": g["task_id"].nunique(),
                "n_trials": g["trial"].nunique(),
                "success_rate": g["success"].mean(),
                "cost_per_task_usd": g["cost_usd"].mean(),
                "cost_per_success_usd": cost_per_success,
                "latency_p50_ms": g["latency_ms"].median(),
                "latency_p95_ms": g["latency_ms"].quantile(0.95),
                "tool_call_accuracy": tool_acc,
                "fallback_rate": g["fallback_used"].mean(),
                "route_stability": _route_stability(g),
                "mean_confidence": g["confidence"].mean(),
                "easy_success_rate": band_success.get("easy", np.nan),
                "medium_success_rate": band_success.get("medium", np.nan),
                "hard_success_rate": band_success.get("hard", np.nan),
                "robustness_std": float(band_success.std()),
            }
        )
    out = pd.DataFrame(rows).sort_values(["benchmark_name", "success_rate"], ascending=[True, False])
    return out.reset_index(drop=True)


def compute_router_overall_metrics(df: pd.DataFrame) -> pd.DataFrame:
    per_bm = compute_router_benchmark_metrics(df)
    agg = (
        per_bm.groupby("router_name")
        .agg(
            mean_success_rate=("success_rate", "mean"),
            mean_cost_per_task_usd=("cost_per_task_usd", "mean"),
            mean_cost_per_success_usd=("cost_per_success_usd", "mean"),
            mean_latency_p50_ms=("latency_p50_ms", "mean"),
            mean_latency_p95_ms=("latency_p95_ms", "mean"),
            mean_tool_call_accuracy=("tool_call_accuracy", "mean"),
            mean_fallback_rate=("fallback_rate", "mean"),
            mean_route_stability=("route_stability", "mean"),
            mean_robustness_std=("robustness_std", "mean"),
            benchmarks_covered=("benchmark_name", "nunique"),
        )
        .reset_index()
    )
    agg["is_pareto_optimal"] = _pareto_flags(agg, cost_col="mean_cost_per_task_usd", quality_col="mean_success_rate")
    return agg.sort_values("mean_success_rate", ascending=False).reset_index(drop=True)


def _pareto_flags(df: pd.DataFrame, cost_col: str, quality_col: str) -> list[bool]:
    costs = df[cost_col].to_numpy()
    quals = df[quality_col].to_numpy()
    flags = []
    for i in range(len(df)):
        dominated = False
        for j in range(len(df)):
            if i == j:
                continue
            better_or_equal = costs[j] <= costs[i] and quals[j] >= quals[i]
            strictly_better = costs[j] < costs[i] or quals[j] > quals[i]
            if better_or_equal and strictly_better:
                dominated = True
                break
        flags.append(not dominated)
    return flags


def compute_pareto_frontier(overall_df: pd.DataFrame) -> pd.DataFrame:
    return overall_df[overall_df["is_pareto_optimal"]].sort_values("mean_cost_per_task_usd").reset_index(drop=True)
