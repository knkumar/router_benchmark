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

from router_benchmark.protocol.pareto import pareto_membership_with_witness

DIFFICULTY_BAND_NAMES = ("easy", "medium", "hard")


def assign_difficulty_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Assign each benchmark's tasks to count-balanced difficulty terciles.

    Task IDs break tied difficulty values lexicographically, which keeps the
    assignment reproducible from the declared task inventory.
    """
    required = {"benchmark_name", "task_id", "difficulty"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cannot assign difficulty bands without {sorted(missing)}")
    tasks = df[["benchmark_name", "task_id", "difficulty"]].drop_duplicates()
    if tasks.duplicated(["benchmark_name", "task_id"]).any():
        raise ValueError("A task has inconsistent difficulty values")

    assignments: list[dict[str, str]] = []
    for benchmark, group in tasks.groupby("benchmark_name", sort=False):
        ordered = group.sort_values(["difficulty", "task_id"], kind="stable").reset_index(drop=True)
        n_tasks = len(ordered)
        for index, row in ordered.iterrows():
            assignments.append(
                {
                    "benchmark_name": benchmark,
                    "task_id": row["task_id"],
                    "difficulty_band": DIFFICULTY_BAND_NAMES[min(2, (index * 3) // n_tasks)],
                }
            )
    return pd.DataFrame(assignments)


def _route_stability(group: pd.DataFrame) -> float:
    """Fraction of tasks whose selected_candidate is identical across every
    trial for that task. Stability is undefined without two decisions per
    task."""
    task_groups = group.groupby("task_id")["selected_candidate"]
    if (task_groups.size() < 2).any():
        return np.nan
    per_task_nunique = task_groups.nunique()
    return float((per_task_nunique == 1).mean())


def compute_router_benchmark_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["difficulty_band"], errors="ignore").merge(
        assign_difficulty_bands(df),
        on=["benchmark_name", "task_id"],
        how="left",
        validate="many_to_one",
    )

    rows = []
    for (router, benchmark), g in df.groupby(["router_name", "benchmark_name"]):
        tool_rows = g[g["tool_call_required"]]
        tool_acc = tool_rows["tool_call_correct"].mean() if len(tool_rows) else np.nan

        band_success = g.groupby("difficulty_band")["success"].mean()
        band_success = band_success.reindex(["easy", "medium", "hard"])

        n_success = g["success"].sum()
        model_api_cost = g["cost_usd"]
        cost_per_success = model_api_cost.sum() / n_success if n_success > 0 else np.nan

        rows.append(
            {
                "router_name": router,
                "benchmark_name": benchmark,
                "n_tasks": g["task_id"].nunique(),
                "n_trials": g["trial"].nunique(),
                "success_rate": g["success"].mean(),
                "cost_per_task_usd": model_api_cost.mean(),
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
    points = [
        {"index": str(index), "cost": row[cost_col], "success": row[quality_col]}
        for index, (_, row) in enumerate(df.iterrows())
    ]
    return [
        result["is_pareto_nondominated"]
        for result in pareto_membership_with_witness(
            points,
            id_key="index",
            cost_key="cost",
            success_key="success",
        )
    ]


def compute_pareto_frontier(overall_df: pd.DataFrame) -> pd.DataFrame:
    return overall_df[overall_df["is_pareto_optimal"]].sort_values("mean_cost_per_task_usd").reset_index(drop=True)
