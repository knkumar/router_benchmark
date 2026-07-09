"""One-off script: filter the existing simulated study results down to
Paper 1's scope (LiteLLM Router, RouteLLM, Aurelio Semantic Router, vLLM
Semantic Router x RouterBench, BFCL v4, tau2-bench, WebArena; keep all 5
baselines since they anchor the Pareto frontier), recompute metrics, and
regenerate figures -- no re-simulation needed since the underlying
per-task rows are unchanged, just filtered.

    python experiments/build_paper1_subset.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from router_benchmark.metrics import (
    compute_pareto_frontier,
    compute_router_benchmark_metrics,
    compute_router_overall_metrics,
)
from router_benchmark.plots import generate_all_plots

ROUTERS = {
    "LiteLLM Router",
    "RouteLLM",
    "Aurelio Semantic Router",
    "vLLM Semantic Router",
    "Baseline: Always Cheapest",
    "Baseline: Always Strongest",
    "Baseline: Heuristic Difficulty",
    "Baseline: Oracle",
    "Baseline: Random",
}
BENCHMARKS = {"RouterBench", "BFCL v4", "tau2-bench", "WebArena"}

OUT_DIR = Path(__file__).resolve().parents[1] / "output" / "paper1_sim"


def main() -> None:
    results_df = pd.read_csv(Path(__file__).resolve().parents[1] / "output" / "results.csv")
    subset = results_df[
        results_df["router_name"].isin(ROUTERS) & results_df["benchmark_name"].isin(BENCHMARKS)
    ].copy()
    print(f"Subset: {len(subset):,} rows (from {len(results_df):,})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subset.to_csv(OUT_DIR / "results.csv", index=False)

    per_bm_df = compute_router_benchmark_metrics(subset)
    per_bm_df.to_csv(OUT_DIR / "metrics_per_benchmark.csv", index=False)

    overall_df = compute_router_overall_metrics(subset)
    overall_df.to_csv(OUT_DIR / "metrics_overall.csv", index=False)

    frontier_df = compute_pareto_frontier(overall_df)
    frontier_df.to_csv(OUT_DIR / "pareto_frontier.csv", index=False)

    print("\n=== Paper 1 simulated ranking (mean success rate) ===")
    cols = ["router_name", "mean_success_rate", "mean_cost_per_task_usd", "mean_latency_p50_ms"]
    print(overall_df[cols].sort_values("mean_success_rate", ascending=False).to_string(index=False))

    plots_dir = OUT_DIR / "plots"
    generate_all_plots(subset, per_bm_df, overall_df, plots_dir, label="simulated evaluation (Paper 1 subset)")
    print(f"\nOutput written under: {OUT_DIR}")


if __name__ == "__main__":
    main()
