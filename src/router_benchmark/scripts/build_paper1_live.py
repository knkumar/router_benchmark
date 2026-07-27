"""One-off script: assemble Paper 1's live-evaluation scope (LiteLLM
Router, RouteLLM, Aurelio Semantic Router, vLLM Semantic Router x
RouterBench, BFCL v4, tau2-bench, WebArena) from the existing live phase
outputs -- combined_rb_bfcl/ (RouterBench+BFCL, all 6 routers, filter to
4), phase3/ (tau2-bench, already exactly these 4 routers), phase7c/
(WebArena, all 6 routers, filter to 4) -- then recompute metrics and
regenerate figures. No new experiments: every row here is real data
already collected for the main 6x6 paper.

    python -m router_benchmark.build_paper1_live
"""

from __future__ import annotations

from pathlib import Path

from router_benchmark.scripts._paths import repository_root

REPO_ROOT = repository_root()

import pandas as pd

from router_benchmark.metrics import (
    compute_pareto_frontier,
    compute_router_benchmark_metrics,
    compute_router_overall_metrics,
)
from router_benchmark.plots import generate_all_plots

ROUTERS = {
    "LiteLLM Router (live)",
    "RouteLLM (live)",
    "Aurelio Semantic Router (live)",
    "vLLM Semantic Router (live)",
}

LIVE_ROOT = REPO_ROOT / "output" / "live"
OUT_DIR = LIVE_ROOT / "paper1_live"


def main() -> None:
    frames = []
    for phase in ["combined_rb_bfcl", "phase3", "phase7c"]:
        df = pd.read_csv(LIVE_ROOT / phase / "results.csv")
        df = df[df["router_name"].isin(ROUTERS)]
        frames.append(df)
        print(f"{phase}: {len(df)} rows kept ({sorted(df['benchmark_name'].unique())})")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nTotal combined: {len(combined)} rows")
    print(combined.groupby(["router_name", "benchmark_name"]).size())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_DIR / "results.csv", index=False)

    per_bm_df = compute_router_benchmark_metrics(combined)
    per_bm_df.to_csv(OUT_DIR / "metrics_per_benchmark.csv", index=False)

    overall_df = compute_router_overall_metrics(combined)
    overall_df.to_csv(OUT_DIR / "metrics_overall.csv", index=False)

    frontier_df = compute_pareto_frontier(overall_df)
    frontier_df.to_csv(OUT_DIR / "pareto_frontier.csv", index=False)

    print("\n=== Paper 1 live ranking (mean success rate, overall across 4 benchmarks) ===")
    cols = ["router_name", "mean_success_rate", "mean_cost_per_task_usd", "mean_latency_p50_ms", "is_pareto_optimal"]
    print(overall_df[cols].sort_values("mean_success_rate", ascending=False).to_string(index=False))

    plots_dir = OUT_DIR / "plots"
    generate_all_plots(combined, per_bm_df, overall_df, plots_dir, label="live evaluation (Paper 1 subset)")
    print(f"\nOutput written under: {OUT_DIR}")


if __name__ == "__main__":
    main()
