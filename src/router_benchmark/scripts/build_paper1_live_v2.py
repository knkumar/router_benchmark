"""Rebuild of Paper 1's live-evaluation scope after fixing two adapter
bugs (see router_bench_adapter_bugs memory): LiteLLMRouterLive never
called the real litellm.Router it constructed (hardcoded difficulty
ladder instead), and AurelioSemanticRouterLive silently defaulted to
"mid-general" with a fabricated confidence whenever semantic-router found
no match. Both are fixed in live_routers.py; this script assembles the
corrected results for LiteLLM Router + Aurelio Semantic Router (phase1_fix,
phase3_fix, phase7c_fix) with the unchanged RouteLLM + vLLM Semantic Router
rows (phase2, phase2b, phase3, phase7c filtered to those two routers only
-- no bug was found in either adapter).

    python -m router_benchmark.build_paper1_live_v2
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

FIXED_ROUTERS = {"LiteLLM Router (live)", "Aurelio Semantic Router (live)"}
UNCHANGED_ROUTERS = {"RouteLLM (live)", "vLLM Semantic Router (live)"}

LIVE_ROOT = REPO_ROOT / "output" / "live"
OUT_DIR = LIVE_ROOT / "paper1_live_v2"

# (fixed-adapter source phase, unchanged-adapter source phase(s))
SOURCES = [
    ("phase1_fix", ["phase2", "phase2b"]),  # RouterBench + BFCL v4
    ("phase10", []),  # tau2-bench (all routers)
    ("phase9", []),  # WebArena (all routers)
]


def main() -> None:
    frames = []
    for fixed_phase, unchanged_phases in SOURCES:
        fixed_df = pd.read_csv(LIVE_ROOT / fixed_phase / "results.csv")
        
        # For phase9 and phase10, we want ALL routers, not just FIXED_ROUTERS
        if fixed_phase in ["phase9", "phase10"]:
            frames.append(fixed_df)
            print(f"{fixed_phase} (all routers): {len(fixed_df)} rows kept")
        else:
            fixed_df = fixed_df[fixed_df["router_name"].isin(FIXED_ROUTERS)]
            frames.append(fixed_df)
            print(f"{fixed_phase} (fixed adapters): {len(fixed_df)} rows kept")

        for unchanged_phase in unchanged_phases:
            unchanged_df = pd.read_csv(LIVE_ROOT / unchanged_phase / "results.csv")
            unchanged_df = unchanged_df[unchanged_df["router_name"].isin(UNCHANGED_ROUTERS)]
            frames.append(unchanged_df)
            print(f"{unchanged_phase} (unchanged): {len(unchanged_df)} rows kept")

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

    print("\n=== Paper 1 (v2, fixed adapters) live ranking (mean success rate, overall across 4 benchmarks) ===")
    cols = ["router_name", "mean_success_rate", "mean_cost_per_task_usd", "mean_latency_p50_ms", "is_pareto_optimal"]
    print(overall_df[cols].sort_values("mean_success_rate", ascending=False).to_string(index=False))

    print("\n=== Per-benchmark breakdown ===")
    pb_cols = ["router_name", "benchmark_name", "success_rate", "cost_per_task_usd", "latency_p50_ms", "fallback_rate"]
    print(per_bm_df[pb_cols].sort_values(["benchmark_name", "success_rate"], ascending=[True, False]).to_string(index=False))

    plots_dir = OUT_DIR / "plots"
    generate_all_plots(combined, per_bm_df, overall_df, plots_dir, label="live evaluation (Paper 1 subset, fixed adapters)")
    print(f"\nOutput written under: {OUT_DIR}")


if __name__ == "__main__":
    main()
