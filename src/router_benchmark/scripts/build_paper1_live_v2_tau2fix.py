"""Second correction pass on top of build_paper1_live_v2.py.

tau2-bench was scaled from n=8 (original pilot) to n=100 in phase10
(all 6 routers, run 2026-07-08). phase10 has a harness bug specific to
vLLM Semantic Router and NVIDIA AI Blueprint LLM Router: cost_usd is
exactly 0.0 on every row for both, tau2's own CLI output reports
"Infra Errors" (excluded from its metrics) rather than a graded
failure, and both fail 100% regardless of which candidate tier they
picked -- while other routers picking the identical model (e.g.
Aurelio also selecting "mid-general" -> claude-sonnet-4-6 on the same
task) succeed normally. This is a harness/infra failure, not real
router performance.

vLLM Semantic Router was re-run cleanly as phase11 (2026-07-08), fixing
the bug for this router: 0.85 success, non-zero cost, realistic
latency, matching the pattern of every other successfully-run router
in phase10. This script replaces vLLM's broken phase10 tau2-bench rows
with phase11's, and restricts the merge to this paper's actual 4-router
scope (RouteLLM, LiteLLM Router, vLLM Semantic Router, Aurelio Semantic
Router) -- LLMRouter and NVIDIA AI Blueprint LLM Router are out of
scope for paper1.tex (Related Work mentions only) and are dropped here,
including NVIDIA's still-unfixed phase10 tau2-bench rows.

WebArena was scaled from n=16 (original pilot) to n=100 in phase9 (all
6 routers, run 2026-07-08); no bug was found there for any of the 4
paper routers (non-zero costs, realistic 60-125s per-task browser
latencies for all four).

    python -m router_benchmark.build_paper1_live_v2_tau2fix
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

PAPER_ROUTERS = {
    "RouteLLM (live)",
    "LiteLLM Router (live)",
    "vLLM Semantic Router (live)",
    "Aurelio Semantic Router (live)",
}
FIXED_ROUTERS = {"LiteLLM Router (live)", "Aurelio Semantic Router (live)"}
UNCHANGED_ROUTERS = {"RouteLLM (live)", "vLLM Semantic Router (live)"}

LIVE_ROOT = REPO_ROOT / "output" / "live"
OUT_DIR = LIVE_ROOT / "paper1_live_v2"


def main() -> None:
    frames = []

    # RouterBench + BFCL v4: unchanged from build_paper1_live_v2.py
    rb_bfcl_fixed = pd.read_csv(LIVE_ROOT / "phase1_fix" / "results.csv")
    rb_bfcl_fixed = rb_bfcl_fixed[rb_bfcl_fixed["router_name"].isin(FIXED_ROUTERS)]
    frames.append(rb_bfcl_fixed)
    print(f"phase1_fix (fixed adapters): {len(rb_bfcl_fixed)} rows kept")

    for phase in ["phase2", "phase2b"]:
        df = pd.read_csv(LIVE_ROOT / phase / "results.csv")
        df = df[df["router_name"].isin(UNCHANGED_ROUTERS)]
        frames.append(df)
        print(f"{phase} (unchanged): {len(df)} rows kept")

    # tau2-bench: phase10 for the 3 routers whose runs are clean;
    # phase11 (repair run) for vLLM Semantic Router.
    tau2_phase10 = pd.read_csv(LIVE_ROOT / "phase10" / "results.csv")
    tau2_phase10 = tau2_phase10[
        tau2_phase10["router_name"].isin(PAPER_ROUTERS - {"vLLM Semantic Router (live)"})
    ]
    frames.append(tau2_phase10)
    print(f"phase10 tau2-bench (Aurelio/LiteLLM/RouteLLM, clean): {len(tau2_phase10)} rows kept")

    tau2_phase11 = pd.read_csv(LIVE_ROOT / "phase11" / "results.csv")
    tau2_phase11 = tau2_phase11[tau2_phase11["router_name"] == "vLLM Semantic Router (live)"]
    frames.append(tau2_phase11)
    print(f"phase11 tau2-bench (vLLM Semantic Router, repair run): {len(tau2_phase11)} rows kept")

    # WebArena: phase9, restricted to this paper's 4 routers.
    webarena = pd.read_csv(LIVE_ROOT / "phase9" / "results.csv")
    webarena = webarena[webarena["router_name"].isin(PAPER_ROUTERS)]
    frames.append(webarena)
    print(f"phase9 WebArena (4-router subset): {len(webarena)} rows kept")

    combined = pd.concat(frames, ignore_index=True)
    assert set(combined["router_name"]) == PAPER_ROUTERS, set(combined["router_name"])
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

    print("\n=== Paper 1 (v2, tau2 harness-bug fix) live ranking (mean success rate, overall across 4 benchmarks) ===")
    cols = ["router_name", "mean_success_rate", "mean_cost_per_task_usd", "mean_latency_p50_ms", "is_pareto_optimal"]
    print(overall_df[cols].sort_values("mean_success_rate", ascending=False).to_string(index=False))

    print("\n=== Per-benchmark breakdown ===")
    pb_cols = ["router_name", "benchmark_name", "success_rate", "cost_per_task_usd", "latency_p50_ms", "fallback_rate"]
    print(per_bm_df[pb_cols].sort_values(["benchmark_name", "success_rate"], ascending=[True, False]).to_string(index=False))

    plots_dir = OUT_DIR / "plots"
    generate_all_plots(combined, per_bm_df, overall_df, plots_dir, label="live evaluation (Paper 1 subset, tau2 harness-bug fix)")
    print(f"\nOutput written under: {OUT_DIR}")


if __name__ == "__main__":
    main()
