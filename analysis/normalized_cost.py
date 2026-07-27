#!/usr/bin/env python3
"""Cost normalized to the strong-frontier candidate's real per-task cost,
so the Pareto-frontier shape survives future price changes even though
the raw-dollar tables (tied to PRICING_ASOF 2026-07-02) will not.
Reads sweep_results.csv (Task 1) for the strong-frontier baseline cost
per task, and paper1_live_v3/results.csv for each router's real cost.

CANONICAL variant (main_canonical / success_per_dollar): the reviewer
flagged the legacy two-lineage sourcing above as a provenance liability, so
success-per-dollar is computed entirely from the CANONICAL paper1 bundle and
its canonical analysis dir -- one lineage for all seven policies (four routers
plus the three fixed-tier baselines). success_per_dollar = success_rate /
cost_per_task, reported under both cost bases (candidate-only, and
candidate+router-service-fee). See main_canonical()."""
import csv
from collections import defaultdict
from pathlib import Path

# --- canonical single-lineage sources -------------------------------------
CANON_DIR = Path(__file__).parent / "output" / "paper1_canonical"
METRIC_SUITE = CANON_DIR / "canonical_metric_suite_per_benchmark.csv"
BASELINE_SUMMARY = CANON_DIR / "baseline_summary.csv"

BENCHES = ["RouterBench", "BFCL", "tau2-bench", "WebArena"]
ROUTER_ORDER = [
    "Aurelio Semantic Router",
    "vLLM Semantic Router",
    "LiteLLM Router",
    "RouteLLM",
]
BASELINE_ORDER = ["Always-Cheapest", "Always-Mid", "Always-Strongest"]
POLICY_ORDER = ROUTER_ORDER + BASELINE_ORDER

# Flat synthetic per-route router-service fee (USD), matching
# analysis/expected_utility.py SERVICE_FEE. Baselines run no router -> 0.
SERVICE_FEE = {
    "Aurelio Semantic Router": 0.01,
    "vLLM Semantic Router": 0.05,
    "LiteLLM Router": 0.00,
    "RouteLLM": 0.01,
    "Always-Cheapest": 0.00,
    "Always-Mid": 0.00,
    "Always-Strongest": 0.00,
}
COST_BASES = ["candidate", "candidate_plus_service"]

_BENCH_ALIAS = {
    "RouterBench (live)": "RouterBench",
    "BFCL v4 (live)": "BFCL",
    "tau2-bench (live)": "tau2-bench",
    "WebArena (live)": "WebArena",
}


def _read_router_cells(path):
    """{policy: {bench: (success_rate, candidate_cost_per_task)}} for the 4 routers."""
    out = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["router"] == "router":  # stray header row guard
                continue
            out[r["router"]][r["benchmark"]] = (
                float(r["success_rate"]), float(r["cost_per_task_usd"])
            )
    return out


def _read_baseline_cells(path):
    """{policy: {bench: (success_rate, candidate_cost_per_task)}} for the 3 fixed tiers."""
    out = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            policy = r["baseline_id"].replace(" Baseline (live)", "")
            bench = _BENCH_ALIAS.get(r["benchmark_id"], r["benchmark_id"])
            out[policy][bench] = (
                float(r["success_rate"]), float(r["model_api_cost_usd_mean"])
            )
    return out


def main_canonical():
    cells = {}
    cells.update(_read_router_cells(METRIC_SUITE))
    cells.update(_read_baseline_cells(BASELINE_SUMMARY))
    policies = [p for p in POLICY_ORDER if p in cells]

    out_path = CANON_DIR / "success_per_dollar.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "policy", "benchmark", "cost_basis",
            "success_rate", "cost_per_task_usd", "success_per_dollar",
        ])
        for pol in policies:
            fee = SERVICE_FEE.get(pol, 0.0)
            for basis in COST_BASES:
                add = fee if basis == "candidate_plus_service" else 0.0
                # per-benchmark rows
                srs, costs = [], []
                for b in BENCHES:
                    if b not in cells[pol]:
                        continue
                    sr, cand_cost = cells[pol][b]
                    cost = cand_cost + add
                    srs.append(sr)
                    costs.append(cost)
                    spd = sr / cost if cost > 0 else float("inf")
                    w.writerow([pol, b, basis, f"{sr:.6f}", f"{cost:.6f}", f"{spd:.4f}"])
                # aggregate row: uniform mean success / uniform mean cost across benches
                if srs:
                    msr = sum(srs) / len(srs)
                    mcost = sum(costs) / len(costs)
                    spd = msr / mcost if mcost > 0 else float("inf")
                    w.writerow([pol, "AGGREGATE", basis, f"{msr:.6f}",
                                f"{mcost:.6f}", f"{spd:.4f}"])
    print(f"wrote {out_path}")
    return out_path


def group_key(benchmark_name):
    if "RouterBench" in benchmark_name:
        return "RouterBench"
    if "BFCL" in benchmark_name:
        return "BFCL"
    if "tau2" in benchmark_name:
        return "tau2-bench"
    if "WebArena" in benchmark_name:
        return "WebArena"
    return benchmark_name


def main():
    sweep_path = Path(__file__).parent / "../output/live/sweep_v1/sweep_results.csv"
    with open(sweep_path) as f:
        sweep = list(csv.DictReader(f))
    strong_cost = {
        (r["benchmark_name"], r["task_id"]): float(r["cost_usd"])
        for r in sweep if r["candidate_tier"] == "strong-frontier"
    }

    results_path = Path(__file__).parent / "../output/results.csv"
    with open(results_path) as f:
        results = list(csv.DictReader(f))

    ratios = defaultdict(list)
    for r in results:
        key = (r["benchmark_name"], r["task_id"])
        if key not in strong_cost or strong_cost[key] == 0:
            continue
        ratio = float(r["cost_usd"]) / strong_cost[key]
        ratios[(r["router_name"], group_key(r["benchmark_name"]))].append(ratio)

    out_path = Path(__file__).parent / "output" / "normalized_cost.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["router", "benchmark_group", "mean_cost_ratio_to_strong_frontier", "n_tasks"])
        for (router, group), vals in sorted(ratios.items()):
            w.writerow([router, group, f"{sum(vals)/len(vals):.4f}", len(vals)])
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main_canonical()
