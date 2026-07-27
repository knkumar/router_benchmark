#!/usr/bin/env python3
"""Cost- and latency-aware expected utility per policy (existing-data only).

Reviewer point: the benchmark-mixture table reweights success only and carries no
router cost, latency, or value-of-success term, so it cannot express a genuine
deployment utility. This script computes

    U(R) = sum_b w_b [ V * s(R,b) - lambda_c * c(R,b) - lambda_l * ell(R,b) ]

from the canonical per-benchmark metric suite (success s, candidate cost per task,
median latency ell in seconds), under uniform benchmark weights w_b, unit cost
weight lambda_c = 1 (dollars at face value), a grid of the value-of-success V, and
an optional latency price lambda_l.

Cost is reported under two bases so the paper can show the router-service fee is
disclosed and bounded rather than hidden:
  * ``candidate``               -- candidate-model API cost only (measured, dated).
  * ``candidate_plus_service``  -- candidate cost plus the router-service fee.

The router-service fee is a flat per-route charge, verified constant per router in
the canonical routes.csv (Aurelio/RouteLLM $0.01, vLLM $0.05, LiteLLM $0). It is a
configurable nominal fee, not a measured market price, so main() also prints the
Aurelio-vs-LiteLLM crossover as a function of that fee: the qualitative conclusion
(a success must be worth roughly half a dollar before the mid-tier default repays
its premium) barely moves with the fee, so the fee is not load-bearing. Fixed-tier
baselines (Always-Cheapest/Mid/Strongest) run no router, so their service fee is 0;
they are included as rows so the table compares router utilities against a fixed
tier declaration, not only raw success. Infrastructure cost was not recorded and is
therefore excluded from every basis.

This writes only the analysis CSV. The paper LaTeX fragments are rendered from that
CSV by scripts/generate_paper1_canonical_tables.py, so the canonical driver remains
the single source of the paper tables.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

BENCHES = ["RouterBench", "BFCL", "tau2-bench", "WebArena"]
# Paper display order for the router rows, then the fixed-tier baselines.
ROUTER_ORDER = [
    "Aurelio Semantic Router",
    "vLLM Semantic Router",
    "LiteLLM Router",
    "RouteLLM",
]
BASELINE_ORDER = ["Always-Cheapest", "Always-Mid", "Always-Strongest"]
POLICY_ORDER = ROUTER_ORDER + BASELINE_ORDER

# Flat synthetic per-route router-service fee (USD). Baselines run no router -> 0.
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

# (value-of-success V in USD, latency price lambda_l in USD/second).
COLUMNS = [
    (0.10, 0.0),
    (0.50, 0.0),
    (1.00, 0.0),
    (10.0, 0.0),
    (1.00, 0.001),
]
LAMBDA_C = 1.0
WEIGHT = 0.25  # uniform over the four benchmarks

_BENCH_ALIAS = {
    "RouterBench (live)": "RouterBench",
    "BFCL v4 (live)": "BFCL",
    "tau2-bench (live)": "tau2-bench",
    "WebArena (live)": "WebArena",
}

# (success, candidate_cost_usd, latency_seconds) per benchmark.
Row = dict[str, tuple[float, float, float]]


def _read_suite(path: Path) -> dict[str, Row]:
    data: dict[str, Row] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["router"] == "router":  # a stray header row is present in the CSV
                continue
            data[r["router"]][r["benchmark"]] = (
                float(r["success_rate"]),
                float(r["cost_per_task_usd"]),
                float(r["latency_p50_ms"]) / 1000.0,
            )
    return data


def _read_baselines(path: Path) -> dict[str, Row]:
    data: dict[str, Row] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            policy = r["baseline_id"].replace(" Baseline (live)", "")
            bench = _BENCH_ALIAS.get(r["benchmark_id"], r["benchmark_id"])
            # p50 latency keeps baselines consistent with the routers, whose metric
            # suite reports latency_p50_ms; the mean over-penalizes tiers with a long
            # WebArena tail. Fall back to the mean for older baseline summaries.
            latency_ms = r.get("generation_latency_ms_p50") or r["generation_latency_ms_mean"]
            data[policy][bench] = (
                float(r["success_rate"]),
                float(r["model_api_cost_usd_mean"]),
                float(latency_ms) / 1000.0,
            )
    return data


def _cost(cell: tuple[float, float, float], fee: float, basis: str) -> float:
    return cell[1] + (fee if basis == "candidate_plus_service" else 0.0)


def utility(row: Row, fee: float, value: float, lam_l: float, basis: str) -> float:
    return sum(
        WEIGHT * (value * row[b][0] - LAMBDA_C * _cost(row[b], fee, basis) - lam_l * row[b][2])
        for b in BENCHES
    )


def crossover_value(a: Row, fee_a: float, b: Row, fee_b: float, basis: str) -> float:
    """Value-of-success at which mean utilities of a and b are equal (lambda_l = 0):
    V* = mean cost gap / mean success gap."""
    ds = sum(WEIGHT * (a[x][0] - b[x][0]) for x in BENCHES)
    dc = sum(WEIGHT * (_cost(a[x], fee_a, basis) - _cost(b[x], fee_b, basis)) for x in BENCHES)
    return dc / ds if ds else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).parent / "output" / "paper1_canonical"
    ap.add_argument("--metric-suite", type=Path, default=base / "canonical_metric_suite_per_benchmark.csv")
    ap.add_argument(
        "--baseline-summary",
        type=Path,
        default=None,
        help="baseline_summary.csv; defaults to the sibling of --metric-suite",
    )
    ap.add_argument("--csv-output", type=Path, default=base / "expected_utility.csv")
    args = ap.parse_args()

    suite = _read_suite(args.metric_suite)
    baseline_path = args.baseline_summary or (args.metric_suite.parent / "baseline_summary.csv")
    baselines = _read_baselines(baseline_path) if baseline_path.exists() else {}

    combined: dict[str, Row] = {}
    for name in ROUTER_ORDER:
        if name in suite:
            combined[name] = suite[name]
    for name in BASELINE_ORDER:
        if name in baselines:
            combined[name] = baselines[name]
    policies = [p for p in POLICY_ORDER if p in combined]

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "router", "cost_basis", "value_of_success_usd",
            "latency_price_usd_per_s", "utility_usd_per_task",
        ])
        for pol in policies:
            fee = SERVICE_FEE.get(pol, 0.0)
            for basis in COST_BASES:
                for value, lam_l in COLUMNS:
                    # High precision so the driver's display rounding matches the
                    # true value rather than a pre-rounded CSV cell.
                    u = utility(combined[pol], fee, value, lam_l, basis)
                    w.writerow([pol, basis, value, lam_l, f"{u:.10f}"])

    print(f"wrote {args.csv_output}")
    a, b = combined["Aurelio Semantic Router"], combined["LiteLLM Router"]
    fa, fb = SERVICE_FEE["Aurelio Semantic Router"], SERVICE_FEE["LiteLLM Router"]
    for basis in COST_BASES:
        print(f"  Aurelio vs LiteLLM crossover V* ({basis}) = ${crossover_value(a, fa, b, fb, basis):.3f}")
    # Fee sensitivity: Aurelio pays fee f on every benchmark (weights sum to 1),
    # LiteLLM pays 0, so V*(f) = (candidate cost gap + f) / success gap.
    ds = sum(WEIGHT * (a[x][0] - b[x][0]) for x in BENCHES)
    dc0 = sum(WEIGHT * (a[x][1] - b[x][1]) for x in BENCHES)
    for f in (0.0, 0.01, 0.05):
        print(f"  fee-sensitivity V*(service_fee=${f:.2f}) = ${(dc0 + f) / ds:.3f}")


if __name__ == "__main__":
    main()
