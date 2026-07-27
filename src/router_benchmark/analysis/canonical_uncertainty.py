#!/usr/bin/env python3
"""Canonical rank and Pareto uncertainty from a locked rebuild bundle."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from router_benchmark.analysis.decision_uncertainty import (
    average_tie_ranks,
    pareto_nondominance_probabilities,
    rank_uncertainty,
)
from router_benchmark.protocol.canonical import validate_bundle
from router_benchmark.protocol.protocol_tools import load_yaml


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"{path.name} cannot be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_draws(path: Path) -> dict[str, list[tuple[tuple[str, int], ...]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    draws: dict[str, list[tuple[tuple[str, int], ...]]] = {}
    for benchmark, benchmark_draws in raw.items():
        draws[benchmark] = [
            tuple((str(task), int(replicate)) for task, replicate in draw)
            for draw in benchmark_draws
        ]
    return draws


def _router_names(bundle: Path) -> dict[str, str]:
    config = json.loads((bundle / "router_configs.json").read_text(encoding="utf-8"))
    return {config_id: value["router_name"] for config_id, value in config.items()}


def _task_replicate_metrics(
    bundle: Path,
    protocol: dict[str, Any],
) -> tuple[
    dict[tuple[str, str, str, int], list[float]],
    dict[tuple[str, str, str, int], list[float]],
]:
    candidate_by_key = {
        (row["benchmark_id"], row["task_id"], row["candidate_id"], row["outcome_replicate"]): row
        for row in _read_csv(bundle / "candidate_outcomes.csv")
    }
    routes = {
        (row["router_config_id"], row["benchmark_id"], row["task_id"], row["routing_seed"]): row
        for row in _read_csv(bundle / "routes.csv")
    }
    router_names = _router_names(bundle)
    success: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    cost: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for row in _read_csv(bundle / "outcomes.csv"):
        route_key = (row["router_config_id"], row["benchmark_id"], row["task_id"], row["routing_seed"])
        route = routes[route_key]
        replicate = int(row["outcome_replicate"])
        candidate_key = (route["benchmark_id"], route["task_id"], route["selected_candidate"], row["outcome_replicate"])
        candidate = candidate_by_key[candidate_key]
        outcome_replicates = protocol["benchmarks"][route["benchmark_id"]]["outcome_replicates_per_task_candidate"]
        metric_key = (route["benchmark_id"], router_names[route["router_config_id"]], route["task_id"], replicate)
        success[metric_key].append(1.0 if candidate["success"].lower() == "true" else 0.0)
        cost[metric_key].append(
            float(candidate["model_api_cost_usd"]) + float(route["router_service_usd"]) / outcome_replicates
        )
    return success, cost


def _mean_metric(
    values: dict[tuple[str, str, str, int], list[float]],
    *,
    benchmark: str,
    router: str,
    task: str,
    replicate: int,
) -> float:
    observed = values[(benchmark, router, task, replicate)]
    if not observed:
        raise ValueError(f"missing metric for {benchmark}/{router}/{task}/{replicate}")
    return sum(observed) / len(observed)


# Fixed-tier deployable policies, keyed by the candidate tier each one always picks.
BASELINE_TIERS = {
    "Always-Cheapest": "cheap-small",
    "Always-Mid": "mid-general",
    "Always-Strongest": "strong-frontier",
}
BASELINE_ORDER = ["Always-Cheapest", "Always-Mid", "Always-Strongest"]


def _baseline_task_replicate_metrics(
    bundle: Path,
) -> tuple[
    dict[tuple[str, str, str, int], list[float]],
    dict[tuple[str, str, str, int], list[float]],
]:
    """Per task-replicate success and cost for the three fixed-tier baselines,
    read straight from ``candidate_outcomes.csv``. A fixed-tier policy always
    selects one candidate tier, so its per-task outcome is that tier's candidate
    row; it runs no router, so its cost carries no router-service fee. Keyed the
    same way as :func:`_task_replicate_metrics` (benchmark, policy, task,
    replicate) so the two share the saved task-clustered draws."""
    success: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    cost: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    tier_to_policy = {tier: policy for policy, tier in BASELINE_TIERS.items()}
    for row in _read_csv(bundle / "candidate_outcomes.csv"):
        policy = tier_to_policy.get(row["candidate_id"])
        if policy is None:
            continue
        key = (row["benchmark_id"], policy, row["task_id"], int(row["outcome_replicate"]))
        success[key].append(1.0 if row["success"].lower() == "true" else 0.0)
        cost[key].append(float(row["model_api_cost_usd"]))
    return success, cost


def _draw_metrics(
    bundle: Path,
    protocol_path: Path,
    draws_path: Path,
) -> tuple[
    list[str],
    dict[str, list[dict[str, float]]],
    dict[str, list[list[dict[str, object]]]],
]:
    """Rebuild the per-benchmark success and Pareto draws from saved resampling.

    Returns the router list, ``{benchmark: [success_draw, ...]}`` (each draw a
    ``{router: success}`` map), and ``{benchmark: [pareto_draw, ...]}``. Shared by
    the per-benchmark uncertainty table and the cross-benchmark rank aggregation,
    so both read the identical task-clustered draws.
    """
    validate_bundle(bundle, protocol_path)
    protocol = load_yaml(protocol_path)
    saved_draws = _load_draws(draws_path)
    success_values, cost_values = _task_replicate_metrics(bundle, protocol)
    routers = sorted(protocol["routers"])
    success_by_benchmark: dict[str, list[dict[str, float]]] = {}
    pareto_by_benchmark: dict[str, list[list[dict[str, object]]]] = {}
    for benchmark in sorted(protocol["benchmarks"]):
        benchmark_draws = saved_draws.get(benchmark)
        if not benchmark_draws:
            raise ValueError(f"missing saved draws for {benchmark}")
        success_draws: list[dict[str, float]] = []
        pareto_draws: list[list[dict[str, object]]] = []
        for draw in benchmark_draws:
            success_draw: dict[str, float] = {}
            pareto_draw: list[dict[str, object]] = []
            for router in routers:
                sampled_success = [
                    _mean_metric(success_values, benchmark=benchmark, router=router, task=task, replicate=replicate)
                    for task, replicate in draw
                ]
                sampled_cost = [
                    _mean_metric(cost_values, benchmark=benchmark, router=router, task=task, replicate=replicate)
                    for task, replicate in draw
                ]
                success_mean = sum(sampled_success) / len(sampled_success)
                cost_mean = sum(sampled_cost) / len(sampled_cost)
                success_draw[router] = success_mean
                pareto_draw.append({"router": router, "success": success_mean, "cost": cost_mean})
            success_draws.append(success_draw)
            pareto_draws.append(pareto_draw)
        success_by_benchmark[benchmark] = success_draws
        pareto_by_benchmark[benchmark] = pareto_draws
    return routers, success_by_benchmark, pareto_by_benchmark


def uncertainty_rows(
    bundle: Path,
    protocol_path: Path,
    draws_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    routers, success_by_benchmark, pareto_by_benchmark = _draw_metrics(bundle, protocol_path, draws_path)
    rank_rows: list[dict[str, Any]] = []
    pareto_rows: list[dict[str, Any]] = []
    for benchmark in sorted(success_by_benchmark):
        success_draws = success_by_benchmark[benchmark]
        pareto_draws = pareto_by_benchmark[benchmark]
        benchmark_draws = success_draws
        ranks = rank_uncertainty(success_draws)
        pareto = pareto_nondominance_probabilities(
            pareto_draws, id_key="router", cost_key="cost", success_key="success"
        )
        for router in routers:
            rank_rows.append({
                "benchmark_id": benchmark,
                "router_name": router,
                "draws": len(benchmark_draws),
                "mean_rank": ranks[router]["mean_rank"],
                "rank_variance": ranks[router]["rank_variance"],
                "rank_probabilities_json": json.dumps(ranks[router]["rank_probabilities"], sort_keys=True),
                "pairwise_inversion_probability_json": json.dumps(
                    ranks[router]["pairwise_inversion_probability"], sort_keys=True
                ),
            })
            pareto_rows.append({
                "benchmark_id": benchmark,
                "router_name": router,
                "draws": len(benchmark_draws),
                "pareto_nondominance_probability": pareto[router],
                "success_definition": "mean success over saved task-replicate draws and routing seeds",
                "cost_definition": (
                    "mean candidate model API cost plus router service cost allocated across outcome replicates"
                ),
            })
    return rank_rows, pareto_rows


def all_policy_pareto_rows(
    bundle: Path,
    protocol_path: Path,
    draws_path: Path,
) -> list[dict[str, Any]]:
    """All-policy Pareto nondominance: the four routers plus the three fixed-tier
    baselines, resampled on the SAME saved task-clustered draws as the router-only
    frontier. Router cost carries the router-service fee (as in
    :func:`_task_replicate_metrics`); the fixed-tier policies run no router, so
    their cost is candidate API cost only. This is the deployment-relevant
    universe: a practitioner chooses among routers AND fixed declarations, so a
    router is only interesting if it is nondominated against Always-Cheapest/Mid/
    Strongest, not merely against the other routers."""
    validate_bundle(bundle, protocol_path)
    protocol = load_yaml(protocol_path)
    saved_draws = _load_draws(draws_path)
    r_success, r_cost = _task_replicate_metrics(bundle, protocol)
    b_success, b_cost = _baseline_task_replicate_metrics(bundle)
    routers = sorted(protocol["routers"])
    policies = routers + BASELINE_ORDER

    rows: list[dict[str, Any]] = []
    for benchmark in sorted(protocol["benchmarks"]):
        benchmark_draws = saved_draws.get(benchmark)
        if not benchmark_draws:
            raise ValueError(f"missing saved draws for {benchmark}")
        pareto_draws: list[list[dict[str, object]]] = []
        for draw in benchmark_draws:
            pareto_draw: list[dict[str, object]] = []
            for router in routers:
                s = [_mean_metric(r_success, benchmark=benchmark, router=router, task=t, replicate=rep) for t, rep in draw]
                c = [_mean_metric(r_cost, benchmark=benchmark, router=router, task=t, replicate=rep) for t, rep in draw]
                pareto_draw.append({"policy": router, "success": sum(s) / len(s), "cost": sum(c) / len(c)})
            for policy in BASELINE_ORDER:
                s = [_mean_metric(b_success, benchmark=benchmark, router=policy, task=t, replicate=rep) for t, rep in draw]
                c = [_mean_metric(b_cost, benchmark=benchmark, router=policy, task=t, replicate=rep) for t, rep in draw]
                pareto_draw.append({"policy": policy, "success": sum(s) / len(s), "cost": sum(c) / len(c)})
            pareto_draws.append(pareto_draw)
        pareto = pareto_nondominance_probabilities(
            pareto_draws, id_key="policy", cost_key="cost", success_key="success"
        )
        for policy in policies:
            rows.append({
                "benchmark_id": benchmark,
                "policy_name": policy,
                "policy_type": "router" if policy in routers else "fixed-tier",
                "draws": len(benchmark_draws),
                "pareto_nondominance_probability": pareto[policy],
                "universe": "all-policy (routers + fixed-tier baselines)",
                "cost_definition": (
                    "router: candidate model API cost plus router service cost allocated "
                    "across outcome replicates; fixed-tier: candidate model API cost only (no router)"
                ),
            })
    return rows


def _percentiles(values: list[float], lower: float = 0.025, upper: float = 0.975) -> tuple[float, float]:
    ordered = sorted(values)
    n = len(ordered)
    return ordered[int(lower * n)], ordered[min(n - 1, int(upper * n))]


def cross_benchmark_rank_rows(
    bundle: Path,
    protocol_path: Path,
    draws_path: Path,
) -> list[dict[str, Any]]:
    """Posterior of the cross-benchmark mean rank and rank variance (Definition 4,
    strict per-benchmark grouping) from the saved task-clustered draws.

    Each joint draw takes the aligned ``i``-th resample of every benchmark, ranks
    the routers within each benchmark, and forms each router's mean rank and rank
    variance across the benchmark set. Aggregating over draws propagates the
    per-benchmark success uncertainty into the rank statistics the point table
    reports as bare numbers, so a rank flip driven by a within-noise success gap
    shows up as a wide interval rather than a hard integer change.
    """
    routers, success_by_benchmark, _ = _draw_metrics(bundle, protocol_path, draws_path)
    benchmarks = sorted(success_by_benchmark)
    draw_count = len(success_by_benchmark[benchmarks[0]])
    if any(len(success_by_benchmark[b]) != draw_count for b in benchmarks):
        raise ValueError("benchmarks must share the same number of saved draws")

    mean_rank_samples: dict[str, list[float]] = {r: [] for r in routers}
    variance_samples: dict[str, list[float]] = {r: [] for r in routers}
    best_mean_rank_credit: dict[str, float] = {r: 0.0 for r in routers}
    for index in range(draw_count):
        per_router_ranks: dict[str, list[float]] = {r: [] for r in routers}
        for benchmark in benchmarks:
            ranks = average_tie_ranks(success_by_benchmark[benchmark][index])
            for router in routers:
                per_router_ranks[router].append(ranks[router])
        draw_mean_rank: dict[str, float] = {}
        for router in routers:
            values = per_router_ranks[router]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            mean_rank_samples[router].append(mean)
            variance_samples[router].append(variance)
            draw_mean_rank[router] = mean
        best = min(draw_mean_rank.values())
        winners = [r for r in routers if draw_mean_rank[r] == best]
        for router in winners:
            best_mean_rank_credit[router] += 1.0 / len(winners)

    rows: list[dict[str, Any]] = []
    for router in routers:
        mean_ranks = mean_rank_samples[router]
        variances = variance_samples[router]
        mr_lo, mr_hi = _percentiles(mean_ranks)
        var_lo, var_hi = _percentiles(variances)
        rows.append({
            "router_name": router,
            "draws": draw_count,
            "benchmarks": len(benchmarks),
            "posterior_mean_rank": sum(mean_ranks) / len(mean_ranks),
            "mean_rank_ci_low": mr_lo,
            "mean_rank_ci_high": mr_hi,
            "posterior_rank_variance": sum(variances) / len(variances),
            "rank_variance_ci_low": var_lo,
            "rank_variance_ci_high": var_hi,
            "prob_best_mean_rank": best_mean_rank_credit[router] / draw_count,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--draws", type=Path, required=True)
    parser.add_argument("--rank-output", type=Path, required=True)
    parser.add_argument("--pareto-output", type=Path, required=True)
    parser.add_argument("--rank-consistency-output", type=Path, default=None)
    parser.add_argument("--all-policy-pareto-output", type=Path, default=None)
    args = parser.parse_args()
    rank_rows, pareto_rows = uncertainty_rows(args.bundle, args.protocol, args.draws)
    _write_csv(args.rank_output, rank_rows)
    _write_csv(args.pareto_output, pareto_rows)
    print(f"Canonical rank uncertainty written to {args.rank_output}.")
    print(f"Canonical Pareto uncertainty written to {args.pareto_output}.")
    if args.all_policy_pareto_output is not None:
        all_policy_rows = all_policy_pareto_rows(args.bundle, args.protocol, args.draws)
        _write_csv(args.all_policy_pareto_output, all_policy_rows)
        print(f"All-policy Pareto uncertainty written to {args.all_policy_pareto_output}.")
    if args.rank_consistency_output is not None:
        consistency_rows = cross_benchmark_rank_rows(args.bundle, args.protocol, args.draws)
        _write_csv(args.rank_consistency_output, consistency_rows)
        print(f"Cross-benchmark rank consistency uncertainty written to {args.rank_consistency_output}.")


if __name__ == "__main__":
    main()
