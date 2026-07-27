#!/usr/bin/env python3
"""Share-matched permutation analysis for vLLM Semantic Router routes.

For each benchmark, this analysis keeps vLLM's observed candidate-tier counts
fixed and randomly reassigns those tier labels across tasks.  It therefore
tests task-to-tier alignment, not the value of using a more expensive tier more
often.

The exchangeable unit is the TASK, not the individual (task, routing-seed) route
row.  A task's two routing seeds are replicates of one routing decision (they
select the same tier on 289 of 290 tasks), so permuting them independently would
double-count -- inflating the effective sample size, understating the spread of
the permutation null, and making the observed statistic look more extreme than it
is.  Each task's tier labels are therefore permuted as one block that stays
matched to that task's success cells; this leaves the observed statistic
unchanged and widens the null to its correct task-level spread.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from router_benchmark.protocol.canonical import validate_bundle
from router_benchmark.protocol.protocol_tools import load_yaml


VLLM_ROUTER_NAME = "vLLM Semantic Router (live)"
DEFAULT_DRAWS = 10_000
DEFAULT_SEED = 20260722

# Equivalence / noninferiority margin on the share-alignment effect
# (actual share-matched success minus the permutation null mean).  The route
# statistic is a mean success rate in [0, 1]; 0.05 (five success-rate
# points) is the smallest task-to-tier alignment advantage we treat as
# practically meaningful.  A benchmark is "equivalent" when the entire null
# percentile interval for the effect falls inside +/- this margin, i.e. vLLM's
# alignment buys nothing beyond its observed tier shares.
DEFAULT_EQUIVALENCE_MARGIN = 0.05


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _holm(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=p_values.__getitem__)
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("permutation draws cannot be empty")
    return sorted_values[min(len(sorted_values) - 1, int(probability * len(sorted_values)))]


def share_matched_permutation_rows(
    bundle: Path,
    protocol: Path,
    *,
    draws: int = DEFAULT_DRAWS,
    seed: int = DEFAULT_SEED,
    excluded_benchmarks: set[str] | None = None,
    equivalence_margin: float = DEFAULT_EQUIVALENCE_MARGIN,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Return per-benchmark vLLM alignment results from a locked bundle.

    ``equivalence_margin`` sets the +/- band (in success-rate units) used for
    the noninferiority / equivalence verdict on the actual-minus-null effect.
    ``excluded_benchmarks`` still defaults to the empty set, so the historical
    call with ``{"WebArena (live)"}`` is unchanged; passing ``set()`` now
    includes WebArena (a two-tier cheap/mid split) alongside the others.
    """
    validate_bundle(bundle, protocol)
    protocol_doc = load_yaml(protocol)
    excluded_benchmarks = excluded_benchmarks or set()
    if draws < 1:
        raise ValueError("draws must be positive")
    if equivalence_margin < 0:
        raise ValueError("equivalence_margin must be non-negative")

    router_configs = json.loads((bundle / "router_configs.json").read_text(encoding="utf-8"))
    matching = [config_id for config_id, value in router_configs.items() if value["router_name"] == VLLM_ROUTER_NAME]
    if len(matching) != 1:
        raise ValueError(f"Expected exactly one {VLLM_ROUTER_NAME} configuration, found {len(matching)}")
    vllm_config = matching[0]

    candidate_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in _read_csv(bundle / "candidate_outcomes.csv"):
        candidate_values[(row["benchmark_id"], row["task_id"], row["candidate_id"])].append(
            float(row["success"].lower() == "true")
        )
    candidate_means = {key: statistics.fmean(values) for key, values in candidate_values.items()}

    routes_by_benchmark: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(bundle / "routes.csv"):
        if row["router_config_id"] == vllm_config and row["benchmark_id"] not in excluded_benchmarks:
            routes_by_benchmark[row["benchmark_id"]].append(row)

    expected = set(protocol_doc["benchmarks"]) - excluded_benchmarks
    if set(routes_by_benchmark) != expected:
        raise ValueError("vLLM routes must cover every non-excluded protocol benchmark")

    rows: list[dict[str, object]] = []
    for index, benchmark in enumerate(sorted(routes_by_benchmark)):
        routes = sorted(routes_by_benchmark[benchmark], key=lambda row: (row["task_id"], int(row["routing_seed"])))
        tiers = [row["selected_candidate"] for row in routes]
        actual = statistics.fmean(candidate_means[(benchmark, row["task_id"], row["selected_candidate"])] for row in routes)
        # Group route rows into per-task blocks (seed order preserved) so the
        # permutation exchanges tier labels across TASKS, not across individual
        # (task, seed) positions. Each task keeps every one of its route rows, so
        # the observed statistic is unchanged; only the null's exchangeable unit
        # -- and therefore its spread -- is corrected from 2*T positions to T tasks.
        by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in routes:
            by_task[row["task_id"]].append(row)
        task_ids = sorted(by_task)
        task_tier_blocks = [[row["selected_candidate"] for row in by_task[task_id]] for task_id in task_ids]
        rng = random.Random(seed + index)
        permutation_draws: list[float] = []
        for _ in range(draws):
            shuffled_blocks = task_tier_blocks.copy()
            rng.shuffle(shuffled_blocks)
            permutation_draws.append(
                statistics.fmean(
                    candidate_means[(benchmark, task_ids[position], tier)]
                    for position in range(len(task_ids))
                    for tier in shuffled_blocks[position]
                )
            )
        permutation_draws.sort()
        null_mean = statistics.fmean(permutation_draws)
        greater_or_equal = sum(value >= actual for value in permutation_draws)
        less_or_equal = sum(value <= actual for value in permutation_draws)
        upper_p = (greater_or_equal + 1) / (draws + 1)
        lower_p = (less_or_equal + 1) / (draws + 1)
        null_ci_low = _percentile(permutation_draws, 0.025)
        null_ci_high = _percentile(permutation_draws, 0.975)
        # Percentile interval for the actual-minus-null effect, obtained by
        # subtracting the null percentiles from the observed statistic.
        effect = actual - null_mean
        effect_ci_low = actual - null_ci_high
        effect_ci_high = actual - null_ci_low
        interval_within_margin = (
            effect_ci_low >= -equivalence_margin and effect_ci_high <= equivalence_margin
        )
        point_within_margin = abs(effect) <= equivalence_margin
        if interval_within_margin:
            equivalence_verdict = "equivalent"
        elif effect_ci_low > equivalence_margin:
            equivalence_verdict = "exceeds_margin_above"
        elif effect_ci_high < -equivalence_margin:
            equivalence_verdict = "exceeds_margin_below"
        else:
            equivalence_verdict = "inconclusive"
        rows.append({
            "benchmark_id": benchmark,
            "router_name": VLLM_ROUTER_NAME,
            "route_positions": len(routes),
            "task_units": len(task_ids),
            "cheap_routes": Counter(tiers)["cheap-small"],
            "mid_routes": Counter(tiers)["mid-general"],
            "strong_routes": Counter(tiers)["strong-frontier"],
            "actual_success": actual,
            "null_mean_success": null_mean,
            "actual_minus_null": effect,
            "null_ci_low": null_ci_low,
            "null_ci_high": null_ci_high,
            "effect_ci_low": effect_ci_low,
            "effect_ci_high": effect_ci_high,
            "equivalence_margin": equivalence_margin,
            "point_within_margin": point_within_margin,
            "equivalence_verdict": equivalence_verdict,
            "one_sided_p_upper": upper_p,
            "one_sided_p_lower": lower_p,
            "two_sided_p_value": min(1.0, 2 * min(upper_p, lower_p)),
        })
    adjusted = _holm([float(row["two_sided_p_value"]) for row in rows])
    comparisons = len(rows)
    multiplicity_scope = f"{comparisons} comparative benchmark-level permutation tests"
    for row, value in zip(rows, adjusted):
        row["holm_adjusted_two_sided_p_value"] = value
        row["multiplicity_scope"] = multiplicity_scope

    metadata = {
        "analysis": "share-matched tier-label permutation across tasks (per-task blocks; routing seeds tied)",
        "router_name": VLLM_ROUTER_NAME,
        "draws": draws,
        "seed": seed,
        "excluded_benchmarks": sorted(excluded_benchmarks),
        "benchmarks_tested": [row["benchmark_id"] for row in rows],
        "permutation_unit": (
            "task (both routing-seed route rows of a task permuted as one block, "
            "not double-counted as independent positions)"
        ),
        "equivalence_margin": equivalence_margin,
        "equivalence_definition": (
            "effect = actual - null_mean; verdict 'equivalent' when the 2.5/97.5 "
            "null percentile interval for the effect lies entirely within +/- the margin"
        ),
        "null_hypothesis": "observed vLLM per-task tier assignments are independent of task identity",
        "success_value": "mean of saved candidate outcome replicates for each task-tier cell",
        "multiplicity_scope": (
            f"{comparisons} comparative benchmark-level two-sided tests adjusted by Holm"
        ),
    }
    return rows, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--exclude-benchmark", action="append", default=[])
    parser.add_argument("--equivalence-margin", type=float, default=DEFAULT_EQUIVALENCE_MARGIN)
    args = parser.parse_args()

    rows, metadata = share_matched_permutation_rows(
        args.bundle,
        args.protocol,
        draws=args.draws,
        seed=args.seed,
        excluded_benchmarks=set(args.exclude_benchmark),
        equivalence_margin=args.equivalence_margin,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
