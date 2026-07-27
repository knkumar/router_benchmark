#!/usr/bin/env python3
"""Oracle upper bound, pessimal lower bound, and idealized cheap-first
cascade, computed entirely from the Task-1 exhaustive sweep -- no new API
calls. Oracle = best real outcome across all 3 tiers per task (success
first, then lowest cost as tiebreak). Pessimal = worst real outcome
(priciest failure if any tier fails, else cheapest success -- a
calibrated-to-be-bad baseline, the mirror image of oracle). Cascade =
try cheap-small first; if the real grader marked it a failure, escalate
to mid-general; if that also failed, escalate to strong-frontier. The
cascade is an idealized upper bound (it uses ground-truth success as its
escalation signal, which a real deployed cascade would not have access to
at decision time) -- documented explicitly as a ceiling on cascade
performance, not a deployable policy. See realistic_cascade.py for a
deployable variant using only pre-answer signals.
"""
import csv
from collections import defaultdict
from pathlib import Path

TIER_ORDER = ["cheap-small", "mid-general", "strong-frontier"]

# --- canonical single-lineage oracle --------------------------------------
# The legacy main() above reads the Task-1 sweep (output/live/sweep_v1). The
# reviewer flagged that second lineage as a provenance liability, so the
# canonical oracle is rebuilt directly from the paper1 CANONICAL bundle's
# candidate_outcomes.csv. candidate_outcomes carries multiple outcome
# replicates per (benchmark, task, tier); we collapse replicates to a per-tier
# MEAN success rate and MEAN model_api_cost before choosing the oracle tier
# (success first, then lowest cost). This tier-mean collapse is what
# reproduces the paper's stated oracle success (RouterBench 0.917, BFCL 0.878,
# tau2-bench 0.910, WebArena 0.277); a per-replicate union over-counts it.
CANON_BUNDLE = (
    Path(__file__).parent
    / "../output/live/paper1_canonical_webarena_repair_v2/candidate_outcomes.csv"
)


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def build_canonical_per_task_tiers(bundle=CANON_BUNDLE):
    """{(benchmark_id, task_id): {tier: (mean_success_rate, mean_cost)}} from
    candidate_outcomes.csv, replicates collapsed to per-tier means."""
    raw = defaultdict(lambda: defaultdict(list))
    with open(bundle) as f:
        for r in csv.DictReader(f):
            raw[(r["benchmark_id"], r["task_id"])][r["candidate_id"]].append(
                (r["success"].strip().lower() == "true",
                 float(r["model_api_cost_usd"]))
            )
    tiers = {}
    for key, per_tier in raw.items():
        tiers[key] = {
            t: (_mean([1.0 if s else 0.0 for s, _ in reps]),
                _mean([c for _, c in reps]))
            for t, reps in per_tier.items()
        }
    return tiers


def build_canonical_oracle(bundle=CANON_BUNDLE):
    """{(benchmark_id, task_id): (oracle_success_rate, oracle_cost, oracle_tier)}.
    Oracle tier = max mean-success, ties broken by lowest mean cost."""
    tiers = build_canonical_per_task_tiers(bundle)
    oracle = {}
    for key, per_tier in tiers.items():
        best = min(per_tier, key=lambda t: (-per_tier[t][0], per_tier[t][1]))
        sr, cost = per_tier[best]
        oracle[key] = (sr, cost, best)
    return oracle


def main_canonical():
    """Write the canonical per-task oracle for provenance / spot checks."""
    oracle = build_canonical_oracle()
    out_path = Path(__file__).parent / "output" / "paper1_canonical" / "oracle_per_task_canonical.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark_id", "task_id", "oracle_success_rate",
                    "oracle_cost_usd", "oracle_tier"])
        for (bench, task), (sr, cost, tier) in sorted(oracle.items()):
            w.writerow([bench, task, f"{sr:.6f}", f"{cost:.8f}", tier])
    print(f"wrote {out_path}")
    return out_path


def main():
    src = Path(__file__).parent / "../output/live/sweep_v1/sweep_results.csv"
    with open(src) as f:
        rows = list(csv.DictReader(f))

    by_task = defaultdict(dict)
    for r in rows:
        by_task[(r["benchmark_name"], r["task_id"])][r["candidate_tier"]] = r

    out_path = Path(__file__).parent / "output" / "oracle_and_cascade.csv"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "benchmark_name", "task_id",
            "oracle_success", "oracle_cost", "oracle_tier",
            "pessimal_success", "pessimal_cost", "pessimal_tier",
            "cascade_success", "cascade_cost", "cascade_tiers_tried",
        ])
        for (bench, task_id), tiers in sorted(by_task.items()):
            # Oracle: any tier succeeds -> success; among successes, cheapest wins; else cheapest overall.
            successes = {t: row for t, row in tiers.items() if row["success"] == "True"}
            pool = successes if successes else tiers
            oracle_tier = min(pool, key=lambda t: float(pool[t]["cost_usd"]))
            oracle_row = tiers[oracle_tier]

            # Pessimal: priciest failure if any tier fails, else cheapest success.
            failures = {t: row for t, row in tiers.items() if row["success"] == "False"}
            if failures:
                pessimal_tier = max(failures, key=lambda t: float(failures[t]["cost_usd"]))
            else:
                pessimal_tier = min(tiers, key=lambda t: float(tiers[t]["cost_usd"]))
            pessimal_row = tiers[pessimal_tier]

            # Cheap-first idealized cascade: escalate through TIER_ORDER until success or exhausted.
            cascade_cost = 0.0
            cascade_success = False
            tried = []
            for tier in TIER_ORDER:
                if tier not in tiers:
                    continue
                tried.append(tier)
                cascade_cost += float(tiers[tier]["cost_usd"])
                if tiers[tier]["success"] == "True":
                    cascade_success = True
                    break

            w.writerow([
                bench, task_id,
                oracle_row["success"], oracle_row["cost_usd"], oracle_tier,
                pessimal_row["success"], pessimal_row["cost_usd"], pessimal_tier,
                cascade_success, f"{cascade_cost:.6f}", "+".join(tried),
            ])
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main_canonical()

