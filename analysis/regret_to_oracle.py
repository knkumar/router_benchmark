#!/usr/bin/env python3
"""Regret-to-oracle: for each real router, how much worse (in cost, at
matched or better success) is it than the oracle upper bound on the same
tasks. Regret per task: both succeed -> router_cost - oracle_cost (>=0);
router fails, oracle succeeds -> oracle_cost + benchmark-group mean
successful-task cost (a fixed failure penalty, since a failed task has no
well-defined "cost of the miss" otherwise); router succeeds, oracle
doesn't (should not happen by construction, guarded anyway) -> 0."""
import csv
from collections import defaultdict
from pathlib import Path

from oracle_and_cascade import (
    build_canonical_oracle,
    build_canonical_per_task_tiers,
)

# --- canonical single-lineage regret --------------------------------------
# Legacy main() below reads output/results.csv + analysis oracle_and_cascade.csv
# (Task-1 sweep lineage). The reviewer flagged that second lineage as a
# provenance liability, so main_canonical() rebuilds regret entirely from the
# CANONICAL bundle for all seven policies (four routers + three fixed tiers).
#
# Oracle and every policy are collapsed per task to a MEAN success rate and a
# MEAN candidate cost over their replicates (routers additionally over routing
# seeds). Because the oracle picks the best tier, its per-task mean success
# dominates any single policy's, so the "policy succeeds while oracle fails"
# branch cannot fire. The original boolean regret formula generalizes to
# fractional per-task success s (reducing to the original when s in {0,1}):
#
#   regret_task = s_p * max(0, c_p - c_o)                       # both succeed
#               + (s_o - s_p) * (c_o + group_mean_success_cost) # policy miss
#               + (1 - s_o) * 0                                 # both fail
#   norm_regret_task = regret_task / max(c_o, 1e-6)
#
# group_mean_success_cost is the fixed per-group failure penalty: the mean
# candidate model_api_cost over all successful candidate outcomes in the group
# (single lineage: candidate_outcomes.csv). Regret uses candidate-only cost
# (no router-service fee), matching the legacy definition.
CANON_BUNDLE_DIR = (
    Path(__file__).parent / "../output/live/paper1_canonical_webarena_repair_v2"
)
CANDIDATE_OUTCOMES = CANON_BUNDLE_DIR / "candidate_outcomes.csv"
ROUTER_RESULTS = CANON_BUNDLE_DIR / "results.csv"

ROUTER_NAME = {
    "aurelio-semantic-router-live": "Aurelio Semantic Router",
    "vllm-semantic-router-live": "vLLM Semantic Router",
    "litellm-router-live": "LiteLLM Router",
    "routellm-live": "RouteLLM",
}
POLICY_ORDER = [
    "Aurelio Semantic Router",
    "vLLM Semantic Router",
    "LiteLLM Router",
    "RouteLLM",
    "Always-Cheapest",
    "Always-Mid",
    "Always-Strongest",
]
TIER_POLICY = {
    "Always-Cheapest": "cheap-small",
    "Always-Mid": "mid-general",
    "Always-Strongest": "strong-frontier",
}


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _group_mean_success_cost(bundle=CANDIDATE_OUTCOMES):
    costs = defaultdict(list)
    with open(bundle) as f:
        for r in csv.DictReader(f):
            if r["success"].strip().lower() == "true":
                costs[group_key(r["benchmark_id"])].append(float(r["model_api_cost_usd"]))
    return {g: _mean(v) for g, v in costs.items()}


def _router_per_task(bundle=CANDIDATE_OUTCOMES, results=ROUTER_RESULTS):
    """{policy: {(benchmark_id, task_id): (mean_success, mean_cost)}} by joining
    the router results' candidate_outcome_key back to candidate_outcomes."""
    lookup = {}
    with open(bundle) as f:
        for r in csv.DictReader(f):
            key = f'{r["benchmark_id"]}|{r["task_id"]}|{r["candidate_id"]}|{r["outcome_replicate"]}'
            lookup[key] = (r["success"].strip().lower() == "true",
                           float(r["model_api_cost_usd"]))
    agg = defaultdict(lambda: defaultdict(list))
    with open(results) as f:
        for r in csv.DictReader(f):
            pol = ROUTER_NAME[r["router_config_id"]]
            s, c = lookup[r["candidate_outcome_key"]]
            agg[pol][(r["benchmark_id"], r["task_id"])].append((1.0 if s else 0.0, c))
    out = {}
    for pol, per_task in agg.items():
        out[pol] = {k: (_mean([s for s, _ in v]), _mean([c for _, c in v]))
                    for k, v in per_task.items()}
    return out


def _norm_regret(s_p, c_p, s_o, c_o, penalty):
    both = s_p * max(0.0, c_p - c_o)
    miss = max(0.0, s_o - s_p) * (c_o + penalty)
    return (both + miss) / max(c_o, 1e-6)


def main_canonical():
    oracle = build_canonical_oracle()          # (bench, task) -> (s, cost, tier)
    tiers = build_canonical_per_task_tiers()    # (bench, task) -> {tier: (s, cost)}
    penalty = _group_mean_success_cost()
    routers = _router_per_task()

    # policy -> {(bench, task): (s_p, c_p)}
    policy_task = dict(routers)
    for pol, tier in TIER_POLICY.items():
        policy_task[pol] = {k: v[tier] for k, v in tiers.items() if tier in v}

    regrets = defaultdict(list)
    for pol in policy_task:
        for key, (s_p, c_p) in policy_task[pol].items():
            if key not in oracle:
                continue
            s_o, c_o, _ = oracle[key]
            g = group_key(key[0])
            regrets[(pol, g)].append(_norm_regret(s_p, c_p, s_o, c_o, penalty.get(g, 0.0)))

    out_path = Path(__file__).parent / "output" / "paper1_canonical" / "regret_to_oracle_canonical.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    order = {p: i for i, p in enumerate(POLICY_ORDER)}
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "benchmark_group", "mean_norm_regret",
                    "worst_norm_regret", "n_tasks"])
        for (pol, g), vals in sorted(regrets.items(),
                                     key=lambda kv: (order.get(kv[0][0], 99), kv[0][1])):
            w.writerow([pol, g, f"{_mean(vals):.6f}", f"{max(vals):.6f}", len(vals)])
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
    oracle_path = Path(__file__).parent / "output" / "oracle_and_cascade.csv"
    with open(oracle_path) as f:
        oracle = {(r["benchmark_name"], r["task_id"]): r for r in csv.DictReader(f)}

    results_path = Path(__file__).parent / "../output/results.csv"
    with open(results_path) as f:
        results = list(csv.DictReader(f))

    group_success_costs = defaultdict(list)
    for r in results:
        if r["success"] == "True":
            group_success_costs[group_key(r["benchmark_name"])].append(float(r["cost_usd"]))
    mean_success_cost = {g: sum(v) / len(v) for g, v in group_success_costs.items() if v}

    regrets = defaultdict(list)
    for r in results:
        key = (r["benchmark_name"], r["task_id"])
        if key not in oracle:
            continue
        o = oracle[key]
        group = group_key(r["benchmark_name"])
        router_success = r["success"] == "True"
        oracle_success = o["oracle_success"] == "True"
        if router_success and oracle_success:
            regret = max(0.0, float(r["cost_usd"]) - float(o["oracle_cost"]))
        elif (not router_success) and oracle_success:
            regret = float(o["oracle_cost"]) + mean_success_cost.get(group, 0.0)
        else:
            regret = 0.0

        norm_regret = regret / max(float(o["oracle_cost"]), 1e-6)
        regrets[(r["router_name"], group)].append(norm_regret)

    out_path = Path(__file__).parent / "output" / "regret_to_oracle.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["router", "benchmark_group", "mean_norm_regret", "worst_norm_regret", "n_tasks"])
        for (router, group), vals in sorted(regrets.items()):
            mean_regret = sum(vals)/len(vals)
            worst_regret = max(vals)
            w.writerow([router, group, f"{mean_regret:.6f}", f"{worst_regret:.6f}", len(vals)])
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main_canonical()
