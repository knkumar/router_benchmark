#!/usr/bin/env python3
"""Phase-1 view.pdf-style narrative metrics computed directly from a locked
canonical bundle (no new API calls). Reproduces the formulas of the older
per-lineage scripts (oracle_and_cascade.py, realistic_cascade.py,
regret_to_oracle.py, mixture_utility.py, rank_consistency_3suite.py,
rank_consistency_4bench.py, candidate_distribution.py) against the
canonical bundle's exhaustive candidate matrix (every tier run on every
task x replicate) and its joined route outcomes.

Emitted CSVs (into --output-dir) plus a phase1_summary.json:
  oracle_cascade.csv, regret_to_oracle.csv, mixture_utility.csv,
  rank_consistency_3suite.csv, rank_consistency_4bench.csv,
  selected_candidate_distribution.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

TIER_ORDER = ["cheap-small", "mid-general", "strong-frontier"]
SHORT = {
    "RouterBench (live)": "RouterBench",
    "BFCL v4 (live)": "BFCL",
    "tau2-bench (live)": "tau2-bench",
    "WebArena (live)": "WebArena",
}
ROUTER_SHORT = {
    "litellm-router-live": "LiteLLM Router",
    "aurelio-semantic-router-live": "Aurelio Semantic Router",
    "routellm-live": "RouteLLM",
    "vllm-semantic-router-live": "vLLM Semantic Router",
}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    bundle, out = args.bundle, args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    cand = _read(bundle / "candidate_outcomes.csv")
    routes = _read(bundle / "routes.csv")
    joins = _read(bundle / "results.csv")

    # candidate lookup by key "benchmark|task|tier|replicate"
    cand_by_key = {
        f"{r['benchmark_id']}|{r['task_id']}|{r['candidate_id']}|{r['outcome_replicate']}": r
        for r in cand
    }
    # per (benchmark, task, tier): list of (success bool, cost) over replicates
    tier_reps: dict[tuple[str, str, str], list[tuple[bool, float]]] = defaultdict(list)
    for r in cand:
        tier_reps[(r["benchmark_id"], r["task_id"], r["candidate_id"])].append(
            (r["success"].lower() == "true", float(r["model_api_cost_usd"]))
        )

    benches = sorted({r["benchmark_id"] for r in cand})
    tasks_by_bench = defaultdict(set)
    for r in cand:
        tasks_by_bench[r["benchmark_id"]].add(r["task_id"])

    def tier_mean(bench, task, tier):
        reps = tier_reps.get((bench, task, tier), [])
        if not reps:
            return None
        s = sum(1 for ok, _ in reps if ok) / len(reps)
        c = sum(c for _, c in reps) / len(reps)
        return s, c

    # ---- Oracle + idealized/realistic cascade (per benchmark) ----
    oracle_rows = {}  # (bench, task) -> dict
    oracle_bench = {}
    ideal_bench = {}
    realistic_bench = {}
    for bench in benches:
        o_succ, o_cost = [], []
        ci_succ, ci_cost = [], []
        rc_succ, rc_cost = [], []
        for task in sorted(tasks_by_bench[bench]):
            tiers = {t: tier_mean(bench, task, t) for t in TIER_ORDER if tier_mean(bench, task, t)}
            # Oracle: tier with max success, tiebreak lowest cost
            best = max(tiers.items(), key=lambda kv: (kv[1][0], -kv[1][1]))
            oracle_rows[(bench, task)] = {"oracle_success": best[1][0], "oracle_cost": best[1][1]}
            o_succ.append(best[1][0])
            o_cost.append(best[1][1])
            # Idealized cascade over per-replicate binary success
            n_reps = max(len(tier_reps[(bench, task, t)]) for t in TIER_ORDER)
            for rep in range(n_reps):
                cost = 0.0
                solved = False
                for t in TIER_ORDER:
                    reps = tier_reps[(bench, task, t)]
                    if rep >= len(reps):
                        continue
                    ok, c = reps[rep]
                    cost += c
                    if ok:
                        solved = True
                        break
                ci_succ.append(1.0 if solved else 0.0)
                ci_cost.append(cost)
                # Realistic cascade: escalate only on infra-failure signal cost==0.
                cost_r = 0.0
                solved_r = None
                for t in TIER_ORDER:
                    reps = tier_reps[(bench, task, t)]
                    if rep >= len(reps):
                        continue
                    ok, c = reps[rep]
                    cost_r += c
                    if c != 0.0:  # billed -> no pre-answer escalation signal, accept
                        solved_r = ok
                        break
                if solved_r is None:
                    solved_r = reps[-1][0]
                rc_succ.append(1.0 if solved_r else 0.0)
                rc_cost.append(cost_r)
        oracle_bench[bench] = (sum(o_succ) / len(o_succ), sum(o_cost) / len(o_cost))
        ideal_bench[bench] = (sum(ci_succ) / len(ci_succ), sum(ci_cost) / len(ci_cost))
        realistic_bench[bench] = (sum(rc_succ) / len(rc_succ), sum(rc_cost) / len(rc_cost))

    with (out / "oracle_cascade.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["benchmark", "oracle_success", "oracle_cost",
                    "idealized_cascade_success", "idealized_cascade_cost",
                    "realistic_cascade_success", "realistic_cascade_cost"])
        for bench in benches:
            w.writerow([SHORT[bench],
                        f"{oracle_bench[bench][0]:.4f}", f"{oracle_bench[bench][1]:.6f}",
                        f"{ideal_bench[bench][0]:.4f}", f"{ideal_bench[bench][1]:.6f}",
                        f"{realistic_bench[bench][0]:.4f}", f"{realistic_bench[bench][1]:.6f}"])

    # ---- Per-router per-benchmark joined success/cost (via results.csv) ----
    route_service = {
        (r["router_config_id"], r["benchmark_id"], r["task_id"], r["routing_seed"]): float(r["router_service_usd"])
        for r in routes
    }
    rj: dict[tuple[str, str], list[tuple[bool, float]]] = defaultdict(list)
    rj_task: dict[tuple[str, str, str], list[tuple[bool, float]]] = defaultdict(list)
    for j in joins:
        c = cand_by_key[j["candidate_outcome_key"]]
        ok = c["success"].lower() == "true"
        cost = float(c["model_api_cost_usd"])
        rj[(j["router_config_id"], j["benchmark_id"])].append((ok, cost))
        rj_task[(j["router_config_id"], j["benchmark_id"], j["task_id"])].append((ok, cost))

    router_success = defaultdict(dict)  # bench -> router -> success
    router_cost = defaultdict(dict)
    for (rid, bench), vals in rj.items():
        router_success[bench][rid] = sum(1 for ok, _ in vals if ok) / len(vals)
        router_cost[bench][rid] = sum(c for _, c in vals) / len(vals)

    # ---- Regret to oracle ----
    group_success_costs = defaultdict(list)
    for j in joins:
        c = cand_by_key[j["candidate_outcome_key"]]
        if c["success"].lower() == "true":
            group_success_costs[j["benchmark_id"]].append(float(c["model_api_cost_usd"]))
    mean_success_cost = {g: sum(v) / len(v) for g, v in group_success_costs.items() if v}

    regret = defaultdict(list)
    for (rid, bench, task), vals in rj_task.items():
        r_succ = sum(1 for ok, _ in vals if ok) / len(vals)
        r_cost = sum(c for _, c in vals) / len(vals)
        o = oracle_rows[(bench, task)]
        router_solved = r_succ >= 0.5
        oracle_solved = o["oracle_success"] >= 0.5
        if router_solved and oracle_solved:
            reg = max(0.0, r_cost - o["oracle_cost"])
        elif (not router_solved) and oracle_solved:
            reg = o["oracle_cost"] + mean_success_cost.get(bench, 0.0)
        else:
            reg = 0.0
        regret[(rid, bench)].append(reg)

    with (out / "regret_to_oracle.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["router", "benchmark", "mean_regret_usd", "n_tasks"])
        for (rid, bench), vals in sorted(regret.items()):
            w.writerow([ROUTER_SHORT[rid], SHORT[bench], f"{sum(vals)/len(vals):.6f}", len(vals)])

    # ---- Mixtures (expected utility) ----
    MIXTURES = {
        "100pct_RouterBench": {"RouterBench (live)": 1.0},
        "50_25_25_RB_BFCL_tau2": {"RouterBench (live)": 0.5, "BFCL v4 (live)": 0.25, "tau2-bench (live)": 0.25},
        "uniform": {b: 0.25 for b in benches},
        "WebArena_heavy": {"RouterBench (live)": 0.1, "BFCL v4 (live)": 0.1, "tau2-bench (live)": 0.1, "WebArena (live)": 0.7},
    }
    mean_cost_g = {b: sum(router_cost[b].values()) / len(router_cost[b]) for b in benches}
    inv = {b: 1.0 / mean_cost_g[b] for b in benches}
    tot = sum(inv.values())
    MIXTURES["cost_constrained"] = {b: inv[b] / tot for b in benches}
    routers = sorted(router_success[benches[0]].keys())
    with (out / "mixture_utility.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["router", "mixture", "expected_success", "expected_cost_per_task"])
        for rid in routers:
            for mix, weights in MIXTURES.items():
                es = sum(wt * router_success[b][rid] for b, wt in weights.items())
                ec = sum(wt * router_cost[b][rid] for b, wt in weights.items())
                w.writerow([ROUTER_SHORT[rid], mix, f"{es:.4f}", f"{ec:.6f}"])

    # ---- Rank consistency (3-suite and 4-suite) ----
    def rank_table(suites: dict[str, dict[str, float]], path: Path):
        groups = list(suites.keys())
        rk = defaultdict(dict)
        for g in groups:
            ordered = sorted(suites[g].items(), key=lambda kv: -kv[1])
            i = 0
            while i < len(ordered):
                j = i
                while j + 1 < len(ordered) and ordered[j + 1][1] == ordered[i][1]:
                    j += 1
                avg = (i + 1 + j + 1) / 2
                for k in range(i, j + 1):
                    rk[ordered[k][0]][g] = avg
                i = j + 1
        rows_out = []
        with path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["router"] + [SHORT.get(g, g) for g in groups] + ["mean_rank", "rank_variance"])
            for rid in routers:
                rv = [rk[rid][g] for g in groups]
                mr = sum(rv) / len(rv)
                var = sum((x - mr) ** 2 for x in rv) / len(rv)
                w.writerow([ROUTER_SHORT[rid]] + [f"{v:.2f}" for v in rv] + [f"{mr:.2f}", f"{var:.3f}"])
                rows_out.append({"router": ROUTER_SHORT[rid], "mean_rank": mr, "rank_variance": var})
        return rows_out

    suites4 = {b: router_success[b] for b in benches}
    suites3 = {
        "RouterBench+BFCL": {rid: (router_success["RouterBench (live)"][rid] + router_success["BFCL v4 (live)"][rid]) / 2 for rid in routers},
        "tau2-bench (live)": router_success["tau2-bench (live)"],
        "WebArena (live)": router_success["WebArena (live)"],
    }
    rank4 = rank_table(suites4, out / "rank_consistency_4bench.csv")
    rank3 = rank_table(suites3, out / "rank_consistency_3suite.csv")

    # ---- Selected-candidate distribution (share + conf + fallback) ----
    dist = defaultdict(lambda: {"cheap-small": 0, "mid-general": 0, "strong-frontier": 0,
                                "n": 0, "conf": 0.0, "fallback": 0})
    for r in routes:
        d = dist[(r["router_config_id"], r["benchmark_id"])]
        d[r["selected_candidate"]] += 1
        d["n"] += 1
        d["conf"] += float(r["confidence"])
        if r["fallback_path"] != "none":
            d["fallback"] += 1
    with (out / "selected_candidate_distribution.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["router", "benchmark", "cheap", "mid", "strong", "mean_conf", "fallback_rate"])
        for (rid, bench), d in sorted(dist.items()):
            n = d["n"]
            w.writerow([ROUTER_SHORT[rid], SHORT[bench],
                        f"{d['cheap-small']/n:.3f}", f"{d['mid-general']/n:.3f}", f"{d['strong-frontier']/n:.3f}",
                        f"{d['conf']/n:.3f}", f"{d['fallback']/n:.3f}"])

    # ---- JSON summary ----
    summary = {
        "oracle": {SHORT[b]: {"success": round(oracle_bench[b][0], 4), "cost": round(oracle_bench[b][1], 6)} for b in benches},
        "idealized_cascade": {SHORT[b]: {"success": round(ideal_bench[b][0], 4), "cost": round(ideal_bench[b][1], 6)} for b in benches},
        "realistic_cascade": {SHORT[b]: {"success": round(realistic_bench[b][0], 4), "cost": round(realistic_bench[b][1], 6)} for b in benches},
        "router_success": {SHORT[b]: {ROUTER_SHORT[rid]: round(v, 4) for rid, v in router_success[b].items()} for b in benches},
        "rank_3suite": rank3,
        "rank_4bench": rank4,
    }
    (out / "phase1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
