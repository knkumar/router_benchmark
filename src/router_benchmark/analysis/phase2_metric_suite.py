#!/usr/bin/env python3
"""Phase-2 deployment metric suite + candidate-pool ablation, computed directly
from locked bundles (no new API calls).

Part A -- 12-metric deployment suite from the canonical 4-benchmark bundle, one
row per router (pooled) plus a per-benchmark CSV. Metrics: success rate,
cost/task, cost/success, latency p50/p95, tool-call accuracy, fallback rate,
route stability (agreement of the selected tier across routing seeds), mean
confidence, easy/medium/hard success (difficulty = number of tiers that solve
the task, a disclosed candidate-agreement proxy), and robustness std (std of
success across the three outcome replicates of the selected tier).

Part B -- narrow-gap vs wide-gap candidate-pool ablation, read straight from the
two older-lineage 2-benchmark ablation bundles (pool2_widegap_v1,
ablation_narrowgap_v1), which already ship the same metric suite. Demonstrates
that static routers are pool-invariant in selection (so their cost tracks the
exogenous price of the tier they are pinned to) while vLLM is pool-sensitive.

Emitted into --output-dir:
  canonical_metric_suite.csv, canonical_metric_suite_per_benchmark.csv,
  pool_ablation_comparison.csv, phase2_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
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
# Order rows so vLLM (the only content-dependent router) reads last.
ROUTER_ROW_ORDER = [
    "aurelio-semantic-router-live",
    "litellm-router-live",
    "routellm-live",
    "vllm-semantic-router-live",
]
# Tool-call correctness is only defined for the tool-use benchmarks.
TOOL_BENCHMARKS = {"BFCL v4 (live)", "tau2-bench (live)"}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _pct(values: list[float], q: float) -> float | None:
    vals = sorted(v for v in values if v is not None and not math.isnan(v))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def canonical_suite(bundle: Path, out: Path) -> dict:
    cand = _read(bundle / "candidate_outcomes.csv")
    routes = _read(bundle / "routes.csv")
    joins = _read(bundle / "results.csv")

    # candidate lookup by joined key "benchmark|task|tier|replicate"
    cand_by_key = {
        f"{r['benchmark_id']}|{r['task_id']}|{r['candidate_id']}|{r['outcome_replicate']}": r
        for r in cand
    }
    # per (benchmark, task, tier): replicate success list -> difficulty + robustness
    tier_reps: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
    for r in cand:
        tier_reps[(r["benchmark_id"], r["task_id"], r["candidate_id"])].append(
            r["success"].lower() == "true"
        )

    benches = sorted({r["benchmark_id"] for r in cand})
    tasks_by_bench = defaultdict(set)
    for r in cand:
        tasks_by_bench[r["benchmark_id"]].add(r["task_id"])

    # Difficulty band per (benchmark, task): number of tiers that solve it
    # (mean replicate success >= 0.5). 3 -> easy, 2 -> medium, <=1 -> hard.
    difficulty: dict[tuple[str, str], str] = {}
    for bench in benches:
        for task in tasks_by_bench[bench]:
            n_solve = 0
            for t in TIER_ORDER:
                reps = tier_reps.get((bench, task, t))
                if reps and (sum(reps) / len(reps)) >= 0.5:
                    n_solve += 1
            difficulty[(bench, task)] = (
                "easy" if n_solve >= 3 else "medium" if n_solve == 2 else "hard"
            )

    # tool_call_correct per joined candidate key, parsed from traces.jsonl
    tool_correct: dict[str, bool | None] = {}
    with (bundle / "traces.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            rt = json.loads(line).get("raw_trace") or {}
            ar = rt.get("adapter_result") or {}
            if not isinstance(rt, dict) or not isinstance(ar, dict):
                continue
            key = f"{rt.get('benchmark_id')}|{rt.get('task_id')}|{rt.get('candidate_id')}|{rt.get('outcome_replicate')}"
            tool_correct[key] = ar.get("tool_call_correct")

    # Selected tier per (router, benchmark, task, seed) for route stability
    sel: dict[tuple[str, str, str], dict[str, str]] = defaultdict(dict)
    conf_sum: dict[tuple[str, str], float] = defaultdict(float)
    conf_n: dict[tuple[str, str], int] = defaultdict(int)
    fb_hit: dict[tuple[str, str], int] = defaultdict(int)
    fb_n: dict[tuple[str, str], int] = defaultdict(int)
    for r in routes:
        rid, bench, task = r["router_config_id"], r["benchmark_id"], r["task_id"]
        sel[(rid, bench)].setdefault(task, {})[r["routing_seed"]] = r["selected_candidate"]
        conf_sum[(rid, bench)] += float(r["confidence"])
        conf_n[(rid, bench)] += 1
        fb_n[(rid, bench)] += 1
        if r["fallback_path"] != "none":
            fb_hit[(rid, bench)] += 1

    def route_stability(rid: str, bench: str) -> float:
        tasks = sel[(rid, bench)]
        if not tasks:
            return float("nan")
        agree = sum(1 for seeds in tasks.values() if len(set(seeds.values())) == 1)
        return agree / len(tasks)

    # Joined per-router outcomes (success/cost/latency/tool/difficulty/robustness)
    per_rb: dict[tuple[str, str], dict] = defaultdict(
        lambda: {
            "n": 0, "succ": 0, "cost": 0.0,
            "lat": [], "tool_ok": 0, "tool_n": 0,
            "band": defaultdict(lambda: [0, 0]), "rob": [],
        }
    )
    for j in joins:
        rid, bench, task = j["router_config_id"], j["benchmark_id"], j["task_id"]
        key = j["candidate_outcome_key"]
        c = cand_by_key[key]
        ok = c["success"].lower() == "true"
        d = per_rb[(rid, bench)]
        d["n"] += 1
        d["succ"] += 1 if ok else 0
        d["cost"] += float(c["model_api_cost_usd"])
        lat = c["generation_latency_ms"]
        if lat not in ("", "NaN") and not math.isnan(float(lat)):
            d["lat"].append(float(lat))
        tc = tool_correct.get(key)
        if bench in TOOL_BENCHMARKS and tc is not None:
            d["tool_n"] += 1
            d["tool_ok"] += 1 if tc else 0
        band = difficulty[(bench, task)]
        d["band"][band][1] += 1
        d["band"][band][0] += 1 if ok else 0
        tier = key.split("|")[2]
        reps = tier_reps[(bench, task, tier)]
        d["rob"].append(statistics.pstdev([1.0 if x else 0.0 for x in reps]) if len(reps) > 1 else 0.0)

    def band_rate(d, band):
        s, n = d["band"][band]
        return s / n if n else float("nan")

    # ---- Per-benchmark CSV ----
    with (out / "canonical_metric_suite_per_benchmark.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["router", "benchmark", "n_tasks", "success_rate", "cost_per_task_usd",
                    "cost_per_success_usd", "latency_p50_ms", "latency_p95_ms",
                    "tool_call_accuracy", "fallback_rate", "route_stability",
                    "mean_confidence", "easy_success", "medium_success", "hard_success",
                    "robustness_std"])
        for rid in ROUTER_ROW_ORDER:
            for bench in benches:
                d = per_rb[(rid, bench)]
                if not d["n"]:
                    continue
                succ = d["succ"] / d["n"]
                cpt = d["cost"] / d["n"]
                cps = d["cost"] / d["succ"] if d["succ"] else float("nan")
                p50, p95 = _pct(d["lat"], 0.5), _pct(d["lat"], 0.95)
                tool = d["tool_ok"] / d["tool_n"] if d["tool_n"] else float("nan")
                w.writerow([
                    ROUTER_SHORT[rid], SHORT[bench], d["n"],
                    f"{succ:.4f}", f"{cpt:.6f}",
                    f"{cps:.6f}" if not math.isnan(cps) else "NaN",
                    f"{p50:.1f}" if p50 is not None else "NaN",
                    f"{p95:.1f}" if p95 is not None else "NaN",
                    f"{tool:.4f}" if not math.isnan(tool) else "NaN",
                    f"{fb_hit[(rid, bench)] / fb_n[(rid, bench)]:.4f}",
                    f"{route_stability(rid, bench):.4f}",
                    f"{conf_sum[(rid, bench)] / conf_n[(rid, bench)]:.4f}",
                    f"{band_rate(d, 'easy'):.4f}", f"{band_rate(d, 'medium'):.4f}",
                    f"{band_rate(d, 'hard'):.4f}", f"{statistics.mean(d['rob']):.4f}",
                ])

    # ---- Pooled (macro-average across benchmarks) CSV + summary ----
    summary_rows = []
    with (out / "canonical_metric_suite.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["router", "success_rate", "cost_per_task_usd", "cost_per_success_usd",
                    "latency_p50_ms", "latency_p95_ms", "tool_call_accuracy",
                    "fallback_rate", "route_stability", "mean_confidence",
                    "easy_success", "medium_success", "hard_success", "robustness_std"])
        for rid in ROUTER_ROW_ORDER:
            def macro(fn):
                vals = [fn(per_rb[(rid, b)]) for b in benches if per_rb[(rid, b)]["n"]]
                vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
                return statistics.mean(vals) if vals else float("nan")

            succ = macro(lambda d: d["succ"] / d["n"])
            cpt = macro(lambda d: d["cost"] / d["n"])
            cps = macro(lambda d: (d["cost"] / d["succ"]) if d["succ"] else None)
            p50 = macro(lambda d: _pct(d["lat"], 0.5))
            p95 = macro(lambda d: _pct(d["lat"], 0.95))
            tool = macro(lambda d: (d["tool_ok"] / d["tool_n"]) if d["tool_n"] else None)
            fb = statistics.mean([fb_hit[(rid, b)] / fb_n[(rid, b)] for b in benches])
            stab = statistics.mean([route_stability(rid, b) for b in benches])
            conf = statistics.mean([conf_sum[(rid, b)] / conf_n[(rid, b)] for b in benches])
            easy = macro(lambda d: band_rate(d, "easy"))
            med = macro(lambda d: band_rate(d, "medium"))
            hard = macro(lambda d: band_rate(d, "hard"))
            rob = macro(lambda d: statistics.mean(d["rob"]))
            row = [ROUTER_SHORT[rid], f"{succ:.4f}", f"{cpt:.6f}", f"{cps:.6f}",
                   f"{p50:.1f}", f"{p95:.1f}", f"{tool:.4f}", f"{fb:.4f}",
                   f"{stab:.4f}", f"{conf:.4f}", f"{easy:.4f}", f"{med:.4f}",
                   f"{hard:.4f}", f"{rob:.4f}"]
            w.writerow(row)
            summary_rows.append({
                "router": ROUTER_SHORT[rid], "success_rate": round(succ, 4),
                "cost_per_task_usd": round(cpt, 6), "tool_call_accuracy": round(tool, 4),
                "fallback_rate": round(fb, 4), "route_stability": round(stab, 4),
                "mean_confidence": round(conf, 4), "robustness_std": round(rob, 4),
                "easy_success": round(easy, 4), "medium_success": round(med, 4),
                "hard_success": round(hard, 4),
            })
    return {"benchmarks": [SHORT[b] for b in benches], "metric_suite": summary_rows}


def pool_ablation(widegap: Path, narrowgap: Path, out: Path) -> dict:
    def load_overall(bundle: Path) -> dict[str, dict[str, str]]:
        return {r["router_name"]: r for r in _read(bundle / "metrics_overall.csv")}

    wide = load_overall(widegap)
    narrow = load_overall(narrowgap)
    wm = json.loads((widegap / "manifest.json").read_text())
    nm = json.loads((narrowgap / "manifest.json").read_text())

    rows = []
    routers = ["Aurelio Semantic Router (live)", "LiteLLM Router (live)",
               "RouteLLM (live)", "vLLM Semantic Router (live)"]
    with (out / "pool_ablation_comparison.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["router", "widegap_success", "narrowgap_success", "success_delta",
                    "widegap_cost_per_task", "narrowgap_cost_per_task", "cost_ratio_narrow_over_wide",
                    "widegap_stability", "narrowgap_stability"])
        for r in routers:
            ws, ns = float(wide[r]["mean_success_rate"]), float(narrow[r]["mean_success_rate"])
            wc, nc = float(wide[r]["mean_cost_per_task_usd"]), float(narrow[r]["mean_cost_per_task_usd"])
            wst, nst = float(wide[r]["mean_route_stability"]), float(narrow[r]["mean_route_stability"])
            ratio = nc / wc if wc else float("nan")
            short = r.replace(" (live)", "")
            w.writerow([short, f"{ws:.4f}", f"{ns:.4f}", f"{ns - ws:+.4f}",
                        f"{wc:.6f}", f"{nc:.6f}", f"{ratio:.2f}",
                        f"{wst:.4f}", f"{nst:.4f}"])
            rows.append({"router": short, "success_delta": round(ns - ws, 4),
                         "cost_ratio_narrow_over_wide": round(ratio, 2)})
    return {
        "widegap_cheap_tier": [k for k in wm["candidate_pricing_usd_per_token"] if k not in
                               ("claude-sonnet-4-6", "claude-opus-4-8")],
        "narrowgap_cheap_tier": [k for k in nm["candidate_pricing_usd_per_token"] if k not in
                                 ("claude-sonnet-4-6", "claude-opus-4-8", "gpt-5.4-nano")],
        "benchmarks": [SHORT.get(b, b) for b in wm["benchmarks"]],
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--widegap-bundle", type=Path, required=True)
    ap.add_argument("--narrowgap-bundle", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    suite = canonical_suite(args.bundle, args.output_dir)
    ablation = pool_ablation(args.widegap_bundle, args.narrowgap_bundle, args.output_dir)
    summary = {"metric_suite": suite, "pool_ablation": ablation}
    (args.output_dir / "phase2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

