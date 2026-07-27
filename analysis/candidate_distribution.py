#!/usr/bin/env python3
"""Per (router, benchmark_group) distribution over selected_candidate,
plus mean confidence and fallback rate. Existing-data only."""
import csv
from collections import defaultdict
from pathlib import Path


def group_key(benchmark_name):
    if "RouterBench" in benchmark_name:
        return "RouterBench"
    if "BFCL" in benchmark_name:
        return "BFCL"
    if "tau2" in benchmark_name:
        return "tau2-bench"
    if "WebArena" in benchmark_name:
        return "WebArena"
    raise ValueError(benchmark_name)


def main():
    src = Path(__file__).parent / "../data/live/paper1_live_v2/results.csv"
    with open(src) as f:
        rows = list(csv.DictReader(f))

    counts = defaultdict(lambda: defaultdict(int))
    confidences = defaultdict(list)
    fallbacks = defaultdict(list)
    totals = defaultdict(int)

    for r in rows:
        key = (r["router_name"], group_key(r["benchmark_name"]))
        totals[key] += 1
        counts[key][r["selected_candidate"]] += 1
        confidences[key].append(float(r["confidence"]))
        fallbacks[key].append(r["fallback_used"] == "True")

    out_path = Path(__file__).parent / "output" / "candidate_distribution.csv"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["router", "benchmark_group", "pct_cheap_small", "pct_mid_general", "pct_strong_frontier", "mean_confidence", "fallback_rate", "n_tasks"])
        for key in sorted(totals.keys()):
            router, group = key
            n = totals[key]
            pct_cheap = counts[key].get("cheap-small", 0) / n
            pct_mid = counts[key].get("mid-general", 0) / n
            pct_strong = counts[key].get("strong-frontier", 0) / n
            mean_conf = sum(confidences[key]) / n
            fb_rate = sum(fallbacks[key]) / n
            w.writerow([router, group, f"{pct_cheap:.3f}", f"{pct_mid:.3f}", f"{pct_strong:.3f}", f"{mean_conf:.3f}", f"{fb_rate:.3f}", n])
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
