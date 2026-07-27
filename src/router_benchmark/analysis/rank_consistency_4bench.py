#!/usr/bin/env python3
"""4-benchmark-group cross-benchmark rank consistency (Definition 4),
computed with RouterBench and BFCL as separate suites rather than
combined -- a stricter companion to the paper's existing 3-suite table.
Reads per-group success rates already computed in bootstrap_ci.py's
output (the point estimates), no new data needed."""
import csv
from collections import defaultdict
from pathlib import Path


def main():
    ci_path = Path(__file__).parent / "output" / "bootstrap_ci.csv"
    with open(ci_path) as f:
        rows = [r for r in csv.DictReader(f) if r["metric"] == "success_rate"]

    success = defaultdict(dict)
    for r in rows:
        success[r["benchmark_group"]][r["router"]] = float(r["point_estimate"])

    groups = sorted(success.keys())
    routers = sorted({rt for g in success.values() for rt in g})
    routers = [rt for rt in routers if all(rt in success[g] for g in groups)]

    ranks = defaultdict(dict)
    for g in groups:
        ordered = sorted(success[g].items(), key=lambda kv: -kv[1])
        i = 0
        while i < len(ordered):
            j = i
            while j + 1 < len(ordered) and ordered[j + 1][1] == ordered[i][1]:
                j += 1
            avg_rank = (i + 1 + j + 1) / 2
            for k in range(i, j + 1):
                ranks[ordered[k][0]][g] = avg_rank
            i = j + 1

    out_path = Path(__file__).parent / "output" / "rank_consistency_4bench.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["router"] + groups + ["mean_rank", "rank_variance"])
        for router in routers:
            rvals = [ranks[router][g] for g in groups]
            mean_rank = sum(rvals) / len(rvals)
            variance = sum((r - mean_rank) ** 2 for r in rvals) / len(rvals)
            w.writerow([router] + [f"{v:.2f}" for v in rvals] + [f"{mean_rank:.2f}", f"{variance:.3f}"])
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

