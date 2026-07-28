#!/usr/bin/env python3
"""3-benchmark-group cross-benchmark rank consistency (Definition 4),
treating RouterBench and BFCL as one combined suite (unweighted mean of
their two success rates, matching Table tab:live-overall / Section
sec:problem's worked example) vs. tau2-bench and WebArena as the other
two suites. Companion to the stricter 4-suite rank_consistency_4bench.py,
which keeps RouterBench and BFCL separate.
Reads per-group success rates already computed in bootstrap_ci.py's
output (the point estimates), no new data needed."""
import csv
from collections import defaultdict
from pathlib import Path


def main():
    ci_path = Path(__file__).parent / "output" / "bootstrap_ci.csv"
    with open(ci_path) as f:
        rows = [r for r in csv.DictReader(f) if r["metric"] == "success_rate"]

    raw = defaultdict(dict)
    for r in rows:
        raw[r["benchmark_group"]][r["router"]] = float(r["point_estimate"])

    routers = sorted({rt for g in raw.values() for rt in g})
    routers = [rt for rt in routers if all(rt in raw[g] for g in ("RouterBench", "BFCL", "tau2-bench", "WebArena"))]

    success = defaultdict(dict)
    for rt in routers:
        success["RouterBench+BFCL"][rt] = (raw["RouterBench"][rt] + raw["BFCL"][rt]) / 2
        success["tau2-bench"][rt] = raw["tau2-bench"][rt]
        success["WebArena"][rt] = raw["WebArena"][rt]

    groups = ["RouterBench+BFCL", "tau2-bench", "WebArena"]

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

    out_path = Path(__file__).parent / "output" / "rank_consistency_3suite.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["router"] + groups + ["mean_rank", "rank_variance"])
        for router in routers:
            rvals = [ranks[router][g] for g in groups]
            mean_rank = sum(rvals) / len(rvals)
            variance = sum((r - mean_rank) ** 2 for r in rvals) / len(rvals)
            w.writerow([router] + [f"{v:.2f}" for v in rvals] + [f"{mean_rank:.2f}", f"{variance:.3f}"])
    print(f"wrote {out_path}")
    for g in groups:
        print(g, {rt: round(v, 3) for rt, v in success[g].items()})


if __name__ == "__main__":
    main()

