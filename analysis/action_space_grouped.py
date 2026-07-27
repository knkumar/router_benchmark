#!/usr/bin/env python3
"""Grouped cheap-vs-strong (binary) action-space view for RouteLLM and LiteLLM.

Reviewer fairness objection: RouteLLM and LiteLLM have a NATIVE BINARY action
space (weak/cheap endpoint vs strong endpoint); they cannot pick the mid tier by
construction. Reporting them on the 3-tier axis arguably penalises them for not
picking mid. This script collapses the 3-tier selection into a binary
{cheap, strong} view for these two routers only (mid-general, if it ever appears,
is grouped with strong as "escalated") and shows the story is unchanged: they pick
cheap ~100% on every benchmark, so the binary regrouping does not rescue them.

Selection shares come from routes.csv (one row per route = router x benchmark x
task x routing_seed). Success under the realised binary policy is computed over
the canonical route->outcome join in results.csv (candidate_outcome_key ->
candidate_outcomes.csv), i.e. the same join the canonical metric suite uses.

Writes analysis/output/paper1_canonical/action_space_grouped.csv with columns:
    router, benchmark, share_cheap, share_strong, success_rate, n_routes
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

BUNDLE = Path("output/live/paper1_canonical_webarena_repair_v2")
OUT = Path("analysis/output/paper1_canonical/action_space_grouped.csv")

# Native binary routers only.
ROUTER_NAME = {
    "routellm-live": "RouteLLM",
    "litellm-router-live": "LiteLLM Router",
}
# cheap-small -> cheap; everything else (mid-general, strong-frontier) -> strong/escalated.
CHEAP_TIER = "cheap-small"


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    routes = _read(BUNDLE / "routes.csv")
    results = _read(BUNDLE / "results.csv")
    cand = _read(BUNDLE / "candidate_outcomes.csv")
    cand_by_key = {
        f"{c['benchmark_id']}|{c['task_id']}|{c['candidate_id']}|{c['outcome_replicate']}": c
        for c in cand
    }

    # Selection shares (binary) + route count from routes.csv.
    sel: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"cheap": 0, "strong": 0, "n": 0}
    )
    for r in routes:
        rid = r["router_config_id"]
        if rid not in ROUTER_NAME:
            continue
        bench = r["benchmark_id"]
        d = sel[(rid, bench)]
        d["n"] += 1
        if r["selected_candidate"] == CHEAP_TIER:
            d["cheap"] += 1
        else:
            d["strong"] += 1

    # Realised success under the policy from the canonical results.csv join.
    succ: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])  # [ok, total]
    for j in results:
        rid = j["router_config_id"]
        if rid not in ROUTER_NAME:
            continue
        bench = j["benchmark_id"]
        c = cand_by_key[j["candidate_outcome_key"]]
        ok = c["success"].strip().lower() == "true"
        cell = succ[(rid, bench)]
        cell[1] += 1
        cell[0] += 1 if ok else 0

    rows = []
    for (rid, bench), d in sel.items():
        n = d["n"]
        share_cheap = d["cheap"] / n if n else float("nan")
        share_strong = d["strong"] / n if n else float("nan")
        ok, tot = succ[(rid, bench)]
        success_rate = ok / tot if tot else float("nan")
        rows.append(
            {
                "router": ROUTER_NAME[rid],
                "benchmark": bench,
                "share_cheap": round(share_cheap, 4),
                "share_strong": round(share_strong, 4),
                "success_rate": round(success_rate, 4),
                "n_routes": n,
            }
        )

    rows.sort(key=lambda x: (x["router"], x["benchmark"]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["router", "benchmark", "share_cheap", "share_strong",
                        "success_rate", "n_routes"],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {OUT} ({len(rows)} rows)")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
