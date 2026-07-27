#!/usr/bin/env python3
"""Extract a WebArena-only candidate view for route replay.

``full_run_routes --include-benchmark`` validates that the staged
``candidate_outcomes.csv`` matches *exactly* the filtered protocol's expected
keys (see ``dry_run_routes._assert_completed_candidate_stage``). The repair-v2
candidate stage holds all 2,610 rows (retained non-WebArena + new WebArena), so
running route replay there with ``--include-benchmark 'WebArena (live)'``
fails that equality check. This script copies just the WebArena rows (and
their matching traces / external-spend lines) into a dedicated stage
directory that route replay can run against.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

WEBARENA = "WebArena (live)"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def prepare_route_stage(*, source_stage: Path, stage_dir: Path) -> None:
    if stage_dir.exists():
        raise ValueError(f"route stage already exists: {stage_dir}")
    required = ("candidate_outcomes.csv", "traces.jsonl", "external_metered_spend.jsonl", "stage_manifest.json")
    missing = [name for name in required if not (source_stage / name).exists()]
    if missing:
        raise ValueError(f"source stage is incomplete: missing {missing}")

    rows = _read_rows(source_stage / "candidate_outcomes.csv")
    webarena_rows = [row for row in rows if row["benchmark_id"] == WEBARENA]
    if not webarena_rows:
        raise ValueError("source stage has no WebArena rows")
    digests = {row["raw_trace_digest"] for row in webarena_rows}
    traces = [
        line for line in (source_stage / "traces.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("digest") in digests
    ]
    if len(traces) != len(digests):
        raise ValueError("source stage is missing a WebArena candidate trace")
    ledger = [
        line for line in (source_stage / "external_metered_spend.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("cell", [None])[0] == WEBARENA
    ]

    stage_dir.mkdir(parents=True)
    with (stage_dir / "candidate_outcomes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(webarena_rows[0]))
        writer.writeheader()
        writer.writerows(webarena_rows)
    (stage_dir / "traces.jsonl").write_text("\n".join(traces) + "\n", encoding="utf-8")
    (stage_dir / "external_metered_spend.jsonl").write_text(
        "\n".join(ledger) + ("\n" if ledger else ""), encoding="utf-8"
    )
    (stage_dir / "stage_manifest.json").write_text(
        json.dumps({"derived_from": str(source_stage), "scope": "WebArena (live) route-replay view"}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-stage", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare_route_stage(source_stage=args.source_stage, stage_dir=args.stage_dir)


if __name__ == "__main__":
    main()

