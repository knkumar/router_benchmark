#!/usr/bin/env python3
"""Seed a replacement full stage with validated non-WebArena execution rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from router_benchmark.protocol.candidate_runner import CandidateStageRunner
from router_benchmark.protocol.protocol_tools import load_yaml
from router_benchmark.scripts.preflight_full_run import (
    candidate_reservations_from_full_protocol,
    external_metered_reservation_from_full_protocol,
    validate_full_run_protocol,
)


WEBARENA = "WebArena (live)"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def prepare_stage(*, source_stage: Path, protocol: dict, stage_dir: Path) -> None:
    """Copy valid non-WebArena cells, traces, and ledger entries into a v2 stage."""
    validate_full_run_protocol(protocol)
    if stage_dir.exists():
        raise ValueError(f"replacement stage already exists: {stage_dir}")
    required = ("candidate_outcomes.csv", "traces.jsonl", "external_metered_spend.jsonl")
    missing = [name for name in required if not (source_stage / name).exists()]
    if missing:
        raise ValueError(f"source stage is incomplete: missing {missing}")

    retained_rows = [
        row for row in _read_rows(source_stage / "candidate_outcomes.csv")
        if row["benchmark_id"] != WEBARENA
    ]
    if not retained_rows:
        raise ValueError("source stage has no non-WebArena rows to retain")
    retained_digests = {row["raw_trace_digest"] for row in retained_rows}
    retained_traces = [
        line for line in (source_stage / "traces.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("digest") in retained_digests
    ]
    if len(retained_traces) != len(retained_digests):
        raise ValueError("source stage is missing a retained candidate trace")
    retained_ledger = [
        line for line in (source_stage / "external_metered_spend.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("cell", [None])[0] != WEBARENA
    ]

    reservations = candidate_reservations_from_full_protocol(protocol)
    runner = CandidateStageRunner(
        stage_dir,
        protocol,
        budget_cap_usd=float(protocol["full_run_cost_reservations"]["total_cap_usd"]),
        reservation_by_cell=reservations,
        external_reserved_usd=external_metered_reservation_from_full_protocol(protocol),
        stage_metadata={
            "protocol_id": protocol["protocol_id"],
            "reservation_source": "full_run_cost_reservations",
            "diagnostic_only": False,
        },
    )
    stage_dir.mkdir(parents=True)
    runner._write_manifest()
    with (stage_dir / "candidate_outcomes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(retained_rows[0]))
        writer.writeheader()
        writer.writerows(retained_rows)
    (stage_dir / "traces.jsonl").write_text("\n".join(retained_traces) + "\n", encoding="utf-8")
    (stage_dir / "external_metered_spend.jsonl").write_text(
        "\n".join(retained_ledger) + ("\n" if retained_ledger else ""), encoding="utf-8"
    )
    (stage_dir / "repair_lineage.json").write_text(
        json.dumps(
            {
                "retained_source_stage": str(source_stage),
                "replacement_benchmark": WEBARENA,
                "retained_candidate_rows": len(retained_rows),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-stage", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare_stage(
        source_stage=args.source_stage,
        protocol=load_yaml(args.protocol),
        stage_dir=args.stage_dir,
    )


if __name__ == "__main__":
    main()

