#!/usr/bin/env python3
"""Generate a no-spend approval packet for the full Paper 1 rebuild."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

from router_benchmark.scripts._paths import repository_root

ROOT = repository_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from router_benchmark.protocol.protocol_tools import load_yaml
from router_benchmark.scripts.audit_full_run_readiness import readiness_report
from router_benchmark.scripts.preflight_full_run import (
    candidate_reservations_from_full_protocol,
    external_metered_reservation_from_full_protocol,
    full_run_candidate_cells,
    router_service_reservation_from_full_protocol,
)


def _usd(value: float) -> str:
    return f"${value:,.2f}"


def _reservation_by_benchmark(protocol: dict[str, Any]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    reservations = candidate_reservations_from_full_protocol(protocol)
    for cell in full_run_candidate_cells(protocol):
        totals[cell.benchmark_id] += reservations[
            f"{cell.benchmark_id}|{cell.task_id}|{cell.candidate_id}|{cell.outcome_replicate}"
        ]
    return dict(sorted(totals.items()))


def _route_rows_by_benchmark(protocol: dict[str, Any]) -> dict[str, int]:
    return {
        benchmark: int(entry["total_route_rows"])
        for benchmark, entry in sorted(protocol["benchmarks"].items())
    }


def build_packet(protocol_path: Path, readiness_path: Path | None = None) -> str:
    protocol = load_yaml(protocol_path)
    report = (
        json.loads(readiness_path.read_text(encoding="utf-8"))
        if readiness_path and readiness_path.exists()
        else readiness_report(protocol_path, check_environment=False)
    )
    reservations = candidate_reservations_from_full_protocol(protocol)
    candidate_total = sum(reservations.values())
    router_total = router_service_reservation_from_full_protocol(protocol)
    external_total = external_metered_reservation_from_full_protocol(protocol) - router_total
    cap = float(protocol["full_run_cost_reservations"]["total_cap_usd"])
    benchmark_totals = _reservation_by_benchmark(protocol)
    route_totals = _route_rows_by_benchmark(protocol)
    blockers = report.get("blockers", [])
    blocker_text = "; ".join(blockers) if blockers else "none"

    lines = [
        "# Full-Run Approval Packet",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "This packet is a review artifact only. It does not approve execution and does not edit `approval_status`.",
        "",
        "## Protocol",
        "",
        f"- Protocol ID: `{protocol['protocol_id']}`",
        f"- Protocol path: `{protocol_path}`",
        f"- Current approval status: `{protocol['execution_budget']['approval_status']}`",
        f"- Readiness status: `{report['status']}`",
        f"- Current blockers: {blocker_text}",
        "",
        "## Cost Envelope",
        "",
        f"- Hard cap: {_usd(cap)}",
        f"- Candidate model API reserve: {_usd(candidate_total)} across {len(reservations)} candidate cells",
        f"- Router-service reserve: {_usd(router_total)} across {sum(route_totals.values())} route rows",
        f"- Noncandidate external metered reserve: {_usd(external_total)}",
        f"- Total reserved before safety margin: {_usd(candidate_total + router_total + external_total)}",
        f"- Unallocated cap margin: {_usd(cap - candidate_total - router_total - external_total)}",
        "",
        "## Candidate Reserve By Benchmark",
        "",
        "| Benchmark | Candidate rows | Candidate reserve | Route rows |",
        "|---|---:|---:|---:|",
    ]
    for benchmark, reserve in benchmark_totals.items():
        rows = protocol["benchmarks"][benchmark]["total_outcome_rows"]
        lines.append(f"| {benchmark} | {rows} | {_usd(reserve)} | {route_totals[benchmark]} |")
    lines.extend([
        "",
        "## Execution Conditions",
        "",
        "- Prompt-output caching must remain disabled.",
        "- Resume is allowed only for already persisted candidate cells.",
        "- Stop before any unreserved cell, cache-derived row, missing trace digest, nonfinite cost, or budget breach.",
        "- Candidate model API cost, router-service spend, and external metered spend must stay separately reported.",
        "",
        "## Approval Action",
        "",
        "Execution may start only after the author explicitly approves the cap and changes:",
        "",
        "```yaml",
        "execution_budget:",
        "  approval_status: approved",
        "```",
        "",
        "## Commands After Approval",
        "",
        "```bash",
        "make full-run-preflight",
        "make full-run-candidates FULL_RUN_STAGE_DIR=output/full_run_stage",
        "make full-run-routes FULL_RUN_STAGE_DIR=output/full_run_stage",
        "make full-run-bundle FULL_RUN_STAGE_DIR=output/full_run_stage CANONICAL_BUNDLE=output/live/paper1_canonical_v1",
        "make submission-audit CANONICAL_BUNDLE=output/live/paper1_canonical_v1",
        "```",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--readiness", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = build_packet(args.protocol, args.readiness)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(packet, encoding="utf-8")
    print(f"Full-run approval packet written to {args.output}.")


if __name__ == "__main__":
    main()
