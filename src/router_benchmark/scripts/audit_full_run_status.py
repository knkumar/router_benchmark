#!/usr/bin/env python3
"""Report step-by-step full-run completion status without provider calls."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from router_benchmark.protocol.protocol_tools import load_yaml


ANALYSIS_FILES = [
    "paired_effects.csv",
    "paired_draws.json",
    "rank_uncertainty.csv",
    "pareto_uncertainty.csv",
    "candidate_tier_summary.csv",
    "baseline_summary.csv",
    "baseline_reconciliation.json",
    "bfcl_route_equivalence.csv",
    "ablation_registry.json",
]

CANONICAL_FILES = [
    "manifest.json",
    "router_configs.json",
    "candidate_outcomes.csv",
    "routes.csv",
    "outcomes.csv",
    "results.csv",
    "traces.jsonl",
    "provenance.json",
    "checksums.sha256",
]


def _csv_row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _json_status(path: Path) -> str | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    status = payload.get("status")
    return str(status) if status is not None else None


def _text_has_no_markers(path: Path, markers: set[str]) -> bool | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    return not any(marker in text for marker in markers)


def _step(name: str, status: str, evidence: str, blocker: str | None = None) -> dict[str, str]:
    row = {"name": name, "status": status, "evidence": evidence}
    if blocker:
        row["blocker"] = blocker
    return row


def _expected_counts(protocol: dict[str, Any]) -> tuple[int, int, int]:
    benchmarks = protocol.get("benchmarks", {})
    candidate_rows = sum(int(entry.get("total_outcome_rows", 0)) for entry in benchmarks.values())
    route_rows = sum(int(entry.get("total_route_rows", 0)) for entry in benchmarks.values())
    task_count = sum(len(entry.get("task_ids", [])) for entry in benchmarks.values())
    return task_count, candidate_rows, route_rows


def status_report(
    *,
    protocol_path: Path,
    readiness_path: Path,
    stage_dir: Path,
    bundle_dir: Path,
    analysis_dir: Path,
    paper_tables_dir: Path,
    pdf: Path,
    arxiv_archive: Path,
    response: Path,
    submission_audit: Path,
    spend_summary: Path,
    output: Path | None = None,
) -> dict[str, Any]:
    protocol = load_yaml(protocol_path)
    task_count, expected_candidate_rows, expected_route_rows = _expected_counts(protocol)
    steps: list[dict[str, str]] = []

    approved = protocol.get("execution_budget", {}).get("approval_status") == "approved"
    steps.append(
        _step(
            "approval",
            "done" if approved else "blocked",
            f"{protocol_path}: approval_status={protocol.get('execution_budget', {}).get('approval_status')}",
            None if approved else "explicit author approval is required before provider-backed execution",
        )
    )

    readiness_status = _json_status(readiness_path)
    readiness_evidence = f"{readiness_path}: status={readiness_status}" if readiness_status else f"{readiness_path}: missing"
    steps.append(_step("readiness", "done" if readiness_status == "ready" else "blocked", readiness_evidence))

    stage_candidate_rows = _csv_row_count(stage_dir / "candidate_outcomes.csv")
    steps.append(
        _step(
            "stage_candidate_matrix",
            "done" if stage_candidate_rows == expected_candidate_rows else "not_started",
            f"{stage_dir / 'candidate_outcomes.csv'}: rows={stage_candidate_rows}, expected={expected_candidate_rows}",
        )
    )

    stage_route_rows = _csv_row_count(stage_dir / "routes.csv")
    steps.append(
        _step(
            "stage_route_replay",
            "done" if stage_route_rows == expected_route_rows else "not_started",
            f"{stage_dir / 'routes.csv'}: rows={stage_route_rows}, expected={expected_route_rows}",
        )
    )

    missing_bundle_files = [name for name in CANONICAL_FILES if not (bundle_dir / name).exists()]
    bundle_candidate_rows = _csv_row_count(bundle_dir / "candidate_outcomes.csv")
    bundle_route_rows = _csv_row_count(bundle_dir / "routes.csv")
    bundle_done = not missing_bundle_files and bundle_candidate_rows == expected_candidate_rows and bundle_route_rows == expected_route_rows
    steps.append(
        _step(
            "canonical_bundle",
            "done" if bundle_done else "not_started",
            (
                f"{bundle_dir}: missing={missing_bundle_files}; "
                f"candidate_rows={bundle_candidate_rows}, expected_candidate_rows={expected_candidate_rows}; "
                f"route_rows={bundle_route_rows}, expected_route_rows={expected_route_rows}"
            ),
        )
    )

    missing_analysis = [name for name in ANALYSIS_FILES if not (analysis_dir / name).exists()]
    steps.append(
        _step(
            "analysis_and_reviewer_gates",
            "done" if not missing_analysis else "not_started",
            f"{analysis_dir}: missing={missing_analysis}",
        )
    )

    appendix = paper_tables_dir / "canonical_rebuild_appendix.tex"
    tables_done = appendix.exists()
    steps.append(_step("canonical_tables", "done" if tables_done else "not_started", f"{appendix}: exists={tables_done}"))

    pdf_done = pdf.exists() and tables_done
    steps.append(
        _step(
            "paper_pdf",
            "done" if pdf_done else "not_started",
            f"{pdf}: exists={pdf.exists()}; canonical_tables_exist={tables_done}",
        )
    )
    archive_done = arxiv_archive.exists() and pdf_done
    steps.append(
        _step(
            "arxiv_archive",
            "done" if archive_done else "not_started",
            f"{arxiv_archive}: exists={arxiv_archive.exists()}; full_findings_pdf_done={pdf_done}",
        )
    )

    response_clean = _text_has_no_markers(response, {"PENDING", "UNCONFIRMED"})
    response_done = response_clean is True and bundle_done and tables_done
    steps.append(
        _step(
            "response_to_reviewer",
            "done" if response_done else "not_started",
            f"{response}: no_pending_markers={response_clean}; canonical_bundle_done={bundle_done}; canonical_tables_exist={tables_done}",
        )
    )

    audit_status = _json_status(submission_audit)
    steps.append(
        _step(
            "submission_audit",
            "done" if audit_status == "passed" else "not_started",
            f"{submission_audit}: status={audit_status}",
        )
    )
    spend_status = _json_status(spend_summary)
    steps.append(
        _step(
            "spend_summary",
            "done" if spend_status == "complete" else "not_started",
            f"{spend_summary}: status={spend_status}",
        )
    )

    blocked = [step for step in steps if step["status"] == "blocked"]
    incomplete = [step for step in steps if step["status"] != "done"]
    report = {
        "status": "complete" if not incomplete else "blocked" if blocked else "incomplete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_id": protocol.get("protocol_id"),
        "task_count": task_count,
        "expected_candidate_rows": expected_candidate_rows,
        "expected_route_rows": expected_route_rows,
        "steps": steps,
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--paper-tables-dir", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--arxiv-archive", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--submission-audit", type=Path, required=True)
    parser.add_argument("--spend-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    report = status_report(
        protocol_path=args.protocol,
        readiness_path=args.readiness,
        stage_dir=args.stage_dir,
        bundle_dir=args.bundle_dir,
        analysis_dir=args.analysis_dir,
        paper_tables_dir=args.paper_tables_dir,
        pdf=args.pdf,
        arxiv_archive=args.arxiv_archive,
        response=args.response,
        submission_audit=args.submission_audit,
        spend_summary=args.spend_summary,
        output=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "complete" and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

