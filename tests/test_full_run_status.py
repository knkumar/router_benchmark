from __future__ import annotations

import json
from pathlib import Path

import yaml

from router_benchmark.scripts.audit_full_run_status import ANALYSIS_FILES, CANONICAL_FILES, status_report
from test_full_run_preflight import _full_protocol


def _write_csv(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["id\n"] + [f"{index}\n" for index in range(rows)]
    path.write_text("".join(lines), encoding="utf-8")


def test_current_full_run_status_reports_unexecuted_workspace() -> None:
    report = status_report(
        protocol_path=Path("protocol/paper1_rebuild.yaml"),
        readiness_path=Path("analysis/output/paper1_canonical/full_run_readiness.json"),
        stage_dir=Path("output/full_run_stage"),
        bundle_dir=Path("output/live/paper1_canonical_v1"),
        analysis_dir=Path("analysis/output/paper1_canonical"),
        paper_tables_dir=Path("paper/tables"),
        pdf=Path("paper/paper1.pdf"),
        arxiv_archive=Path("paper/arxiv/submission.tar"),
        response=Path("review_plans/response-to-reviewer.md"),
        submission_audit=Path("analysis/output/paper1_canonical/submission_audit.json"),
        spend_summary=Path("analysis/output/paper1_canonical/spend_summary.json"),
    )

    assert report["status"] == "blocked"
    assert report["expected_candidate_rows"] == 2610
    assert report["expected_route_rows"] == 2320
    approval = next(step for step in report["steps"] if step["name"] == "approval")
    assert approval["status"] == "done"
    stage = next(step for step in report["steps"] if step["name"] == "stage_candidate_matrix")
    assert stage["status"] == "not_started"


def test_full_run_status_accepts_complete_fixture(tmp_path: Path) -> None:
    protocol = _full_protocol()
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol), encoding="utf-8")
    readiness = tmp_path / "readiness.json"
    readiness.write_text(json.dumps({"status": "ready", "blockers": []}), encoding="utf-8")
    stage = tmp_path / "stage"
    bundle = tmp_path / "bundle"
    analysis = tmp_path / "analysis"
    tables = tmp_path / "tables"
    pdf = tmp_path / "paper.pdf"
    archive = tmp_path / "submission.tar"
    response = tmp_path / "response.md"
    audit = tmp_path / "submission_audit.json"
    spend = tmp_path / "spend_summary.json"
    output = tmp_path / "status.json"

    expected_candidate_rows = sum(entry["total_outcome_rows"] for entry in protocol["benchmarks"].values())
    expected_route_rows = sum(entry["total_route_rows"] for entry in protocol["benchmarks"].values())
    _write_csv(stage / "candidate_outcomes.csv", expected_candidate_rows)
    _write_csv(stage / "routes.csv", expected_route_rows)
    for name in CANONICAL_FILES:
        path = bundle / name
        if name == "candidate_outcomes.csv":
            _write_csv(path, expected_candidate_rows)
        elif name == "routes.csv":
            _write_csv(path, expected_route_rows)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")
    for name in ANALYSIS_FILES:
        path = analysis / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    (tables / "canonical_rebuild_appendix.tex").parent.mkdir(parents=True, exist_ok=True)
    (tables / "canonical_rebuild_appendix.tex").write_text("% fixture\n", encoding="utf-8")
    pdf.write_text("fixture pdf marker\n", encoding="utf-8")
    archive.write_text("fixture archive marker\n", encoding="utf-8")
    response.write_text("Concern 1\nConcern 10\n", encoding="utf-8")
    audit.write_text(json.dumps({"status": "passed"}), encoding="utf-8")
    spend.write_text(json.dumps({"status": "complete"}), encoding="utf-8")

    report = status_report(
        protocol_path=protocol_path,
        readiness_path=readiness,
        stage_dir=stage,
        bundle_dir=bundle,
        analysis_dir=analysis,
        paper_tables_dir=tables,
        pdf=pdf,
        arxiv_archive=archive,
        response=response,
        submission_audit=audit,
        spend_summary=spend,
        output=output,
    )

    assert report["status"] == "complete"
    assert output.exists()
    assert all(step["status"] == "done" for step in report["steps"])
