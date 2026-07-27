from __future__ import annotations

import tarfile
import json
from pathlib import Path

import pytest

from router_benchmark.scripts import audit_submission_package as audit


def _write_archive(path: Path) -> None:
    source = path.parent / "main.tex"
    source.write_text("test\n", encoding="utf-8")
    with tarfile.open(path, "w") as archive:
        archive.add(source, arcname="main.tex")


def _write_response(path: Path) -> None:
    lines = ["# Response", "", "`spend_summary.json`"]
    lines.extend(f"| Concern {index} | complete |" for index in range(1, 11))
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_spend_summary(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "candidate_model_api_usd": 1.0,
                "router_service_usd": 0.2,
                "external_metered_usd": 0.1,
                "infrastructure_usd": 0.0,
                "total_metered_usd": 1.3,
            }
        ),
        encoding="utf-8",
    )


def test_submission_audit_passes_with_clean_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = tmp_path / "paper1.pdf"
    log = tmp_path / "paper1.log"
    archive = tmp_path / "submission.tar"
    response = tmp_path / "response.md"
    spend_summary = tmp_path / "spend_summary.json"
    output = tmp_path / "audit.json"
    pdf.write_bytes(b"%PDF fixture\n")
    log.write_text("Output written on paper1.pdf\n", encoding="utf-8")
    _write_archive(archive)
    _write_response(response)
    _write_spend_summary(spend_summary)
    monkeypatch.setattr(audit, "_pdf_page_count", lambda _pdf: 20)
    monkeypatch.setattr(audit, "_pdf_text", lambda _pdf: "Canonical Full-Rebuild Evidence")

    report = audit.audit_submission_package(
        pdf=pdf,
        log=log,
        arxiv_archive=archive,
        response=response,
        spend_summary=spend_summary,
        output=output,
        required_pdf_phrase="Canonical Full-Rebuild Evidence",
        max_overfull_pt=25.0,
    )

    assert report["status"] == "passed"
    assert output.exists()


def test_submission_audit_rejects_large_overfull_box(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = tmp_path / "paper1.pdf"
    log = tmp_path / "paper1.log"
    archive = tmp_path / "submission.tar"
    response = tmp_path / "response.md"
    spend_summary = tmp_path / "spend_summary.json"
    pdf.write_bytes(b"%PDF fixture\n")
    log.write_text(r"Overfull \hbox (505.12177pt too wide) has occurred", encoding="utf-8")
    _write_archive(archive)
    _write_response(response)
    _write_spend_summary(spend_summary)
    monkeypatch.setattr(audit, "_pdf_page_count", lambda _pdf: 20)
    monkeypatch.setattr(audit, "_pdf_text", lambda _pdf: "Canonical Full-Rebuild Evidence")

    with pytest.raises(ValueError, match="overfull boxes"):
        audit.audit_submission_package(
            pdf=pdf,
            log=log,
            arxiv_archive=archive,
            response=response,
            spend_summary=spend_summary,
            output=tmp_path / "audit.json",
            required_pdf_phrase="Canonical Full-Rebuild Evidence",
            max_overfull_pt=25.0,
        )


def test_submission_audit_requires_response_to_cite_spend_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = tmp_path / "paper1.pdf"
    log = tmp_path / "paper1.log"
    archive = tmp_path / "submission.tar"
    response = tmp_path / "response.md"
    spend_summary = tmp_path / "spend_summary.json"
    pdf.write_bytes(b"%PDF fixture\n")
    log.write_text("Output written on paper1.pdf\n", encoding="utf-8")
    _write_archive(archive)
    response.write_text("\n".join(f"| Concern {index} | complete |" for index in range(1, 11)), encoding="utf-8")
    _write_spend_summary(spend_summary)
    monkeypatch.setattr(audit, "_pdf_page_count", lambda _pdf: 20)
    monkeypatch.setattr(audit, "_pdf_text", lambda _pdf: "Canonical Full-Rebuild Evidence")

    with pytest.raises(ValueError, match="does not cite the observed spend summary"):
        audit.audit_submission_package(
            pdf=pdf,
            log=log,
            arxiv_archive=archive,
            response=response,
            spend_summary=spend_summary,
            output=tmp_path / "audit.json",
            required_pdf_phrase="Canonical Full-Rebuild Evidence",
            max_overfull_pt=25.0,
        )

