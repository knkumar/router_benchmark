#!/usr/bin/env python3
"""Audit the regenerated Paper 1 PDF, arXiv archive, and reviewer matrix."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path


OVERFULL_RE = re.compile(r"Overfull \\hbox \(([0-9.]+)pt too wide\)")
BAD_LOG_RE = re.compile(
    r"(^!|Emergency stop|Fatal error|LaTeX Warning:.*(?:Citation|Reference).*undefined)",
    re.MULTILINE,
)


def _pdf_page_count(pdf: Path) -> int:
    output = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
    for line in output.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise ValueError(f"Could not read PDF page count from {pdf}")


def _pdf_text(pdf: Path) -> str:
    return subprocess.check_output(["pdftotext", str(pdf), "-"], text=True, errors="replace")


def _archive_files(path: Path) -> list[str]:
    with tarfile.open(path) as archive:
        return sorted(member.name for member in archive.getmembers() if member.isfile())


def _validated_spend_summary(path: Path) -> dict[str, object]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "complete":
        raise ValueError("spend summary is not complete")
    for key in (
        "candidate_model_api_usd",
        "router_service_usd",
        "external_metered_usd",
        "infrastructure_usd",
        "total_metered_usd",
    ):
        value = report.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise ValueError(f"spend summary has invalid {key}")
    return report


def _large_overfulls(log_text: str, threshold_pt: float) -> list[float]:
    return [float(match.group(1)) for match in OVERFULL_RE.finditer(log_text) if float(match.group(1)) > threshold_pt]


def audit_submission_package(
    *,
    pdf: Path,
    log: Path,
    arxiv_archive: Path,
    response: Path,
    spend_summary: Path,
    output: Path,
    required_pdf_phrase: str,
    max_overfull_pt: float,
) -> dict[str, object]:
    missing = [path for path in (pdf, log, arxiv_archive, response, spend_summary) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing submission artifacts: " + ", ".join(str(path) for path in missing))

    log_text = log.read_text(encoding="utf-8", errors="replace")
    bad_log = [match.group(0) for match in BAD_LOG_RE.finditer(log_text)]
    large_overfulls = _large_overfulls(log_text, max_overfull_pt)
    if bad_log:
        raise ValueError("LaTeX log contains fatal errors or undefined references/citations")
    if large_overfulls:
        raise ValueError(f"LaTeX log contains overfull boxes above {max_overfull_pt}pt: {large_overfulls}")

    page_count = _pdf_page_count(pdf)
    if page_count < 1:
        raise ValueError("PDF has no pages")
    pdf_text = " ".join(_pdf_text(pdf).split())
    if required_pdf_phrase not in pdf_text:
        raise ValueError(f"PDF text does not contain required phrase: {required_pdf_phrase}")

    archive_files = _archive_files(arxiv_archive)
    if "main.tex" not in archive_files:
        raise ValueError("arXiv archive does not contain main.tex")
    unsafe = [name for name in archive_files if not re.fullmatch(r"[A-Za-z0-9._/-]+", name)]
    if unsafe:
        raise ValueError(f"arXiv archive contains unsafe filenames: {unsafe}")

    response_text = response.read_text(encoding="utf-8")
    if "PENDING" in response_text or "UNCONFIRMED" in response_text:
        raise ValueError("response-to-reviewer still contains unresolved markers")
    for concern in range(1, 11):
        if f"Concern {concern}" not in response_text:
            raise ValueError(f"response-to-reviewer is missing Concern {concern}")
    if spend_summary.name not in response_text:
        raise ValueError("response-to-reviewer does not cite the observed spend summary")
    spend = _validated_spend_summary(spend_summary)

    report = {
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pdf": str(pdf),
        "pdf_pages": page_count,
        "log": str(log),
        "max_overfull_pt": max_overfull_pt,
        "arxiv_archive": str(arxiv_archive),
        "arxiv_files": archive_files,
        "response_to_reviewer": str(response),
        "spend_summary": str(spend_summary),
        "total_metered_usd": spend["total_metered_usd"],
        "required_pdf_phrase": required_pdf_phrase,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--arxiv-archive", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--spend-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-pdf-phrase", default="Canonical Full-Rebuild Evidence")
    parser.add_argument("--max-overfull-pt", type=float, default=25.0)
    args = parser.parse_args()
    report = audit_submission_package(
        pdf=args.pdf,
        log=args.log,
        arxiv_archive=args.arxiv_archive,
        response=args.response,
        spend_summary=args.spend_summary,
        output=args.output,
        required_pdf_phrase=args.required_pdf_phrase,
        max_overfull_pt=args.max_overfull_pt,
    )
    print(f"Submission package audit passed: {report['output'] if 'output' in report else args.output}.")


if __name__ == "__main__":
    main()

