from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from router_benchmark.analysis.canonical_uncertainty import uncertainty_rows
from router_benchmark.analysis.paired_tests import paired_effects
from router_benchmark.analysis.reviewer_gates import run_reviewer_gates
from router_benchmark.analysis.vllm_share_permutation import share_matched_permutation_rows
from router_benchmark.scripts import generate_response_to_reviewer as generator
from router_benchmark.scripts.summarize_full_run_spend import summarize_spend
from test_outcome_matrix import _build_bundle


def _write_analysis(bundle: Path, protocol: Path, output: Path) -> None:
    rows, draws = paired_effects(bundle, protocol)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "paired_effects.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "paired_draws.json").write_text(json.dumps(draws, sort_keys=True), encoding="utf-8")
    rank_rows, pareto_rows = uncertainty_rows(bundle, protocol, output / "paired_draws.json")
    with (output / "rank_uncertainty.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rank_rows[0]))
        writer.writeheader()
        writer.writerows(rank_rows)
    with (output / "pareto_uncertainty.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pareto_rows[0]))
        writer.writeheader()
        writer.writerows(pareto_rows)
    permutation_rows, permutation_metadata = share_matched_permutation_rows(
        bundle, protocol, draws=10, excluded_benchmarks={"WebArena (live)"}
    )
    with (output / "vllm_share_permutation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(permutation_rows[0]))
        writer.writeheader()
        writer.writerows(permutation_rows)
    (output / "vllm_share_permutation.json").write_text(json.dumps(permutation_metadata), encoding="utf-8")
    run_reviewer_gates(bundle, protocol, output)


def _write_spend_summary(bundle: Path, output: Path) -> Path:
    spend_summary = output / "spend_summary.json"
    summarize_spend(
        bundle,
        json_output=spend_summary,
        benchmark_output=output / "benchmark_spend.csv",
    )
    return spend_summary


def test_build_response_replaces_pending_with_artifact_backed_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle, protocol = _build_bundle(tmp_path)
    analysis = tmp_path / "analysis"
    _write_analysis(bundle, protocol, analysis)
    spend_summary = _write_spend_summary(bundle, analysis)
    pdf = tmp_path / "paper1.pdf"
    archive = tmp_path / "submission.tar"

    monkeypatch.setattr(generator, "_pdf_page_count", lambda _pdf: 18)
    monkeypatch.setattr(generator, "_archive_listing", lambda _archive: ["main.tex", "figure.png"])

    page_map = {
        "canonical": "12",
        "reproducibility": "10",
        "baselines": "13",
        "spend": "12",
        "bfcl": "15",
        "statistics": "14",
        "ablation": "16",
    }
    text = generator.build_response(
        bundle=bundle,
        protocol=protocol,
        analysis_dir=analysis,
        spend_summary=spend_summary,
        pdf=pdf,
        arxiv_archive=archive,
        page_map=page_map,
    )

    assert "PENDING" not in text
    assert "Canonical bundle digest" in text
    assert "Concern 7" in text
    assert "rank_uncertainty.csv" in text
    assert "pareto_uncertainty.csv" in text
    assert "spend_summary.json" in text
    assert "complete" in text


def test_generate_response_fails_when_required_analysis_artifact_is_missing(tmp_path: Path) -> None:
    bundle, protocol = _build_bundle(tmp_path)
    analysis = tmp_path / "analysis"
    _write_analysis(bundle, protocol, analysis)
    spend_summary = _write_spend_summary(bundle, analysis)
    (analysis / "pareto_uncertainty.csv").unlink()

    with pytest.raises(FileNotFoundError, match="pareto_uncertainty.csv"):
        generator.generate_response(
            bundle=bundle,
            protocol=protocol,
            analysis_dir=analysis,
            spend_summary=spend_summary,
            pdf=tmp_path / "paper1.pdf",
            arxiv_archive=tmp_path / "submission.tar",
            output=tmp_path / "response.md",
        )


def test_generate_response_fails_when_observed_spend_summary_is_missing(tmp_path: Path) -> None:
    bundle, protocol = _build_bundle(tmp_path)
    analysis = tmp_path / "analysis"
    _write_analysis(bundle, protocol, analysis)

    with pytest.raises(FileNotFoundError, match="spend_summary.json"):
        generator.generate_response(
            bundle=bundle,
            protocol=protocol,
            analysis_dir=analysis,
            spend_summary=analysis / "spend_summary.json",
            pdf=tmp_path / "paper1.pdf",
            arxiv_archive=tmp_path / "submission.tar",
            output=tmp_path / "response.md",
        )

