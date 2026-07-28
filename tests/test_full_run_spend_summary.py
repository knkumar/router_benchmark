from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from router_benchmark.scripts.summarize_full_run_spend import summarize_spend


def test_spend_summary_matches_locked_dry_run_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "candidate_outcomes.csv").write_text(
        "benchmark_id,model_api_cost_usd,provider_generation_usd,fallback_generation_usd\n"
        "alpha,0.30,0.20,0.10\n"
        "beta,0.70,0.50,0.20\n",
        encoding="utf-8",
    )
    (bundle / "routes.csv").write_text(
        "benchmark_id,router_service_usd\n"
        "alpha,0.04\n"
        "beta,0.06\n",
        encoding="utf-8",
    )
    (bundle / "provenance.json").write_text(
        json.dumps({"external_metered_usd": 0.15, "infrastructure_usd": 0.05}),
        encoding="utf-8",
    )
    (bundle / "manifest.json").write_text(json.dumps({"protocol_id": "fixture"}), encoding="utf-8")
    report = summarize_spend(
        bundle,
        json_output=tmp_path / "spend_summary.json",
        benchmark_output=tmp_path / "benchmark_spend.csv",
    )

    assert report["status"] == "complete"
    assert report["candidate_rows"] == 2
    assert report["route_rows"] == 2
    assert report["candidate_model_api_usd"] == pytest.approx(1.0)
    assert report["router_service_usd"] == pytest.approx(0.10)
    assert report["external_metered_usd"] == pytest.approx(0.15)
    assert report["total_metered_usd"] == pytest.approx(1.30)
    with (tmp_path / "benchmark_spend.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["benchmark_id"] for row in rows} == {"alpha", "beta"}


def test_spend_summary_rejects_negative_cost(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "candidate_outcomes.csv").write_text(
        "benchmark_id,model_api_cost_usd,provider_generation_usd,fallback_generation_usd\n"
        "RouterBench (live),-1.0,0.0,0.0\n",
        encoding="utf-8",
    )
    (bundle / "routes.csv").write_text(
        "benchmark_id,router_service_usd\nRouterBench (live),0.0\n",
        encoding="utf-8",
    )
    (bundle / "provenance.json").write_text("{}", encoding="utf-8")
    (bundle / "manifest.json").write_text(json.dumps({"protocol_id": "fixture"}), encoding="utf-8")

    with pytest.raises(ValueError, match="finite and nonnegative"):
        summarize_spend(bundle, json_output=tmp_path / "summary.json", benchmark_output=tmp_path / "bench.csv")
