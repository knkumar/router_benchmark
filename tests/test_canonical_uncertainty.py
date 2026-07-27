from __future__ import annotations

import json
from pathlib import Path

from router_benchmark.analysis.canonical_uncertainty import (
    cross_benchmark_rank_rows,
    uncertainty_rows,
)
from router_benchmark.analysis.paired_tests import paired_effects
from test_outcome_matrix import _build_bundle


def test_uncertainty_rows_use_saved_draws_and_cover_all_router_benchmarks(tmp_path: Path) -> None:
    bundle, protocol = _build_bundle(tmp_path)
    _rows, draws = paired_effects(bundle, protocol)
    draws_path = tmp_path / "paired_draws.json"
    draws_path.write_text(json.dumps(draws, sort_keys=True), encoding="utf-8")

    rank_rows, pareto_rows = uncertainty_rows(bundle, protocol, draws_path)

    assert len(rank_rows) == 16
    assert len(pareto_rows) == 16
    assert {row["draws"] for row in rank_rows} == {10000}
    assert {row["draws"] for row in pareto_rows} == {10000}
    assert all(1.0 <= float(row["mean_rank"]) <= 4.0 for row in rank_rows)
    assert all(0.0 <= float(row["pareto_nondominance_probability"]) <= 1.0 for row in pareto_rows)
    assert all("rank_probabilities_json" in row for row in rank_rows)


def test_cross_benchmark_rank_rows_summarize_mean_rank_posterior(tmp_path: Path) -> None:
    bundle, protocol = _build_bundle(tmp_path)
    _rows, draws = paired_effects(bundle, protocol)
    draws_path = tmp_path / "paired_draws.json"
    draws_path.write_text(json.dumps(draws, sort_keys=True), encoding="utf-8")

    rows = cross_benchmark_rank_rows(bundle, protocol, draws_path)

    n_routers = len(rows)
    assert n_routers >= 2
    assert {row["draws"] for row in rows} == {10000}
    for row in rows:
        assert 1.0 <= float(row["posterior_mean_rank"]) <= float(n_routers)
        assert float(row["mean_rank_ci_low"]) <= float(row["posterior_mean_rank"]) <= float(row["mean_rank_ci_high"])
        assert float(row["rank_variance_ci_low"]) <= float(row["posterior_rank_variance"]) <= float(row["rank_variance_ci_high"])
        assert 0.0 <= float(row["prob_best_mean_rank"]) <= 1.0
    # Exactly one unit of "best mean rank" credit is distributed across routers per draw.
    assert abs(sum(float(row["prob_best_mean_rank"]) for row in rows) - 1.0) < 1e-9

