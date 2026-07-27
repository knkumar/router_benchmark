from __future__ import annotations

from router_benchmark.analysis.vllm_share_permutation import share_matched_permutation_rows
from test_outcome_matrix import _build_bundle


def test_share_matched_permutation_preserves_observed_tier_counts(tmp_path) -> None:
    bundle, protocol = _build_bundle(tmp_path)
    rows, metadata = share_matched_permutation_rows(
        bundle,
        protocol,
        draws=20,
        excluded_benchmarks={"WebArena (live)"},
    )
    assert len(rows) == 3
    assert {row["route_positions"] for row in rows} == {2}
    assert all(row["cheap_routes"] + row["mid_routes"] + row["strong_routes"] == 2 for row in rows)
    assert all(row["actual_success"] == 1.0 for row in rows)
    assert metadata["excluded_benchmarks"] == ["WebArena (live)"]

