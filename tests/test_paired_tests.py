from __future__ import annotations

from router_benchmark.analysis.paired_tests import paired_effects
from test_outcome_matrix import _build_bundle


def test_paired_effects_reads_locked_bundle_and_records_protocol_hash(tmp_path) -> None:
    bundle, protocol = _build_bundle(tmp_path)
    rows, draws = paired_effects(bundle, protocol)
    assert len(rows) == 24
    # paired_n is now the task-level cluster count (one task per benchmark in the
    # fixture), not the joined-row count, so inference is not pseudoreplicated.
    assert {row["paired_n"] for row in rows} == {1}
    assert {row["test"] for row in rows} == {"task_cluster_bootstrap"}
    assert all(row["correction_family"] == "six router-pair comparisons within benchmark" for row in rows)
    assert all(len(str(row["rebuild_protocol_sha256"])) == 64 for row in rows)
    assert all(len(str(row["analysis_protocol_sha256"])) == 64 for row in rows)
    assert set(draws) == {"RouterBench (live)", "BFCL v4 (live)", "tau2-bench (live)", "WebArena (live)"}

