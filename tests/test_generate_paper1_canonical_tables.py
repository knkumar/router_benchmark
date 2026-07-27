from __future__ import annotations

from pathlib import Path

import pytest

from router_benchmark.analysis.canonical_uncertainty import uncertainty_rows
from router_benchmark.analysis.paired_tests import paired_effects
from router_benchmark.analysis.vllm_share_permutation import share_matched_permutation_rows
from router_benchmark.analysis.reviewer_gates import run_reviewer_gates
from router_benchmark.scripts.generate_paper1_canonical_tables import generate_tables
from test_outcome_matrix import _build_bundle


def _write_paired_outputs(bundle: Path, protocol: Path, output: Path) -> None:
    rows, draws = paired_effects(bundle, protocol)
    output.mkdir(parents=True, exist_ok=True)
    import csv
    import json

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
    (output / "canonical_metric_suite.csv").write_text(
        "router,success_rate,cost_per_task_usd,cost_per_success_usd,latency_p50_ms,"
        "latency_p95_ms,tool_call_accuracy,fallback_rate,route_stability,mean_confidence,"
        "easy_success,medium_success,hard_success,robustness_std\n"
        "vLLM Semantic Router,0.5131,0.063545,0.121100,27688.8,67050.1,0.7511,0.0892,"
        "0.9917,0.6376,0.9463,0.3361,0.1859,0.0491\n",
        encoding="utf-8",
    )
    (output / "pool_ablation_comparison.csv").write_text(
        "router,widegap_success,narrowgap_success,success_delta,widegap_cost_per_task,"
        "narrowgap_cost_per_task,cost_ratio_narrow_over_wide,widegap_stability,narrowgap_stability\n"
        "LiteLLM Router,0.5667,0.5500,-0.0167,0.000038,0.000618,16.33,1.0000,1.0000\n",
        encoding="utf-8",
    )
    (output / "expected_utility.csv").write_text(
        "router,cost_basis,value_of_success_usd,latency_price_usd_per_s,utility_usd_per_task\n"
        "Aurelio Semantic Router,candidate,0.1,0.0,-0.0235\n"
        "Aurelio Semantic Router,candidate,1.0,0.0,0.5453\n"
        "Aurelio Semantic Router,candidate_plus_service,0.1,0.0,-0.0335\n"
        "Aurelio Semantic Router,candidate_plus_service,1.0,0.0,0.5353\n"
        "LiteLLM Router,candidate,0.1,0.0,0.0429\n"
        "LiteLLM Router,candidate,1.0,0.0,0.4449\n"
        "LiteLLM Router,candidate_plus_service,0.1,0.0,0.0429\n"
        "LiteLLM Router,candidate_plus_service,1.0,0.0,0.4449\n"
        "Always-Mid,candidate,0.1,0.0,-0.0230\n"
        "Always-Mid,candidate,1.0,0.0,0.5460\n"
        "Always-Mid,candidate_plus_service,0.1,0.0,-0.0230\n"
        "Always-Mid,candidate_plus_service,1.0,0.0,0.5460\n",
        encoding="utf-8",
    )
    (output / "pareto_uncertainty_all_policy.csv").write_text(
        "benchmark_id,policy_name,policy_type,draws,pareto_nondominance_probability,"
        "universe,cost_definition\n"
        "RouterBench (live),Aurelio Semantic Router (live),router,10000,0.0,"
        "all-policy (routers + fixed-tier baselines),router+service vs candidate\n"
        "RouterBench (live),vLLM Semantic Router (live),router,10000,0.0,"
        "all-policy (routers + fixed-tier baselines),router+service vs candidate\n"
        "RouterBench (live),LiteLLM Router (live),router,10000,1.0,"
        "all-policy (routers + fixed-tier baselines),router+service vs candidate\n"
        "RouterBench (live),RouteLLM (live),router,10000,0.0,"
        "all-policy (routers + fixed-tier baselines),router+service vs candidate\n"
        "RouterBench (live),Always-Cheapest,fixed-tier,10000,1.0,"
        "all-policy (routers + fixed-tier baselines),router+service vs candidate\n"
        "RouterBench (live),Always-Mid,fixed-tier,10000,0.5,"
        "all-policy (routers + fixed-tier baselines),router+service vs candidate\n"
        "RouterBench (live),Always-Strongest,fixed-tier,10000,0.25,"
        "all-policy (routers + fixed-tier baselines),router+service vs candidate\n"
        "tau2-bench (live),Aurelio Semantic Router (live),router,10000,0.4979,"
        "all-policy (routers + fixed-tier baselines),router+service vs candidate\n"
        "tau2-bench (live),vLLM Semantic Router (live),router,10000,0.7529,"
        "all-policy (routers + fixed-tier baselines),router+service vs candidate\n"
        "tau2-bench (live),LiteLLM Router (live),router,10000,1.0,"
        "all-policy (routers + fixed-tier baselines),router+service vs candidate\n"
        "tau2-bench (live),RouteLLM (live),router,10000,0.0,"
        "all-policy (routers + fixed-tier baselines),router+service vs candidate\n"
        "tau2-bench (live),Always-Cheapest,fixed-tier,10000,1.0,"
        "all-policy (routers + fixed-tier baselines),router+service vs candidate\n"
        "tau2-bench (live),Always-Mid,fixed-tier,10000,0.7312,"
        "all-policy (routers + fixed-tier baselines),router+service vs candidate\n"
        "tau2-bench (live),Always-Strongest,fixed-tier,10000,0.3116,"
        "all-policy (routers + fixed-tier baselines),router+service vs candidate\n",
        encoding="utf-8",
    )
    (output / "cascade_operating_metrics.csv").write_text(
        "benchmark,success,avg_calls,avg_latency_s,cost_per_task_usd,success_per_usd\n"
        "RouterBench,0.9167,1.9500,0.000000,0.0009912000,924.804950\n"
        "WebArena,0.2900,2.6333,288.318487,0.1854326070,1.563910\n",
        encoding="utf-8",
    )


def test_generate_tables_requires_and_writes_canonical_fragments(tmp_path: Path) -> None:
    bundle, protocol = _build_bundle(tmp_path)
    analysis = tmp_path / "analysis"
    tables = tmp_path / "tables"

    _write_paired_outputs(bundle, protocol, analysis)
    run_reviewer_gates(bundle, protocol, analysis)

    generate_tables(bundle, protocol, analysis, tables)

    expected = {
        "canonical_rebuild_summary.tex",
        "canonical_spend_summary.tex",
        "canonical_candidate_tiers.tex",
        "canonical_baselines.tex",
        "canonical_router_summary.tex",
        "canonical_route_selection_summary.tex",
        "canonical_paired_effects.tex",
        "canonical_rank_uncertainty.tex",
        "canonical_pareto_uncertainty.tex",
        "canonical_pareto_uncertainty_all_policy.tex",
        "canonical_bfcl_equivalence.tex",
        "canonical_ablation_registry.tex",
        "canonical_vllm_share_permutation.tex",
        "canonical_metric_suite.tex",
        "canonical_pool_ablation.tex",
        "canonical_expected_utility.tex",
        "canonical_expected_utility_candidate.tex",
        "canonical_cascade_operating.tex",
        "canonical_artifact_manifest.tex",
        "canonical_rebuild_appendix.tex",
    }
    assert {path.name for path in tables.iterdir()} == expected
    combined = "\n".join((tables / name).read_text(encoding="utf-8") for name in sorted(expected))
    assert "PENDING" not in combined
    assert "RouterBench" in combined
    assert "Always-Cheapest" in combined
    assert "BFCL v4" in combined
    assert "deferred" in combined
    appendix = (tables / "canonical_rebuild_appendix.tex").read_text(encoding="utf-8")
    assert r"\label{app:canonical-full-rebuild}" in appendix
    assert "canonical_paired_effects.tex" in appendix
    assert "canonical_pareto_uncertainty.tex" in appendix
    # The vLLM permutation table lives in the Results body, not the appendix, to
    # avoid a multiply-defined label; the appendix must NOT re-input it.
    assert "canonical_vllm_share_permutation.tex" not in appendix


def test_generate_tables_fails_when_analysis_artifact_is_missing(tmp_path: Path) -> None:
    bundle, protocol = _build_bundle(tmp_path)
    analysis = tmp_path / "analysis"
    tables = tmp_path / "tables"
    _write_paired_outputs(bundle, protocol, analysis)
    run_reviewer_gates(bundle, protocol, analysis)
    (analysis / "bfcl_route_equivalence.csv").unlink()

    with pytest.raises(FileNotFoundError, match="bfcl_route_equivalence.csv"):
        generate_tables(bundle, protocol, analysis, tables)

