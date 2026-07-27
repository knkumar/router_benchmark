from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from router_benchmark.protocol.canonical import REQUIRED_FILES, assert_route_equivalent_outcomes, validate_bundle
from router_benchmark.protocol.bundle_writer import write_locked_bundle
from router_benchmark.protocol.protocol_tools import ProtocolValidationError


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _build_bundle(tmp_path: Path) -> tuple[Path, Path]:
    protocol = {
        "protocol_id": "fixture-v1", "routers": ["LiteLLM Router (live)", "Aurelio Semantic Router (live)", "RouteLLM (live)", "vLLM Semantic Router (live)"],
        "candidates": ["cheap-small", "mid-general", "strong-frontier"], "calibration_policy": "out_of_the_box_only",
        "cross_benchmark_decision": "weighted_success_reported_separately_from_cost",
        "baselines": [
            {"name": "Always-Cheapest Baseline (live)", "candidate_policy": "always cheap-small", "randomization": "deterministic", "routing_seeds": [], "cascade_order": [], "stopping_rule": "fixture", "fallback_behavior": "none"},
            {"name": "Always-Strongest Baseline (live)", "candidate_policy": "always strong-frontier", "randomization": "deterministic", "routing_seeds": [], "cascade_order": [], "stopping_rule": "fixture", "fallback_behavior": "none"},
        ],
        "analysis_paired_comparisons": [["LiteLLM Router (live)", "Aurelio Semantic Router (live)"], ["LiteLLM Router (live)", "RouteLLM (live)"], ["LiteLLM Router (live)", "vLLM Semantic Router (live)"], ["Aurelio Semantic Router (live)", "RouteLLM (live)"], ["Aurelio Semantic Router (live)", "vLLM Semantic Router (live)"], ["RouteLLM (live)", "vLLM Semantic Router (live)"]],
        "pricing": {"as_of": "2026-07-02"}, "model_snapshots": {candidate: "fixture" for candidate in ["cheap-small", "mid-general", "strong-frontier"]},
        "grader_versions": {benchmark: "fixture" for benchmark in ["RouterBench (live)", "BFCL v4 (live)", "tau2-bench (live)", "WebArena (live)"]},
        "benchmarks": {},
        "execution_budget": {"estimated_api_usd": "1", "estimated_infrastructure_usd": "0", "estimated_wall_time": "1 minute", "stopping_rule": "fixture", "approval_status": "approved"},
    }
    for benchmark in protocol["grader_versions"]:
        task_ids = [f"{benchmark}-task"]
        protocol["benchmarks"][benchmark] = {"subset_id": benchmark, "task_ids": task_ids, "task_id_sha256": hashlib.sha256(task_ids[0].encode()).hexdigest(), "router_trials_per_task": 2, "routing_seed_count": 2, "outcome_replicates_per_task_candidate": 1, "total_route_rows": 8, "total_outcome_rows": 3}
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol), encoding="utf-8")
    bundle = tmp_path / "bundle"; bundle.mkdir()
    routers = ["r1", "r2", "r3", "r4"]
    candidates = protocol["candidates"]
    candidate_rows = []
    trace_rows = []
    for benchmark, entry in protocol["benchmarks"].items():
        task = entry["task_ids"][0]
        for candidate in candidates:
            digest = f"trace-{benchmark}-{candidate}"
            trace_rows.append({"digest": digest})
            candidate_rows.append({"benchmark_id": benchmark, "task_id": task, "candidate_id": candidate, "outcome_replicate": "0", "execution_seed": "0", "grader_version": "fixture", "raw_trace_digest": digest, "success": "true", "provider_generation_usd": "1.0", "fallback_generation_usd": "0.0", "model_api_cost_usd": "1.0", "generation_latency_ms": "1", "failure_status": "none", "cache_flag": "false", "model_version": "fixture", "pricing_snapshot": "fixture"})
    _write_csv(bundle / "candidate_outcomes.csv", candidate_rows)
    routes = []; outcomes = []
    for benchmark, entry in protocol["benchmarks"].items():
        task = entry["task_ids"][0]
        for router in routers:
            for seed in range(2):
                candidate = candidates[seed % len(candidates)]
                route = {"router_config_id": router, "benchmark_id": benchmark, "task_id": task, "routing_seed": str(seed), "selected_candidate": candidate, "confidence": "1", "fallback_path": "none", "decision_latency_ms": "1", "router_service_usd": "0", "package_version": "fixture", "configuration_digest": "fixture", "route_vector_hash": f"{router}-{benchmark}-{seed}"}
                routes.append(route)
                outcomes.append({**{key: route[key] for key in ("router_config_id", "benchmark_id", "task_id", "routing_seed")}, "outcome_replicate": "0", "candidate_outcome_key": f"{benchmark}|{task}|{candidate}|0"})
    _write_csv(bundle / "routes.csv", routes); _write_csv(bundle / "outcomes.csv", outcomes); _write_csv(bundle / "results.csv", outcomes)
    router_names = protocol["routers"]
    (bundle / "router_configs.json").write_text(json.dumps({router: {"router_name": name} for router, name in zip(routers, router_names)}), encoding="utf-8")
    (bundle / "traces.jsonl").write_text("\n".join(json.dumps(row) for row in trace_rows) + "\n", encoding="utf-8")
    (bundle / "provenance.json").write_text(json.dumps({"excluded_historical_artifacts": [{"path": "paper1_live_v3", "reason": "router-specific historical outcomes"}]}), encoding="utf-8")
    benchmark_counts = {
        benchmark: {
            "unique_tasks": 1, "router_trials_per_task": 2, "routing_seed_count": 2,
            "outcome_replicates_per_task_candidate": 1, "total_route_rows": 8,
            "total_outcome_rows": 3, "subset_id": benchmark,
        }
        for benchmark in protocol["benchmarks"]
    }
    (bundle / "manifest.json").write_text(json.dumps({"protocol_id": "fixture-v1", "bundle_status": "locked", "router_config_ids": routers, "benchmark_counts": benchmark_counts}), encoding="utf-8")
    checksums = {filename: hashlib.sha256((bundle / filename).read_bytes()).hexdigest() for filename in REQUIRED_FILES - {"checksums.sha256"}}
    (bundle / "checksums.sha256").write_text("".join(f"{digest} {name}\n" for name, digest in sorted(checksums.items())), encoding="utf-8")
    return bundle, protocol_path


def test_canonical_bundle_schema_accepts_complete_locked_bundle(tmp_path: Path) -> None:
    bundle, protocol = _build_bundle(tmp_path)
    validate_bundle(bundle, protocol)


def test_bundle_writer_finalizes_a_canonical_locked_bundle(tmp_path: Path) -> None:
    source_bundle, protocol_path = _build_bundle(tmp_path)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    candidate_outcomes = list(csv.DictReader((source_bundle / "candidate_outcomes.csv").open(encoding="utf-8")))
    routes = list(csv.DictReader((source_bundle / "routes.csv").open(encoding="utf-8")))
    traces = [json.loads(line) for line in (source_bundle / "traces.jsonl").read_text(encoding="utf-8").splitlines()]
    written = write_locked_bundle(
        tmp_path / "finalized",
        protocol,
        router_configs=json.loads((source_bundle / "router_configs.json").read_text(encoding="utf-8")),
        candidate_outcomes=candidate_outcomes,
        routes=routes,
        traces=traces,
        provenance=json.loads((source_bundle / "provenance.json").read_text(encoding="utf-8")),
    )
    validate_bundle(written, protocol_path)
    assert {path.name for path in written.iterdir()} == REQUIRED_FILES


def test_canonical_bundle_schema_rejects_cache_derived_cell(tmp_path: Path) -> None:
    bundle, protocol = _build_bundle(tmp_path)
    content = (bundle / "candidate_outcomes.csv").read_text(encoding="utf-8").replace(",false,fixture,fixture", ",true,fixture,fixture", 1)
    (bundle / "candidate_outcomes.csv").write_text(content, encoding="utf-8")
    checksums = {filename: hashlib.sha256((bundle / filename).read_bytes()).hexdigest() for filename in REQUIRED_FILES - {"checksums.sha256"}}
    (bundle / "checksums.sha256").write_text("".join(f"{digest} {name}\n" for name, digest in sorted(checksums.items())), encoding="utf-8")
    with pytest.raises(ProtocolValidationError, match="cache-derived"):
        validate_bundle(bundle, protocol)


def test_canonical_bundle_schema_rejects_route_to_wrong_candidate_outcome(tmp_path: Path) -> None:
    bundle, protocol = _build_bundle(tmp_path)
    rows = list(csv.DictReader((bundle / "outcomes.csv").open(encoding="utf-8")))
    rows[0]["candidate_outcome_key"] = rows[0]["candidate_outcome_key"].replace("cheap-small", "mid-general")
    _write_csv(bundle / "outcomes.csv", rows)
    _write_csv(bundle / "results.csv", rows)
    checksums = {filename: hashlib.sha256((bundle / filename).read_bytes()).hexdigest() for filename in REQUIRED_FILES - {"checksums.sha256"}}
    (bundle / "checksums.sha256").write_text("".join(f"{digest} {name}\n" for name, digest in sorted(checksums.items())), encoding="utf-8")
    with pytest.raises(ProtocolValidationError, match="selected candidate"):
        validate_bundle(bundle, protocol)


def test_canonical_bundle_schema_rejects_negative_cost(tmp_path: Path) -> None:
    bundle, protocol = _build_bundle(tmp_path)
    rows = list(csv.DictReader((bundle / "candidate_outcomes.csv").open(encoding="utf-8")))
    rows[0]["provider_generation_usd"] = "-1.0"
    _write_csv(bundle / "candidate_outcomes.csv", rows)
    checksums = {filename: hashlib.sha256((bundle / filename).read_bytes()).hexdigest() for filename in REQUIRED_FILES - {"checksums.sha256"}}
    (bundle / "checksums.sha256").write_text("".join(f"{digest} {name}\n" for name, digest in sorted(checksums.items())), encoding="utf-8")
    with pytest.raises(ProtocolValidationError, match="negative cost"):
        validate_bundle(bundle, protocol)


def test_canonical_bundle_schema_rejects_nonfinite_cost(tmp_path: Path) -> None:
    bundle, protocol = _build_bundle(tmp_path)
    rows = list(csv.DictReader((bundle / "candidate_outcomes.csv").open(encoding="utf-8")))
    rows[0]["provider_generation_usd"] = "nan"
    _write_csv(bundle / "candidate_outcomes.csv", rows)
    checksums = {filename: hashlib.sha256((bundle / filename).read_bytes()).hexdigest() for filename in REQUIRED_FILES - {"checksums.sha256"}}
    (bundle / "checksums.sha256").write_text("".join(f"{digest} {name}\n" for name, digest in sorted(checksums.items())), encoding="utf-8")
    with pytest.raises(ProtocolValidationError, match="negative cost"):
        validate_bundle(bundle, protocol)


def test_canonical_bundle_schema_rejects_negative_router_service_cost(tmp_path: Path) -> None:
    bundle, protocol = _build_bundle(tmp_path)
    rows = list(csv.DictReader((bundle / "routes.csv").open(encoding="utf-8")))
    rows[0]["router_service_usd"] = "-0.01"
    _write_csv(bundle / "routes.csv", rows)
    checksums = {filename: hashlib.sha256((bundle / filename).read_bytes()).hexdigest() for filename in REQUIRED_FILES - {"checksums.sha256"}}
    (bundle / "checksums.sha256").write_text("".join(f"{digest} {name}\n" for name, digest in sorted(checksums.items())), encoding="utf-8")
    with pytest.raises(ProtocolValidationError, match="negative cost"):
        validate_bundle(bundle, protocol)


def test_canonical_bundle_schema_rejects_manifest_count_mismatch(tmp_path: Path) -> None:
    bundle, protocol = _build_bundle(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    manifest["benchmark_counts"]["BFCL v4 (live)"]["unique_tasks"] = 25
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    checksums = {filename: hashlib.sha256((bundle / filename).read_bytes()).hexdigest() for filename in REQUIRED_FILES - {"checksums.sha256"}}
    (bundle / "checksums.sha256").write_text("".join(f"{digest} {name}\n" for name, digest in sorted(checksums.items())), encoding="utf-8")
    with pytest.raises(ProtocolValidationError, match="manifest count mismatch"):
        validate_bundle(bundle, protocol)


def test_canonical_bundle_schema_rejects_out_of_scope_router(tmp_path: Path) -> None:
    bundle, protocol = _build_bundle(tmp_path)
    configs = json.loads((bundle / "router_configs.json").read_text(encoding="utf-8"))
    configs["r1"]["router_name"] = "NVIDIA AI Blueprint LLM Router (live)"
    (bundle / "router_configs.json").write_text(json.dumps(configs), encoding="utf-8")
    checksums = {filename: hashlib.sha256((bundle / filename).read_bytes()).hexdigest() for filename in REQUIRED_FILES - {"checksums.sha256"}}
    (bundle / "checksums.sha256").write_text("".join(f"{digest} {name}\n" for name, digest in sorted(checksums.items())), encoding="utf-8")
    with pytest.raises(ProtocolValidationError, match="undeclared router"):
        validate_bundle(bundle, protocol)


def test_canonical_bundle_schema_rejects_untracked_rerun_file(tmp_path: Path) -> None:
    bundle, protocol = _build_bundle(tmp_path)
    (bundle / "router_specific_rerun.csv").write_text("not a canonical artifact\n", encoding="utf-8")
    with pytest.raises(ProtocolValidationError, match="untracked files"):
        validate_bundle(bundle, protocol)


def test_router_specific_records_fail_route_equivalence_audit() -> None:
    rows = [
        {"benchmark_name": "BFCL v4 (live)", "task_id": "task", "trial": "0", "selected_candidate": "cheap-small", "success": "true", "cost_usd": "0.1", "latency_ms": "1"},
        {"benchmark_name": "BFCL v4 (live)", "task_id": "task", "trial": "0", "selected_candidate": "cheap-small", "success": "false", "cost_usd": "0.1", "latency_ms": "1"},
    ]
    with pytest.raises(ProtocolValidationError, match="router-specific outcome"):
        assert_route_equivalent_outcomes(rows)
