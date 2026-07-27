"""Router-free candidate execution for a bounded diagnostic dry run.

The caller supplies benchmark adapters after the dry-run protocol has passed
preflight.  Each selected task is scored once for every frozen candidate tier;
no router is constructed or replayed in this module.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import zlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from router_benchmark.interfaces import Benchmark, RouteDecision, Task
from router_benchmark.protocol.candidate_runner import CandidateCell, CandidateStageRunner
from router_benchmark.protocol.protocol_tools import load_yaml, validate_rebuild_protocol
from router_benchmark.scripts.preflight_dry_run import (
    candidate_reservations_from_protocol,
    external_metered_reservation_from_protocol,
    validate_dry_run_protocol,
)


class AdapterFactory(Protocol):
    """Build adapters only after the dry-run protocol has been validated."""

    def __call__(self, dry_protocol: Mapping[str, Any]) -> Mapping[str, Benchmark]: ...


def _seed_for(cell: CandidateCell) -> int:
    return zlib.crc32("|".join(cell.key).encode("utf-8")) & 0xFFFFFFFF


def _mapping_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class FrozenCandidateExecutor:
    """Force frozen candidates through benchmark adapters without router replay."""

    def __init__(self, protocol: Mapping[str, Any], adapters: Mapping[str, Benchmark]) -> None:
        expected = set(protocol["benchmarks"])
        if set(adapters) != expected:
            raise ValueError("adapter mapping must cover exactly the dry-run benchmarks")
        self.protocol = protocol
        self.adapters = dict(adapters)
        self.tasks_by_benchmark = {
            benchmark_id: self._select_frozen_tasks(benchmark_id, adapter)
            for benchmark_id, adapter in self.adapters.items()
        }

    def _select_frozen_tasks(self, benchmark_id: str, adapter: Benchmark) -> dict[str, Task]:
        expected_ids = [str(task_id) for task_id in self.protocol["benchmarks"][benchmark_id]["task_ids"]]
        available = list(adapter.generate_tasks(rng=None))
        by_id = {str(task.task_id): task for task in available}
        if len(by_id) != len(available):
            raise ValueError(f"adapter returned duplicate task IDs for {benchmark_id}")
        missing = [task_id for task_id in expected_ids if task_id not in by_id]
        if missing:
            raise ValueError(f"adapter is missing frozen task IDs for {benchmark_id}: {missing}")
        return {task_id: by_id[task_id] for task_id in expected_ids}

    def execute(self, cell: CandidateCell) -> Mapping[str, Any]:
        task = self.tasks_by_benchmark[cell.benchmark_id][cell.task_id]
        candidate_names = {candidate.name for candidate in task.candidates}
        if cell.candidate_id not in candidate_names:
            raise ValueError(f"frozen candidate {cell.candidate_id} is unavailable for {cell.benchmark_id}/{cell.task_id}")
        decision = RouteDecision(
            selected_candidate=cell.candidate_id,
            confidence=1.0,
            fallback_used=False,
            metadata={"candidate_execution": "forced_frozen_candidate", "router_replay": False},
        )
        result = dict(self.adapters[cell.benchmark_id].score(task, decision, np.random.default_rng(_seed_for(cell))))
        latency_ms = float(result.get("generation_latency_ms", result.get("latency_ms", 0.0)) or 0.0)
        if not math.isfinite(latency_ms) or latency_ms < 0:
            latency_ms = 0.0
        return {
            "success": bool(result.get("success", False)),
            "provider_generation_usd": result.get("provider_generation_usd", result.get("cost_usd", 0.0)),
            "fallback_generation_usd": result.get("fallback_generation_usd", 0.0),
            "model_api_cost_usd": result.get("model_api_cost_usd", result.get("cost_usd", 0.0)),
            "external_metered_usd": result.get("external_metered_usd", 0.0),
            "generation_latency_ms": latency_ms,
            "failure_status": result.get("failure_status", "none"),
            "raw_trace": {
                "benchmark_id": cell.benchmark_id,
                "task_id": cell.task_id,
                "candidate_id": cell.candidate_id,
                "outcome_replicate": cell.outcome_replicate,
                "forced_candidate": True,
                "adapter_result": result,
            },
        }


def run_dry_candidate_stage(
    dry_protocol: Mapping[str, Any],
    frozen_protocol: Mapping[str, Any],
    *,
    stage_dir: Path,
    adapters: Mapping[str, Benchmark],
    estimate_cost_usd: Callable[[CandidateCell], float] | None = None,
    resume: bool = False,
) -> list[dict[str, str]]:
    """Preflight, then execute only the frozen diagnostic candidate cells."""
    validate_rebuild_protocol(frozen_protocol)
    validate_dry_run_protocol(dict(dry_protocol), dict(frozen_protocol))
    reservations = candidate_reservations_from_protocol(dict(dry_protocol))
    external_metered_reserve = external_metered_reservation_from_protocol(dict(dry_protocol))
    executor = FrozenCandidateExecutor(dry_protocol, adapters)
    runner = CandidateStageRunner(
        stage_dir,
        dry_protocol,
        budget_cap_usd=float(dry_protocol["dry_run_budget_cap_usd"]),
        reservation_by_cell=reservations,
        external_reserved_usd=external_metered_reserve,
        stage_metadata={
            "dry_protocol_sha256": _mapping_digest(dry_protocol),
            "frozen_protocol_sha256": _mapping_digest(frozen_protocol),
            "reservation_source": "dry_run_cost_reservations",
        },
    )
    estimator = estimate_cost_usd or (lambda _cell: 0.0)
    return runner.run(executor.execute, estimator, resume=resume)


def _load_factory(spec: str) -> AdapterFactory:
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("--adapter-factory must use module:callable syntax")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise ValueError("--adapter-factory must name a callable")
    return factory


def _assert_required_metered_environment(skip_benchmarks: set[str] | None = None) -> None:
    skip = skip_benchmarks or set()
    missing = [name for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY") if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"missing required provider environment variables: {', '.join(missing)}")
    package_root = Path(__file__).resolve().parents[1]
    
    required_paths = []
    if "tau2-bench (live)" not in skip:
        required_paths.append(package_root / "live" / "tau2env" / "tau2-bench")
    if "WebArena (live)" not in skip:
        required_paths.append(Path.home() / ".local" / "share" / "router_bench_vendor" / "webarena")
        
    absent = [str(path) for path in required_paths if not path.exists()]
    if absent:
        raise RuntimeError(
            f"missing required local harness paths for enabled benchmarks: {', '.join(absent)}"
        )
    print("Metered environment: satisfied")
    parser.add_argument("--dry-protocol", type=Path, required=True)
    parser.add_argument("--frozen-protocol", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--adapter-factory", required=True, help="module:callable, invoked only after preflight")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    dry_protocol = load_yaml(args.dry_protocol)
    frozen_protocol = load_yaml(args.frozen_protocol)
    validate_rebuild_protocol(frozen_protocol)
    validate_dry_run_protocol(dry_protocol, frozen_protocol)
    _assert_required_metered_environment() # We don't expose stop_before_benchmark in dry run yet.
    adapters = _load_factory(args.adapter_factory)(dry_protocol)
    rows = run_dry_candidate_stage(
        dry_protocol,
        frozen_protocol,
        stage_dir=args.stage_dir,
        adapters=adapters,
        resume=args.resume,
    )
    print(f"Dry candidate stage wrote {len(rows)} rows to {args.stage_dir}.")


if __name__ == "__main__":
    main()
