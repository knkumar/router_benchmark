"""Router-free candidate execution for the approved full Paper 1 rebuild."""

from __future__ import annotations

import argparse
import importlib
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from router_benchmark.protocol.candidate_runner import CandidateCell, CandidateStageRunner
from router_benchmark.protocol.dry_run_execution import FrozenCandidateExecutor
from router_benchmark.protocol.protocol_tools import load_yaml
from router_benchmark.scripts.preflight_full_run import (
    RESERVATION_FIELD,
    candidate_reservations_from_full_protocol,
    external_metered_reservation_from_full_protocol,
    validate_full_run_protocol,
)


class AdapterFactory(Protocol):
    def __call__(self, protocol: Mapping[str, Any]) -> Mapping[str, object]: ...


def run_full_candidate_stage(
    protocol: Mapping[str, Any],
    *,
    stage_dir: Path,
    adapters: Mapping[str, object],
    estimate_cost_usd: Callable[[CandidateCell], float] | None = None,
    resume: bool = False,
    stop_before_benchmark: str | None = None,
) -> list[dict[str, str]]:
    """Preflight, then execute only the frozen full candidate matrix."""
    validate_full_run_protocol(dict(protocol))
    reservations = candidate_reservations_from_full_protocol(protocol)
    external_metered_reserve = external_metered_reservation_from_full_protocol(protocol)
    executor = FrozenCandidateExecutor(protocol, adapters)  # type: ignore[arg-type]
    runner = CandidateStageRunner(
        stage_dir,
        protocol,
        budget_cap_usd=float(protocol[RESERVATION_FIELD]["total_cap_usd"]),
        reservation_by_cell=reservations,
        external_reserved_usd=external_metered_reserve,
        stage_metadata={
            "protocol_id": protocol["protocol_id"],
            "reservation_source": RESERVATION_FIELD,
            "diagnostic_only": False,
        },
    )
    estimator = estimate_cost_usd or (lambda _cell: 0.0)
    return runner.run(
        executor.execute,
        estimator,
        resume=resume,
        stop_before_benchmark=stop_before_benchmark,
    )


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
    if os.environ.get("ROUTER_BENCHMARK_LLM_CACHE", "").lower() in {"1", "true", "yes", "on"}:
        raise RuntimeError("ROUTER_BENCHMARK_LLM_CACHE must be disabled for canonical full execution")
    if os.environ.get("ROUTER_BENCHMARK_TAU2_USE_RESULT_CACHE", "").lower() in {"1", "true", "yes", "on"}:
        raise RuntimeError("ROUTER_BENCHMARK_TAU2_USE_RESULT_CACHE must be disabled for canonical full execution")
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

def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--adapter-factory", required=True, help="module:callable, invoked only after preflight")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-before-benchmark")
    args = parser.parse_args(argv)
    protocol = load_yaml(args.protocol)
    validate_full_run_protocol(protocol)
    
    skip_benchmarks = ({args.stop_before_benchmark} if args.stop_before_benchmark else None)
    
    _assert_required_metered_environment(skip_benchmarks=skip_benchmarks)
    
    # We pass skip_benchmarks to factory
    factory = _load_factory(args.adapter_factory)
    try:
        adapters = factory(protocol, skip_benchmarks=skip_benchmarks)
    except TypeError:
        # fallback if factory doesn't support skips yet
        adapters = factory(protocol)
        
    rows = run_full_candidate_stage(
        protocol,
        stage_dir=args.stage_dir,
        adapters=adapters,
        resume=args.resume,
        stop_before_benchmark=args.stop_before_benchmark,
    )
    print(f"Full candidate stage wrote {len(rows)} rows to {args.stage_dir}.")


if __name__ == "__main__":
    main()
