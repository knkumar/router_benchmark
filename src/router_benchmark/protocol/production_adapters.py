"""Build the four allowlisted adapters for the Paper 1 diagnostic run.

Construction is deliberately separate from execution. Importing this module
does not create a provider client or send a request; the live adapters are
constructed only after dry-run preflight accepts the protocol.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from router_benchmark.interfaces import RouteDecision, Router, Task
from router_benchmark.scripts.preflight_dry_run import request_limits_from_protocol
from router_benchmark.scripts.preflight_full_run import request_limits_from_full_protocol


_BENCHMARKS = {
    "RouterBench (live)",
    "BFCL v4 (live)",
    "tau2-bench (live)",
    "WebArena (live)",
}


def _task_ids_from_protocol(protocol: Mapping[str, Any], *, label: str) -> dict[str, list[str]]:
    benchmarks = protocol.get("benchmarks")
    if not isinstance(benchmarks, Mapping) or set(benchmarks) != _BENCHMARKS:
        raise ValueError(f"{label} protocol must declare exactly the four allowlisted benchmarks")
    task_ids: dict[str, list[str]] = {}
    for name in _BENCHMARKS:
        entry = benchmarks[name]
        if not isinstance(entry, Mapping) or not isinstance(entry.get("task_ids"), list):
            raise ValueError(f"{name} must declare task_ids")
        task_ids[name] = entry["task_ids"]
    return task_ids


def _build_adapters(protocol: Mapping[str, Any], request_limits: Mapping[str, Mapping[str, int]], *, label: str, skip_benchmarks: set[str] | None = None) -> dict[str, object]:
    task_ids = _task_ids_from_protocol(protocol, label=label)
    skip = skip_benchmarks or set()

    from router_benchmark.live.live_benchmarks import BFCLLive, RouterBenchLive
    adapters: dict[str, object] = {}
    
    if "RouterBench (live)" not in skip:
        adapters["RouterBench (live)"] = RouterBenchLive(
            n_tasks=60, task_ids=task_ids["RouterBench (live)"]
        )
    if "BFCL v4 (live)" not in skip:
        adapters["BFCL v4 (live)"] = BFCLLive(
            n_tasks=30, task_ids=task_ids["BFCL v4 (live)"]
        )
        
    if "tau2-bench (live)" not in skip:
        from router_benchmark.live.tau2_live import Tau2BenchLive
        adapters["tau2-bench (live)"] = Tau2BenchLive(
            task_ids=task_ids["tau2-bench (live)"],
            max_steps=request_limits["tau2-bench (live)"]["max_steps"],
            max_output_tokens=request_limits["tau2-bench (live)"]["max_output_tokens_per_call"],
            require_cost_ledger=True,
        )
    if "WebArena (live)" not in skip:
        from router_benchmark.live.webarena_live import WebArenaLive
        adapters["WebArena (live)"] = WebArenaLive(
            task_ids=task_ids["WebArena (live)"],
            max_steps=request_limits["WebArena (live)"]["max_steps"],
            max_output_tokens=request_limits["WebArena (live)"]["max_output_tokens_per_call"],
            require_trace_cost=True,
        )

    expected_names = _BENCHMARKS - skip
    if {adapter.name for adapter in adapters.values()} != expected_names:
        raise ValueError(f"live adapter names do not match the {label} allowlist (minus skips)")
    return adapters


def build_dry_run_adapters(dry_protocol: Mapping[str, Any], skip_benchmarks: set[str] | None = None) -> dict[str, object]:
    """Construct exactly the frozen four adapters with their diagnostic IDs.

    RouterBench and BFCL task identifiers encode positions from the original
    60- and 30-task frozen samples. Their historical sample sizes are retained
    while the adapters return only the requested dry-run IDs.
    """
    return _build_adapters(
        dry_protocol,
        request_limits_from_protocol(dict(dry_protocol)),
        label="dry-run",
        skip_benchmarks=skip_benchmarks,
    )


def build_full_run_adapters(protocol: Mapping[str, Any], skip_benchmarks: set[str] | None = None) -> dict[str, object]:
    """Construct exactly the four allowlisted adapters for the full rebuild."""
    return _build_adapters(
        protocol,
        request_limits_from_full_protocol(dict(protocol)),
        label="full-run",
        skip_benchmarks=skip_benchmarks,
    )


class _EstimatedMeteredRouter(Router):
    """Attach separately accounted router-service spend to live router rows.

    Some third-party router packages call provider APIs internally but do not
    expose provider usage blocks through their public routing result.  For the
    diagnostic dry run, route replay records a conservative reservation debit
    in ``router_service_usd`` instead of folding any router-service spend into
    candidate model API cost.
    """

    def __init__(self, router: Router, *, router_service_usd: float, metering_basis: str) -> None:
        self.router = router
        self.name = router.name
        self.router_service_usd = float(router_service_usd)
        self.metering_basis = metering_basis

    def route(self, task: Task, context: dict, rng) -> RouteDecision:
        try:
            decision = self.router.route(task, context, rng)
        except Exception as exc:
            return RouteDecision(
                selected_candidate="mid-general",
                confidence=0.0,
                fallback_used=True,
                metadata={
                    "router_service_usd": self.router_service_usd,
                    "router_service_metering_basis": self.metering_basis,
                    "fallback_path": "router_service_error",
                    "router_service_error": f"{type(exc).__name__}: {exc}",
                },
            )
        metadata = dict(decision.metadata)
        metadata.setdefault("router_service_usd", self.router_service_usd)
        metadata.setdefault("router_service_metering_basis", self.metering_basis)
        return RouteDecision(
            selected_candidate=decision.selected_candidate,
            confidence=decision.confidence,
            fallback_used=decision.fallback_used,
            metadata=metadata,
        )


def build_dry_run_routers(dry_protocol: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, Router]]:
    """Construct the four allowlisted live routers for dry-run route replay."""
    router_names = list(dry_protocol.get("routers", []))
    expected = [
        "LiteLLM Router (live)",
        "Aurelio Semantic Router (live)",
        "RouteLLM (live)",
        "vLLM Semantic Router (live)",
    ]
    if router_names != expected:
        raise ValueError("dry protocol must declare the four allowlisted routers in canonical order")

    from router_benchmark.live.live_routers import AurelioSemanticRouterLive, LiteLLMRouterLive
    from router_benchmark.live.routellm_live import RouteLLMLive
    from router_benchmark.live.vllm_sr_live import VLLMSemanticRouterLive

    router_specs: list[tuple[str, Router, float, str, str]] = [
        (
            "litellm-router-live",
            LiteLLMRouterLive(),
            0.0,
            "LiteLLM cost-based deployment selection; no provider call in route()",
            "litellm cost-based-routing",
        ),
        (
            "aurelio-semantic-router-live",
            AurelioSemanticRouterLive(),
            0.01,
            "conservative debit for semantic-router OpenAI embedding route call",
            "semantic-router OpenAIEncoder text-embedding-3-small",
        ),
        (
            "routellm-live",
            RouteLLMLive(),
            0.01,
            "conservative debit for RouteLLM OpenAI embedding route call",
            "routellm sw_ranking text-embedding-3-small",
        ),
        (
            "vllm-semantic-router-live",
            VLLMSemanticRouterLive(),
            0.05,
            "conservative debit for vLLM Semantic Router probe completion capped by adapter",
            "vllm semantic-router MoM probe",
        ),
    ]
    router_configs = {
        router_id: {
            "router_name": router.name,
            "package_version": package_version,
            "router_service_usd_per_route": per_route_cost,
            "router_service_metering_basis": metering_basis,
            **(
                {
                    "embedding_request_timeout_s": router.embedding_request_timeout_s,
                    "embedding_max_retries": router.embedding_max_retries,
                }
                if isinstance(router, (AurelioSemanticRouterLive, RouteLLMLive))
                else {}
            ),
        }
        for router_id, router, per_route_cost, metering_basis, package_version in router_specs
    }
    routers = {
        router_id: _EstimatedMeteredRouter(
            router,
            router_service_usd=per_route_cost,
            metering_basis=metering_basis,
        )
        for router_id, router, per_route_cost, metering_basis, _package_version in router_specs
    }
    return router_configs, routers


def build_full_run_routers(protocol: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, Router]]:
    """Construct the four allowlisted live routers for full route replay."""
    return build_dry_run_routers(protocol)
