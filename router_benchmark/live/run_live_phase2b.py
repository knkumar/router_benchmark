"""Phase 2b live entry point: vLLM Semantic Router (real Docker/Envoy
service, see live/vllm_sr/config.yaml) against RouterBench + BFCL v4.

Requires the service already running:
    cd router_benchmark/live/vllm_sr
    VLLM_SR_PORT_OFFSET=10 vllm-sr serve --minimal --config config.yaml

    python -m router_benchmark.live.run_live_phase2b
"""

from __future__ import annotations

from router_benchmark.live.live_benchmarks import build_live_benchmarks
from router_benchmark.live.run_common import run_live_phase
from router_benchmark.live.vllm_sr_live import VLLMSemanticRouterLive


def main() -> None:
    routers = [VLLMSemanticRouterLive()]
    benchmarks = build_live_benchmarks(routerbench_n=60, bfcl_n=30)
    run_live_phase(
        "phase2b",
        routers,
        benchmarks,
        seed=1234,
        n_trials=2,
        extra_manifest={"vllm_sr_service": "vllm-sr v0.3.0, algorithm=automix, VLLM_SR_PORT_OFFSET=10"},
    )


if __name__ == "__main__":
    main()
