"""Phase 4c live entry point: corrected rerun of Phase 4b (SWE-bench
Verified, 4 remaining routers: Aurelio Semantic Router, vLLM Semantic
Router, NVIDIA AI Blueprint LLM Router, LLMRouter). Phase 4b is invalid:
swebench_live.py shelled out to a bare "python3" that resolved via PATH
to an interpreter without the `swebench` package installed, so every
task failed with ModuleNotFoundError before any real Docker evaluation
ran -- confirmed via traces.jsonl (returncode=1, stderr "No module named
'swebench'", $0 cost) for all 8 of Phase 4b's task-trials. Fixed by
pinning the subprocess to router_benchmark/.venv/bin/python3 explicitly.

    python -m router_benchmark.live.run_live_phase4c
"""

from __future__ import annotations

from router_benchmark.live.live_routers import AurelioSemanticRouterLive
from router_benchmark.live.llmrouter_live import LLMRouterLive
from router_benchmark.live.nvidia_blueprint_live import NVIDIABlueprintRouterLive
from router_benchmark.live.run_common import run_live_phase
from router_benchmark.live.swebench_live import build_swebench_live
from router_benchmark.live.vllm_sr_live import VLLMSemanticRouterLive


def main() -> None:
    routers = [
        AurelioSemanticRouterLive(),
        VLLMSemanticRouterLive(),
        NVIDIABlueprintRouterLive(),
        LLMRouterLive(),
    ]
    benchmarks = [build_swebench_live(n_tasks=2)]
    run_live_phase("phase4c", routers, benchmarks, seed=1234, n_trials=1)


if __name__ == "__main__":
    main()
