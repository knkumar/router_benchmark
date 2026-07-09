"""Phase 7c live entry point: corrected rerun of the WebArena expanded
sample (16 tasks/router). Phase 7 (8 tasks) and Phase 7b (16 tasks) are
both invalid: WebArenaLive.generate_tasks() only set metadata["intent"],
but 5 of the 6 router adapters (all except LiteLLM Router, which is
difficulty-threshold-only) key off metadata["prompt"]/["user_msg"] to make
their real routing decision. Missing that field, they silently used their
fixed no-signal fallback for every single WebArena task in both prior
phases -- confirmed by fallback_used=1.0 for those 5 routers in phase7 and
phase7b's results.csv, and by every router picking the identical tier
(mid-general) on every task. webarena_live.py now also populates
metadata["prompt"]/["user_msg"] with the real task intent; a direct
route()-only smoke test (no browser/API cost) confirmed routers now
genuinely diverge (e.g. NVIDIA Blueprint alternates cheap-small/
strong-frontier depending on task content).

    python -m router_benchmark.live.run_live_phase7c
"""

from __future__ import annotations

from router_benchmark.live.live_routers import AurelioSemanticRouterLive, LiteLLMRouterLive
from router_benchmark.live.llmrouter_live import LLMRouterLive
from router_benchmark.live.nvidia_blueprint_live import NVIDIABlueprintRouterLive
from router_benchmark.live.routellm_live import RouteLLMLive
from router_benchmark.live.run_common import run_live_phase
from router_benchmark.live.vllm_sr_live import VLLMSemanticRouterLive
from router_benchmark.live.webarena_live import WebArenaLive


def main() -> None:
    routers = [
        LiteLLMRouterLive(),
        AurelioSemanticRouterLive(),
        RouteLLMLive(),
        LLMRouterLive(),
        NVIDIABlueprintRouterLive(),
        VLLMSemanticRouterLive(),
    ]
    benchmarks = [WebArenaLive(n_tasks=16, sites=("gitlab", "shopping"))]
    run_live_phase("phase7c", routers, benchmarks, seed=1234, n_trials=1)


if __name__ == "__main__":
    main()
