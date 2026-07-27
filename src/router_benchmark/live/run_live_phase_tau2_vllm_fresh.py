"""Fresh, cache-bypassed re-measurement of vLLM Semantic Router on tau2-bench.

Why this phase exists. The paper's vLLM Semantic Router tau2-bench number
(0.850, n=100) was 81% cache-served rather than executed. tau2_live.score()
consults live/tau2_cache.json keyed by f"{task_id}_{selected_candidate}",
with no router identity and no trial index, so any (task, tier) pair already
recorded by an earlier run is replayed verbatim. Trace audit of the phase
that produced the published number (phase11/traces.jsonl) shows vLLM's split
was 81 cached / 19 fresh: 36 cheap-small cached, 44 mid-general cached, 1
strong-frontier cached, and only its 19 strong-frontier tasks executed live.
The cache's contents (100 cheap-small, 97 mid-general, 1 strong-frontier)
match the other routers' tau2 routing exactly (LiteLLM and RouteLLM at 100%
cheap; Aurelio at 2/97/1), so those replayed outcomes were measured during
*other* routers' runs.

Why it matters. tau2 runs its agent at temperature=1, and a 5-task x 3-trial
cache-bypassed pilot measured outcome flips on 2 of 10 (router, task) cells,
so a replayed entry is one stochastic draw, not a stable value. The affected
number is one side of the paper's Proposition 1 rank-inversion instantiation
(vLLM 0.850 < Aurelio 0.910 on tau2-bench), so it should rest on an
independent measurement.

Scope: vLLM Semantic Router only, the full n=100 retail task set, 1 trial,
matching the paper's tau2 convention and seed. The other three routers'
tau2 numbers were already fully fresh (phase10/phase10_backup show 0
cache-served lines) and are deliberately not re-run. The runner script must
move live/tau2_cache.json aside for the duration; the cache is read-only in
tau2_live.py, so nothing repopulates it.
"""

from router_benchmark.live.run_common import run_live_phase
from router_benchmark.live.tau2_live import Tau2BenchLive
from router_benchmark.live.vllm_sr_live import VLLMSemanticRouterLive

N_TASKS = 100


def main() -> None:
    run_live_phase(
        "tau2_vllm_fresh_v1",
        [VLLMSemanticRouterLive()],
        [Tau2BenchLive(n_tasks=N_TASKS)],
        seed=1234,
        n_trials=1,
        extra_manifest={
            "purpose": "fresh cache-bypassed re-measurement of vLLM Semantic Router on "
            "tau2-bench; supersedes the 81%-cache-served number from phase11",
            "supersedes_phase": "phase11",
            "cache_bypassed": True,
            "n_tasks": N_TASKS,
        },
    )


if __name__ == "__main__":
    main()

