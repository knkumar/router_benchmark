"""Phase B Task 9 PILOT: does tau2-bench actually vary run to run?

The paper's tau2-bench numbers are single trial, so route stability and
run-to-run success variance are undefined there. Before paying for a full
multi-trial pass we measure whether repeated execution of the SAME (task,
tier) pair even changes the outcome. tau2 runs its agent at
temperature=1 (tau2_live.py passes --agent-llm-args '{"temperature": 1}'),
so some variance is expected, but its size is unmeasured.

IMPORTANT -- the result cache must be bypassed for this to measure anything.
tau2_live.score() consults live/tau2_cache.json keyed by
f"{task_id}_{selected_candidate}", with NO trial index, so a populated cache
returns a byte-identical result for every trial and would report exactly zero
variance as an artifact. The cache is read-only (nothing writes it), so the
runner script moves it aside for the duration and restores it afterwards.

Scope: two routers chosen to cover the two tiers the real routers actually
use on tau2-bench -- LiteLLM Router (cheap-small on 100% of tasks) and
Aurelio Semantic Router (mid-general fallback on ~97%) -- over the first 5
retail tasks, 3 trials each. That is 30 real tau2 simulations. Outcome
variance is a property of (task, tier, agent LLM) rather than of the router
wrapper, so two routers suffice to characterize both tiers.

Decision gate: if success flips across trials for a meaningful fraction of
(router, task) pairs, the full n=100 x 3 pass is worth its cost; if outcomes
are effectively deterministic, a multi-trial pass would only restate the
single-trial numbers and should be skipped.
"""

from router_benchmark.live.live_routers import AurelioSemanticRouterLive, LiteLLMRouterLive
from router_benchmark.live.tau2_live import Tau2BenchLive
from router_benchmark.live.run_common import run_live_phase

N_TASKS = 5
N_TRIALS = 3


def main() -> None:
    routers = [LiteLLMRouterLive(), AurelioSemanticRouterLive()]
    run_live_phase(
        "tau2_trials_pilot_v1",
        routers,
        [Tau2BenchLive(n_tasks=N_TASKS)],
        seed=1234,
        n_trials=N_TRIALS,
        extra_manifest={
            "purpose": "pilot: tau2-bench run-to-run success variance across repeated "
            "trials of the same (task, tier), with the result cache bypassed",
            "n_tasks": N_TASKS,
            "n_trials": N_TRIALS,
            "cache_bypassed": True,
        },
    )


if __name__ == "__main__":
    main()

