"""Phase B (review-response) Task 4: router configuration fairness sweep.

Reviewer objection: the "complex routers add no value" result may be an
artifact of leaving each router at its shipped default threshold. This phase
sweeps the two configurable escalation thresholds and re-measures:

  - RouteLLM's escalation threshold (route to strong iff strong_winrate >=
    threshold; default 0.5). Lowering it should make RouteLLM escalate to
    strong-frontier more often, if its classifier signal is usable at all.
  - Aurelio Semantic Router's semantic score_threshold (a real
    semantic-router instance attribute; default 0.3). Lowering it should let
    more prompts match a tier instead of taking the mid-general fallback.

Both are swept on RouterBench (replay; free) and BFCL v4 (live), the two
benchmarks that carry this paper's default-threshold trial convention
(n=60 / n=30, 2 trials each). The default wide-gap pool (gpt-5.4-nano /
claude-sonnet-4-6 / claude-opus-4-8) is unchanged.

Implementation note: one base RouteLLMLive and one base
AurelioSemanticRouterLive are constructed once (each is expensive to build:
Arena battle datasets, OpenAI-embedded reference utterances). Because the
harness runs routers strictly sequentially (harness.py: for benchmark -> for
router -> for task), each thin wrapper safely sets the shared base's
threshold immediately before delegating route(). No adapter source is
modified; the base decision path is exactly the paper's.
"""

from router_benchmark.interfaces import Router
from router_benchmark.live.routellm_live import RouteLLMLive
from router_benchmark.live.live_routers import AurelioSemanticRouterLive
from router_benchmark.live.live_benchmarks import build_live_benchmarks
from router_benchmark.live.run_common import run_live_phase

ROUTELLM_THRESHOLDS = (0.5, 0.4, 0.3, 0.2, 0.1)
AURELIO_SCORE_THRESHOLDS = (0.30, 0.20, 0.10, 0.05)


class RouteLLMThresh(Router):
    """Shares one RouteLLMLive; applies its own escalation threshold per call."""

    def __init__(self, base: RouteLLMLive, threshold: float):
        self._base = base
        self.threshold = threshold
        self.name = f"RouteLLM (thr={threshold:.2f})"

    def route(self, task, context, rng):
        self._base.threshold = self.threshold
        return self._base.route(task, context, rng)


class AurelioThresh(Router):
    """Shares one AurelioSemanticRouterLive; applies its own score_threshold per call.

    semantic-router honors the PER-ROUTE score_threshold, not the router-level
    one (verified directly: setting only router.score_threshold=0.9 still
    matches a 0.65-scoring prompt, while setting each route.score_threshold=0.9
    correctly forces the fallback). We therefore set the threshold on every
    Route object, and the router-level attribute for good measure.
    """

    def __init__(self, base: AurelioSemanticRouterLive, score_threshold: float):
        self._base = base
        self.score_threshold = score_threshold
        self.name = f"Aurelio (score_thr={score_threshold:.2f})"

    def route(self, task, context, rng):
        self._base._router.score_threshold = self.score_threshold
        for rt in self._base._router.routes:
            rt.score_threshold = self.score_threshold
        return self._base.route(task, context, rng)


def main() -> None:
    rl_base = RouteLLMLive()
    au_base = AurelioSemanticRouterLive()
    routers = [RouteLLMThresh(rl_base, t) for t in ROUTELLM_THRESHOLDS] + [
        AurelioThresh(au_base, s) for s in AURELIO_SCORE_THRESHOLDS
    ]
    benchmarks = build_live_benchmarks(routerbench_n=60, bfcl_n=30)
    run_live_phase(
        "threshold_sweep_v1",
        routers,
        benchmarks,
        seed=1234,
        n_trials=2,
        extra_manifest={
            "purpose": "router config fairness: RouteLLM escalation threshold + "
            "Aurelio semantic score_threshold sweep on RouterBench + BFCL v4",
            "routellm_thresholds": list(ROUTELLM_THRESHOLDS),
            "aurelio_score_thresholds": list(AURELIO_SCORE_THRESHOLDS),
        },
    )


if __name__ == "__main__":
    main()

