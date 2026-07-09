"""Phase B Task 8: candidate-pool ablation. Narrows the cheap/strong
capability gap by swapping cheap-small from gpt-5.4-nano to
claude-haiku-4-5 (Anthropic, $1.00/$5.00 per 1M input/output tokens,
chosen by the user 2026-07-05) -- mid-general and strong-frontier stay
unchanged. Scope: RouterBench + BFCL only (4 real routers, 2 trials
each), matching this paper's existing trial convention for those two
benchmarks -- tau2-bench/WebArena are excluded from this ablation.

Implementation note: router_benchmark.live.live_routers and
router_benchmark.live.live_benchmarks both do
`from router_benchmark.live.llm_client import CANDIDATE_TIERS, PRICING`
and look up tiers by name at call time (not at import time), so mutating
those dicts in place -- BEFORE importing/constructing any router or
benchmark below -- makes every adapter see the narrow-gap pool without
touching adapter source. This only affects this one-off process; it does
not change the wide-gap default for any other script.
"""

from __future__ import annotations

from router_benchmark.live import llm_client

llm_client.PRICING["claude-haiku-4-5"] = (1.00 / 1_000_000, 5.00 / 1_000_000)
llm_client._PROVIDER_OF["claude-haiku-4-5"] = "anthropic"
llm_client.CANDIDATE_TIERS["cheap-small"] = "claude-haiku-4-5"  # was gpt-5.4-nano

from router_benchmark.live.live_benchmarks import build_live_benchmarks  # noqa: E402
from router_benchmark.live.live_routers import AurelioSemanticRouterLive, LiteLLMRouterLive  # noqa: E402
from router_benchmark.live.routellm_live import RouteLLMLive  # noqa: E402
from router_benchmark.live.run_common import run_live_phase  # noqa: E402
from router_benchmark.live.vllm_sr_live import VLLMSemanticRouterLive  # noqa: E402


def main() -> None:
    routers = [LiteLLMRouterLive(), AurelioSemanticRouterLive(), RouteLLMLive(), VLLMSemanticRouterLive()]
    benchmarks = build_live_benchmarks(routerbench_n=60, bfcl_n=30)
    run_live_phase(
        "ablation_narrowgap_v1",
        routers,
        benchmarks,
        seed=1234,
        n_trials=2,
        extra_manifest={"purpose": "narrow-gap candidate-pool ablation: cheap-small=claude-haiku-4-5"},
    )


if __name__ == "__main__":
    main()
