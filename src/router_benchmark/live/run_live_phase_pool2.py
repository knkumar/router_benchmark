"""Phase B (review-response) Task 5: second candidate-pool ablation.

Reviewer objection: every headline result rests on one three-tier pool, and
the existing narrow-gap ablation (cheap-small -> claude-haiku-4-5) already
shows some conclusions flip when the cheap tier changes, so "trivial
heuristics match complex routers" may be an artifact of the cheap tier being
good enough. One ablation on one axis is not enough.

This phase adds a second, opposite point on the capability-gap axis: it swaps
cheap-small from gpt-5.4-nano to the older, weaker gpt-4.1-nano (OpenAI,
$0.10/$0.40 per 1M input/output tokens, developers.openai.com pricing page),
WIDENING the cheap/strong gap, while the existing ablation NARROWED it. Read
together with the default pool, the three points span the axis:

    wider gap (gpt-4.1-nano) < default (gpt-5.4-nano) < narrower gap (haiku)

in cheap-tier strength. mid-general (claude-sonnet-4-6) and strong-frontier
(claude-opus-4-8) are unchanged. Scope: RouterBench + BFCL v4, 4 real
routers, 2 trials each, matching the existing ablation's convention.

Note on RouterBench: its replay maps tiers to fixed LOGGED models by name
(live_benchmarks._ROUTERBENCH_TIER_MODEL), so the RouterBench column is
invariant to this candidate-pool swap by construction, exactly as in the
narrow-gap ablation; the pool signal is carried by BFCL v4 (live).

Implementation mirrors run_live_phase_ablation.py: the tier dicts are read by
name at call time, so mutating them in place BEFORE importing/constructing
any router or benchmark makes every adapter see the wider-gap pool without
touching adapter source. This only affects this one-off process.
"""

from router_benchmark.live import llm_client

llm_client.PRICING["gpt-4.1-nano"] = (0.10 / 1_000_000, 0.40 / 1_000_000)
llm_client._PROVIDER_OF["gpt-4.1-nano"] = "openai"
llm_client.CANDIDATE_TIERS["cheap-small"] = "gpt-4.1-nano"  # was gpt-5.4-nano


def main() -> None:
    from router_benchmark.live.live_routers import AurelioSemanticRouterLive, LiteLLMRouterLive
    from router_benchmark.live.routellm_live import RouteLLMLive
    from router_benchmark.live.vllm_sr_live import VLLMSemanticRouterLive
    from router_benchmark.live.live_benchmarks import build_live_benchmarks
    from router_benchmark.live.run_common import run_live_phase

    routers = [
        LiteLLMRouterLive(),
        AurelioSemanticRouterLive(),
        RouteLLMLive(),
        VLLMSemanticRouterLive(),
    ]
    benchmarks = build_live_benchmarks(routerbench_n=60, bfcl_n=30)
    run_live_phase(
        "pool2_widegap_v1",
        routers,
        benchmarks,
        seed=1234,
        n_trials=2,
        extra_manifest={
            "purpose": "second candidate-pool ablation: cheap-small -> gpt-4.1-nano "
            "(wider capability gap), complementing the narrow-gap (haiku) ablation",
            "cheap_tier": "gpt-4.1-nano",
            "cheap_tier_pricing_usd_per_1m": [0.10, 0.40],
        },
    )


if __name__ == "__main__":
    main()

