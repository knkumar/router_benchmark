"""Real router adapters, implementing the same Router interface as the
simulated ones in router_benchmark/routers.py, but backed by actual
installed router packages making real routing decisions.

Phase 1 covers the two routers with a real pip-installable package:
  - LiteLLM Router  (package: litellm)
  - Aurelio Semantic Router (package: semantic-router)

RouteLLM, vLLM Semantic Router, LLMRouter, and the NVIDIA AI Blueprint LLM
Router are not simple pip installs (no packaged classifier/checkpoint or a
blueprint-style deployment) and are deferred to Phase 2 -- see
router_benchmark/live/README.md.
"""

from __future__ import annotations

from router_benchmark.interfaces import Candidate, RouteDecision, Router, Task
from router_benchmark.live.llm_client import CANDIDATE_TIERS, PRICING
from router_benchmark.live.routing_context import PerRequestRouter

LIVE_CANDIDATES: tuple[Candidate, ...] = (
    Candidate("cheap-small", tier="cheap", cost_per_1k_tokens=0.0, base_quality=0.0, base_latency_ms=0.0),
    Candidate("mid-general", tier="mid", cost_per_1k_tokens=0.0, base_quality=0.0, base_latency_ms=0.0),
    Candidate("strong-frontier", tier="strong", cost_per_1k_tokens=0.0, base_quality=0.0, base_latency_ms=0.0),
)


class LiteLLMRouterLive(PerRequestRouter, Router):
    """Wraps litellm.Router's real cost-based routing strategy.

    litellm.Router is configured with all three candidate tiers as
    deployments of a single model group and asked to route with
    routing_strategy="cost-based-routing" -- LiteLLM's own documented,
    real deployment-selection policy, driven by each deployment's real
    per-token pricing (Section IV-B / live/llm_client.py PRICING).

    IMPORTANT (see router_bench_adapter_bugs memory / paper1 methodology):
    litellm's own source (litellm/router.py: _select_deployment_sync)
    explicitly omits "cost-based-routing" from its synchronous dispatch
    table, with the comment "LowestCostLoggingHandler only implements
    async_get_available_deployments" -- the strategy has no sync
    implementation. The real decision must be obtained via
    Router.async_get_available_deployment(); the synchronous
    Router.get_available_deployment() silently raises RouterRateLimitError
    for this strategy, which is what an earlier version of this adapter
    worked around with a hand-rolled difficulty ladder instead of ever
    calling the real router. That ladder has been removed: route() below
    calls the real async cost-based-routing path over the real per-task
    prompt. Verified against the real strategy directly (not through this
    adapter): it deterministically selects the cheapest deployment
    (cheap-small) regardless of prompt content, because cost-based-routing
    compares only each deployment's configured $/token, with no
    task-difficulty or task-content signal in its selection rule at all.
    That is a genuine, verified property of litellm's own strategy, not an
    adapter artifact -- and it is the real number reported in Section VI.
    """

    name = "LiteLLM Router (live)"
    _MODEL_GROUP = "agentic-tiers"

    def __init__(self):
        import litellm

        self._router = litellm.Router(
            model_list=[
                {
                    "model_name": self._MODEL_GROUP,
                    "litellm_params": {
                        "model": self._litellm_model_id(model),
                        "input_cost_per_token": PRICING[model][0],
                        "output_cost_per_token": PRICING[model][1],
                    },
                    "model_info": {"id": tier},
                }
                for tier, model in CANDIDATE_TIERS.items()
            ],
            routing_strategy="cost-based-routing",
        )

    @staticmethod
    def _litellm_model_id(model: str) -> str:
        if model.startswith("gpt-"):
            return f"openai/{model}"
        return f"anthropic/{model}"

    def route(self, task: Task, context: dict, rng) -> RouteDecision:
        prompt = task.metadata.get("prompt", "") or task.metadata.get("user_msg", "")
        return self._route_on_text(prompt, rng)

    def _route_on_text(self, text: str, rng) -> RouteDecision:
        import asyncio

        if not text.strip():
            return RouteDecision(selected_candidate="mid-general", confidence=0.0, fallback_used=True)

        deployment = asyncio.run(
            self._router.async_get_available_deployment(
                model=self._MODEL_GROUP,
                messages=[{"role": "user", "content": text}],
                request_kwargs={},
            )
        )
        tier = deployment.get("model_info", {}).get("id", "mid-general")
        return RouteDecision(
            selected_candidate=tier,
            confidence=1.0,
            fallback_used=False,
            metadata={"routing_strategy": "cost-based-routing"},
        )


class AurelioSemanticRouterLive(PerRequestRouter, Router):
    """Wraps aurelio-labs/semantic-router's real embedding-based route
    selection: encodes the task prompt, encodes a small set of reference
    utterances per tier (easy/medium/hard framing), and picks the tier
    whose reference set is closest by cosine similarity -- the package's
    actual mechanism, not a re-implementation of it.
    """

    name = "Aurelio Semantic Router (live)"

    _TIER_UTTERANCES = {
        "cheap-small": [
            "what is the capital of a country",
            "simple one step arithmetic",
            "a short factual lookup question",
        ],
        "mid-general": [
            "explain a moderately complex concept",
            "write a short function to solve a problem",
            "answer a multi-step reasoning question",
        ],
        "strong-frontier": [
            "solve a hard multi-step reasoning or proof problem",
            "debug a subtle error in a long piece of code",
            "resolve a complex ambiguous real-world task",
        ],
    }

    def __init__(self):
        from semantic_router import Route
        from semantic_router.encoders import OpenAIEncoder
        from semantic_router.routers import SemanticRouter

        # HuggingFaceEncoder needs torch>=2.4, which has no wheel available
        # for this platform/Python combination (torch caps at 2.2.2 here).
        # OpenAIEncoder uses the real OpenAI embeddings API instead --
        # still semantic-router's real routing mechanism end to end.
        encoder = OpenAIEncoder(name="text-embedding-3-small")
        routes = [
            Route(name=tier, utterances=utts) for tier, utts in self._TIER_UTTERANCES.items()
        ]
        # aggregation="max" (a real, documented semantic-router option --
        # semantic_router/routers/base.py:_set_aggregation_method) replaces
        # the previous default "mean". With only 3 short utterances per
        # route, "mean" let unrelated low-scoring utterances of the same
        # route dilute a single strong match below the 0.3 score_threshold
        # (verified directly: "What is the capital of France?" scores 0.60
        # against its best-matching cheap-small utterance, but the 3-way
        # mean of everything the index returns in top_k is ~0.25, under
        # threshold, so the router previously returned no match at all).
        # "max" uses each route's single best-matching utterance instead.
        self._router = SemanticRouter(encoder=encoder, routes=routes, auto_sync="local", aggregation="max")

    def route(self, task: Task, context: dict, rng) -> RouteDecision:
        prompt = task.metadata.get("prompt", "") or task.metadata.get("user_msg", "")
        return self._route_on_text(prompt, rng)

    def _route_on_text(self, text: str, rng) -> RouteDecision:
        if not text.strip():
            # A handful of real RouterBench rows carry an empty prompt
            # string, which the embeddings API rejects outright; fall back
            # to the mid tier rather than crashing the live run.
            return RouteDecision(selected_candidate="mid-general", confidence=0.0, fallback_used=True)
        result = self._router(text)
        if result and result.name:
            return RouteDecision(
                selected_candidate=result.name,
                confidence=float(result.similarity_score or 0.0),
                fallback_used=False,
            )
        # Real, honest fallback: none of the three tiers' reference
        # utterances scored above score_threshold=0.3 for this prompt (this
        # is common for WebArena/tau2-bench prompts, whose domains --
        # browser navigation, policy-constrained customer service -- have
        # no representative utterances in _TIER_UTTERANCES above). Earlier
        # versions of this adapter silently substituted "mid-general" here
        # with a fabricated confidence=0.7, indistinguishable in the trace
        # log from a genuine match; this is now logged honestly as a
        # fallback so mean_fallback_rate reflects it.
        return RouteDecision(selected_candidate="mid-general", confidence=0.0, fallback_used=True)


def build_live_routers() -> list[Router]:
    return [LiteLLMRouterLive(), AurelioSemanticRouterLive()]
