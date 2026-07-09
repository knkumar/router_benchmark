"""Router adapters.

IMPORTANT — SIMULATED BACKEND
------------------------------
This environment has no network access and no configured model-provider
credentials, so none of the adapters below call the real upstream projects.
Each adapter instead wraps a small, explicit `RouterProfile` — a set of
numeric parameters (quality bias, domain affinity, cost preference, latency
overhead, fallback rate, tool-call reliability, route stability) chosen to be
*directionally consistent* with each project's own documentation and design
goals (cited per-class below). A fixed `numpy.random.Generator` seeded per
(router, benchmark, task, trial) drives every stochastic decision, so results
are fully reproducible.

This is a stand-in for, not a replacement of, live evaluation. See
paper/paper.md, Section IV (Methodology / Limitations) for the same caveat
in the write-up, and README.md for how to swap in a live adapter later:
replace `route()` in any class below with a real HTTP/library call while
keeping the same `Router` interface, and the harness and metrics are
unaffected.

Citations (see paper/paper.md references for full entries):
  RouteLLM              -- Ong et al., 2024 [1]; https://github.com/lm-sys/RouteLLM
  LiteLLM Router        -- BerriAI, LiteLLM docs [2]; https://docs.litellm.ai/
  vLLM Semantic Router   -- vLLM Project [3]; https://github.com/vllm-project/semantic-router
  Aurelio Semantic Router-- Aurelio Labs [4]; https://github.com/aurelio-labs/semantic-router
  LLMRouter              -- ulab-uiuc [5]; https://github.com/ulab-uiuc/LLMRouter
  NVIDIA AI Blueprint     -- NVIDIA-AI-Blueprints [6]; https://github.com/NVIDIA-AI-Blueprints/llm-router
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from router_benchmark.interfaces import Candidate, Router, RouteDecision, Task, TaskDomain


@dataclass(frozen=True)
class RouterProfile:
    """Parameters governing a synthetic router's behavior."""

    quality_bias: float  # 0-1: probability mass toward the best-fit candidate
    cost_preference: float  # 0 = always pick cheapest, 1 = always pick strongest
    domain_affinity: dict[TaskDomain, float] = field(default_factory=dict)
    latency_overhead_ms: float = 5.0  # routing decision overhead itself
    base_fallback_rate: float = 0.03
    tool_reliability: float = 0.85  # baseline P(tool call correct | routed well)
    stability: float = 0.9  # P(pick the same candidate again on a repeat trial)
    confidence_noise: float = 0.08


class SyntheticProfileRouter(Router):
    """Generic router driven by a RouterProfile. Subclassed per named router."""

    def __init__(self, name: str, profile: RouterProfile):
        self.name = name
        self.profile = profile

    def _domain_multiplier(self, domain: TaskDomain) -> float:
        return self.profile.domain_affinity.get(domain, 1.0)

    def route(self, task: Task, context: dict, rng: np.random.Generator) -> RouteDecision:
        p = self.profile
        candidates = list(task.candidates)
        domain_mult = self._domain_multiplier(task.domain)

        # Expected quality if we route to each candidate: candidate capability
        # discounted by task difficulty, boosted by domain affinity.
        fit_scores = np.array(
            [c.base_quality * domain_mult - 0.35 * task.difficulty * (1 - c.base_quality) for c in candidates]
        )
        # Cost preference reshapes the score: cost_preference near 1 rewards
        # strong/expensive candidates, near 0 rewards cheap ones.
        cost_term = np.array([c.cost_per_1k_tokens for c in candidates])
        cost_range = cost_term.max() - cost_term.min()
        cost_term = (cost_term - cost_term.min()) / (cost_range + 1e-9)
        blended = fit_scores + (p.cost_preference - 0.5) * cost_term

        # quality_bias controls how peaked the softmax is: a sharper router
        # concentrates probability mass on the argmax candidate.
        temperature = max(0.02, 1.0 - p.quality_bias)
        exp_scores = np.exp((blended - blended.max()) / temperature)
        probs = exp_scores / exp_scores.sum()

        idx = rng.choice(len(candidates), p=probs)
        selected = candidates[idx]

        confidence = float(np.clip(probs[idx] + rng.normal(0, p.confidence_noise), 0.05, 0.99))

        fallback_prob = p.base_fallback_rate + 0.25 * task.difficulty * (1 - selected.base_quality)
        fallback_used = bool(rng.random() < fallback_prob)
        if fallback_used:
            strongest = max(candidates, key=lambda c: c.base_quality)
            selected = strongest

        return RouteDecision(
            selected_candidate=selected.name,
            confidence=confidence,
            fallback_used=fallback_used,
            metadata={"router_profile": self.name},
        )


# ---------------------------------------------------------------------------
# The six shortlisted routers
# ---------------------------------------------------------------------------


def make_routellm() -> SyntheticProfileRouter:
    """RouteLLM [1]: weak-vs-strong query-level cost/quality router with a
    matrix-factorization / classifier decision boundary trained on human
    preference data (e.g., Chatbot Arena). Modeled as high quality_bias
    (sharp, well-calibrated weak/strong split), moderate-low cost_preference
    (favors savings when confidently easy), strong on QA/reasoning
    (its original training domain), weaker on tool-use and long-horizon
    agent domains it was not designed for."""
    profile = RouterProfile(
        quality_bias=0.80,
        cost_preference=0.35,
        domain_affinity={
            TaskDomain.QA_REASONING: 1.10,
            TaskDomain.CODE_REPAIR: 0.95,
            TaskDomain.TOOL_USE: 0.75,
            TaskDomain.MULTI_TURN_POLICY: 0.70,
            TaskDomain.WEB_NAVIGATION: 0.65,
            TaskDomain.TERMINAL_AGENT: 0.70,
        },
        latency_overhead_ms=8.0,
        base_fallback_rate=0.02,
        tool_reliability=0.75,
        stability=0.92,
        confidence_noise=0.05,
    )
    return SyntheticProfileRouter("RouteLLM", profile)


def make_litellm_router() -> SyntheticProfileRouter:
    """LiteLLM Router [2]: OpenAI-compatible proxy with load balancing,
    budget-aware fallbacks, and usage-based/adaptive routing. Modeled as a
    production-reliability-first router: high stability and high fallback
    coverage (its core feature is graceful fallback), broad but shallow
    domain affinity because it treats routing as infra, not semantic
    understanding of the task."""
    profile = RouterProfile(
        quality_bias=0.55,
        cost_preference=0.45,
        domain_affinity={d: 0.95 for d in TaskDomain},
        latency_overhead_ms=4.0,
        base_fallback_rate=0.10,
        tool_reliability=0.80,
        stability=0.97,
        confidence_noise=0.12,
    )
    return SyntheticProfileRouter("LiteLLM Router", profile)


def make_vllm_semantic_router() -> SyntheticProfileRouter:
    """vLLM Semantic Router [3]: signal-driven semantic routing across
    local/private/frontier model pools with explicit privacy/cost/latency/
    safety signals. Modeled as latency-optimized (routes fast, low
    overhead) with strong cost_preference toward local/cheap pools, and
    good but newer-project domain coverage skewed toward general QA and
    tool-routing signals rather than long-horizon agent tasks."""
    profile = RouterProfile(
        quality_bias=0.65,
        cost_preference=0.25,
        domain_affinity={
            TaskDomain.QA_REASONING: 1.05,
            TaskDomain.CODE_REPAIR: 0.85,
            TaskDomain.TOOL_USE: 0.95,
            TaskDomain.MULTI_TURN_POLICY: 0.75,
            TaskDomain.WEB_NAVIGATION: 0.70,
            TaskDomain.TERMINAL_AGENT: 0.75,
        },
        latency_overhead_ms=2.0,
        base_fallback_rate=0.05,
        tool_reliability=0.82,
        stability=0.88,
        confidence_noise=0.10,
    )
    return SyntheticProfileRouter("vLLM Semantic Router", profile)


def make_aurelio_semantic_router() -> SyntheticProfileRouter:
    """Aurelio Semantic Router [4]: embedding-based deterministic route
    selection for tools, intents, and agents, optimized for fast dispatch
    speed rather than downstream task success. Modeled with very low
    latency overhead, strong tool/intent domain affinity, but weaker
    quality_bias on tasks needing deep reasoning about difficulty (it
    dispatches on semantic similarity, not task-difficulty estimation)."""
    profile = RouterProfile(
        quality_bias=0.50,
        cost_preference=0.30,
        domain_affinity={
            TaskDomain.QA_REASONING: 0.85,
            TaskDomain.CODE_REPAIR: 0.75,
            TaskDomain.TOOL_USE: 1.20,
            TaskDomain.MULTI_TURN_POLICY: 0.90,
            TaskDomain.WEB_NAVIGATION: 0.80,
            TaskDomain.TERMINAL_AGENT: 0.70,
        },
        latency_overhead_ms=1.5,
        base_fallback_rate=0.04,
        tool_reliability=0.90,
        stability=0.85,
        confidence_noise=0.09,
    )
    return SyntheticProfileRouter("Aurelio Semantic Router", profile)


def make_llmrouter() -> SyntheticProfileRouter:
    """LLMRouter [5]: focused academic library for dynamic query-level model
    selection over candidate LLMs. Modeled as a moderate, general-purpose
    research baseline: decent quality_bias, no strong domain specialization,
    reflecting its stated scope (model selection, not agent/tool routing)
    and the shortlist's caveat to check maturity before large-scale use."""
    profile = RouterProfile(
        quality_bias=0.68,
        cost_preference=0.40,
        domain_affinity={
            TaskDomain.QA_REASONING: 1.00,
            TaskDomain.CODE_REPAIR: 0.90,
            TaskDomain.TOOL_USE: 0.70,
            TaskDomain.MULTI_TURN_POLICY: 0.65,
            TaskDomain.WEB_NAVIGATION: 0.60,
            TaskDomain.TERMINAL_AGENT: 0.65,
        },
        latency_overhead_ms=6.0,
        base_fallback_rate=0.06,
        tool_reliability=0.72,
        stability=0.80,
        confidence_noise=0.14,
    )
    return SyntheticProfileRouter("LLMRouter", profile)


def make_nvidia_blueprint_router() -> SyntheticProfileRouter:
    """NVIDIA AI Blueprint LLM Router [6]: intent router + auto-router
    blueprint for frontier/open model selection under cost-quality-latency
    goals, designed for production-style stacks with retraining/config
    support. Modeled with strong, broad domain affinity (production
    blueprint covers many task types), higher cost_preference (defaults
    toward capable models when uncertain), and moderate fallback coverage."""
    profile = RouterProfile(
        quality_bias=0.72,
        cost_preference=0.55,
        domain_affinity={
            TaskDomain.QA_REASONING: 1.05,
            TaskDomain.CODE_REPAIR: 1.05,
            TaskDomain.TOOL_USE: 1.00,
            TaskDomain.MULTI_TURN_POLICY: 0.95,
            TaskDomain.WEB_NAVIGATION: 0.90,
            TaskDomain.TERMINAL_AGENT: 0.95,
        },
        latency_overhead_ms=7.0,
        base_fallback_rate=0.07,
        tool_reliability=0.86,
        stability=0.90,
        confidence_noise=0.07,
    )
    return SyntheticProfileRouter("NVIDIA AI Blueprint LLM Router", profile)


# ---------------------------------------------------------------------------
# Baselines (Recommended Comparative Evaluation Protocol, item 4)
# ---------------------------------------------------------------------------


class AlwaysCheapestRouter(Router):
    name = "Baseline: Always Cheapest"

    def route(self, task: Task, context: dict, rng: np.random.Generator) -> RouteDecision:
        selected = min(task.candidates, key=lambda c: c.cost_per_1k_tokens)
        return RouteDecision(selected.name, confidence=0.5, fallback_used=False)


class AlwaysStrongestRouter(Router):
    name = "Baseline: Always Strongest"

    def route(self, task: Task, context: dict, rng: np.random.Generator) -> RouteDecision:
        selected = max(task.candidates, key=lambda c: c.base_quality)
        return RouteDecision(selected.name, confidence=0.5, fallback_used=False)


class RandomRouter(Router):
    name = "Baseline: Random"

    def route(self, task: Task, context: dict, rng: np.random.Generator) -> RouteDecision:
        idx = rng.integers(0, len(task.candidates))
        return RouteDecision(task.candidates[idx].name, confidence=float(rng.random()), fallback_used=False)


class HeuristicDifficultyRouter(Router):
    """Keyword/difficulty heuristic: cheap below a difficulty threshold,
    strong above it. No learned signal, no domain awareness."""

    name = "Baseline: Heuristic Difficulty"

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def route(self, task: Task, context: dict, rng: np.random.Generator) -> RouteDecision:
        candidates = sorted(task.candidates, key=lambda c: c.base_quality)
        selected = candidates[-1] if task.difficulty >= self.threshold else candidates[0]
        return RouteDecision(selected.name, confidence=0.6, fallback_used=False)


class OracleRouter(Router):
    """Oracle: always the candidate with highest expected quality for the
    task's true difficulty, ignoring cost. Upper bound on success rate,
    not a fair cost/latency comparator; excluded from cost-based rankings
    in metrics.py."""

    name = "Baseline: Oracle"

    def route(self, task: Task, context: dict, rng: np.random.Generator) -> RouteDecision:
        scored = [(c.base_quality - 0.35 * task.difficulty * (1 - c.base_quality), c) for c in task.candidates]
        selected = max(scored, key=lambda t: t[0])[1]
        return RouteDecision(selected.name, confidence=0.95, fallback_used=False)


def build_all_routers() -> list[Router]:
    """Convenience factory used by run.py."""
    return [
        make_routellm(),
        make_litellm_router(),
        make_vllm_semantic_router(),
        make_aurelio_semantic_router(),
        make_llmrouter(),
        make_nvidia_blueprint_router(),
        AlwaysCheapestRouter(),
        AlwaysStrongestRouter(),
        RandomRouter(),
        HeuristicDifficultyRouter(),
        OracleRouter(),
    ]
