"""Real RouteLLM adapter.

Uses RouteLLM's actual published "sw_ranking" router (Ong et al. 2024,
lm-sys/RouteLLM on PyPI as `routellm`): the real Chatbot Arena battle
datasets (lmsys/lmsys-arena-human-preference-55k, routellm/gpt4_judge_battles)
and their real precomputed embedding datasets
(routellm/arena_battles_embeddings, routellm/gpt4_judge_battles_embeddings),
combined with RouteLLM's real Elo win-rate computation
(compute_elo_mle_with_tie / compute_tiers from
routellm.routers.similarity_weighted.utils) to decide, per real query
embedding (via the real OpenAI embeddings API), whether the "strong" or
"weak" model in RouteLLM's own binary framing would be expected to win.

The "mf" (matrix-factorization) and "bert"/"causal_llm" routers are
RouteLLM's other real options but require transformers>=2.4-era model
loading that this platform's torch build (capped at 2.2.2, no newer wheel
available) cannot run -- sw_ranking has no such dependency, only numpy +
the OpenAI embeddings API, so it is used here.

KNOWN LIBRARY BUG WORKAROUND: routellm 0.2.0's SWRankingRouter.__init__
calls datasets.concatenate_datasets() on the two arena battle datasets
without aligning their Arrow schemas; the current `datasets` library
(5.0.0) enforces exact dtype match and rejects the resulting
string/large_string mismatch. We reimplement __init__ with an explicit
schema cast before concatenation -- everything else (Elo computation,
tiering, win-rate math) is RouteLLM's own unmodified code, called via
`super().__new__` + direct method reuse.
"""

from __future__ import annotations

from datasets import concatenate_datasets, load_dataset

from router_benchmark.interfaces import RouteDecision, Router, Task
from router_benchmark.live.live_routers import LIVE_CANDIDATES
from router_benchmark.live.routing_context import PerRequestRouter

ARENA_BATTLE_DATASETS = ["lmsys/lmsys-arena-human-preference-55k", "routellm/gpt4_judge_battles"]
ARENA_EMBEDDING_DATASETS = ["routellm/arena_battles_embeddings", "routellm/gpt4_judge_battles_embeddings"]

# RouteLLM's own binary framing: a "strong" and a "weak" reference model
# name, used only to key into its own Elo tiering -- not necessarily the
# literal models we route requests to (we map its strong/weak decision onto
# our own strong-frontier/cheap-small tiers).
_REF_STRONG = "gpt-4-1106-preview"
_REF_WEAK = "mixtral-8x7b-instruct-v0.1"


class RouteLLMLive(PerRequestRouter, Router):
    name = "RouteLLM (live)"

    def __init__(self, threshold: float = 0.5):
        import numpy as np

        from routellm.routers.routers import SWRankingRouter
        from routellm.routers.similarity_weighted.utils import (
            compute_elo_mle_with_tie,
            compute_tiers,
            preprocess_battles,
        )

        self.threshold = threshold
        self._np = np

        # Load and schema-align the two real battle datasets before
        # concatenation (see module docstring: library bug workaround).
        battle_dfs = []
        for name in ARENA_BATTLE_DATASETS:
            ds = load_dataset(name, split="train")
            battle_dfs.append(ds)
        target_features = battle_dfs[0].features
        battle_dfs = [ds.cast(target_features) for ds in battle_dfs]

        self._router = object.__new__(SWRankingRouter)
        self._router.strong_model = _REF_STRONG
        self._router.weak_model = _REF_WEAK
        self._router.arena_df = preprocess_battles(concatenate_datasets(battle_dfs).to_pandas())

        embeddings = [
            np.array(load_dataset(d, split="train").to_dict()["embeddings"]) for d in ARENA_EMBEDDING_DATASETS
        ]
        self._router.arena_conv_embedding = np.concatenate(embeddings)
        self._router.embedding_model = "text-embedding-3-small"

        model_ratings = compute_elo_mle_with_tie(self._router.arena_df)
        self._router.model2tier = compute_tiers(model_ratings, num_tiers=10)
        self._router.arena_df["model_a"] = self._router.arena_df["model_a"].apply(
            lambda x: self._router.model2tier[x]
        )
        self._router.arena_df["model_b"] = self._router.arena_df["model_b"].apply(
            lambda x: self._router.model2tier[x]
        )

    def route(self, task: Task, context: dict, rng) -> RouteDecision:
        prompt = task.metadata.get("prompt", "") or task.metadata.get("user_msg", "")
        return self._route_on_text(prompt, rng)

    def _route_on_text(self, text: str, rng) -> RouteDecision:
        if not text.strip():
            return RouteDecision(selected_candidate="mid-general", confidence=0.0, fallback_used=True)
        strong_winrate = self._router.calculate_strong_win_rate(text)
        # RouteLLM's real binary decision rule (Controller.route): route to
        # strong iff win rate exceeds the threshold, else weak.
        tier = "strong-frontier" if strong_winrate >= self.threshold else "cheap-small"
        return RouteDecision(
            selected_candidate=tier,
            confidence=float(strong_winrate if tier == "strong-frontier" else 1 - strong_winrate),
            fallback_used=False,
            metadata={"strong_winrate": float(strong_winrate)},
        )

    @property
    def embedding_request_timeout_s(self) -> float:
        """Read timeout actually configured on routellm's shared OpenAI client."""
        from routellm.routers.routers import OPENAI_CLIENT

        return float(OPENAI_CLIENT.timeout.read)

    @property
    def embedding_max_retries(self) -> int:
        """Retry count routellm's shared OpenAI client applies per embed call."""
        from routellm.routers.routers import OPENAI_CLIENT

        return int(OPENAI_CLIENT.max_retries)
