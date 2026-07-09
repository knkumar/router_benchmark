"""Real LLMRouter (ulab-uiuc) adapter.

LLMRouter (pip: llmrouter-lib) ships 16+ real trainable routers plus real
example training data (data/example_data/ in the GitHub repo: 5,616 real
training queries spanning agentverse-logicgrid/HumanEval/MBPP/MT-Bench/
etc., with real per-model performance labels from actual LLM runs). No
pretrained checkpoints are published anywhere (checked HF hub and GitHub
releases), so this uses the package's own real `knnrouter` trainer on that
real bundled data -- KNeighborsClassifier over query embeddings, predicting
which of their labeled candidate models scored highest on similar past
queries. This is real training, just at small (bundled-example) scale
rather than their full 11-benchmark data-generation pipeline.

PLATFORM SUBSTITUTION: LLMRouter's default query encoder is a live
Longformer forward pass (`get_longformer_embedding`, via
transformers.AutoModel), which requires torch>=2.4. This machine's torch is
capped at 2.2.2 (no newer wheel exists for this platform in either PyPI's
or PyTorch's own index -- confirmed directly). We substitute real OpenAI
embeddings (text-embedding-3-small) consistently on BOTH sides: the KNN was
retrained on real OpenAI embeddings of the same 5,616 real training queries
(see router_benchmark/live/README.md for the re-embedding script), and
route() computes a real OpenAI embedding for each live task prompt the same
way, so train/inference embeddings are in the same space.

The KNN's output is one of LLMRouter's own labeled candidate model names
(not our 3 tiers); we map that prediction onto our tiers by the labeled
model's approximate capability class.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from openai import OpenAI

from router_benchmark.interfaces import RouteDecision, Router, Task
from router_benchmark.live.live_routers import LIVE_CANDIDATES

KNN_MODEL_PATH = str(
    Path.home() / ".local/share/router_bench_vendor/LLMRouter/saved_models/knnrouter/knnrouter.pkl"
)

# LLMRouter's real example training data (default_routing_train_data.jsonl)
# labels each query with one of these 9 real candidate models; map each onto
# our 3 tiers by parameter count.
_MODEL_NAME_TO_TIER = {
    "llama3-chatqa-1.5-8b": "cheap-small",
    "qwen2.5-7b-instruct": "cheap-small",
    "gemma-2-9b-it": "cheap-small",
    "mistral-7b-instruct-v0.3": "cheap-small",
    "llama-3.1-8b-instruct": "cheap-small",
    "codegemma-7b": "cheap-small",
    "llama-3.1-nemotron-51b-instruct": "mid-general",
    "llama-3.3-nemotron-super-49b-v1": "mid-general",
    "llama3-chatqa-1.5-70b": "strong-frontier",
}


def _tier_for(model_name: str) -> str:
    if model_name in _MODEL_NAME_TO_TIER:
        return _MODEL_NAME_TO_TIER[model_name]
    # Unseen label: fall back on rough size/name heuristics.
    name = model_name.lower()
    if any(k in name for k in ("70b", "gpt-4", "opus", "large")):
        return "strong-frontier"
    if any(k in name for k in ("7b", "8b", "mini", "small")):
        return "cheap-small"
    return "mid-general"


class LLMRouterLive(Router):
    name = "LLMRouter (live)"

    def __init__(self, model_path: str = KNN_MODEL_PATH):
        with open(model_path, "rb") as f:
            self.knn_model = pickle.load(f)
        self._client = OpenAI()

    def _embed(self, text: str):
        resp = self._client.embeddings.create(model="text-embedding-3-small", input=[text[:6000]])
        return resp.data[0].embedding

    def route(self, task: Task, context: dict, rng) -> RouteDecision:
        prompt = task.metadata.get("prompt", "") or task.metadata.get("user_msg", "")
        if not prompt.strip():
            return RouteDecision(selected_candidate="mid-general", confidence=0.0, fallback_used=True)

        embedding = [self._embed(prompt)]
        predicted_model = self.knn_model.predict(embedding)[0]
        tier = _tier_for(predicted_model)

        return RouteDecision(
            selected_candidate=tier,
            confidence=0.6,
            fallback_used=False,
            metadata={"knn_predicted_model": predicted_model},
        )
