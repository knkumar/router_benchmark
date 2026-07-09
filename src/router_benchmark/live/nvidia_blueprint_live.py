"""Real NVIDIA AI Blueprint LLM Router adapter.

The blueprint (NVIDIA-AI-Blueprints/llm-router) ships two real routing
methods: (1) "Auto-Routing" via CLIP embeddings + a pretrained neural
network, which requires the NVIDIA NVClip NIM microservice for embedding
generation -- not available without NIM access; and (2) "Intent-Based
Routing" via a small classifier model prompted with the blueprint's own
route taxonomy and prompt templates (src/nat_sfc_router/functions/
hf_intent_objective_fn.py), designed to call a NIM-hosted Qwen classifier
over HTTP.

This adapter reuses the blueprint's real route taxonomy
(`route_config`), its real TASK_INSTRUCTION/FORMAT_PROMPT templates, and
its real JSON-parsing logic (including the single-quote fallback via
ast.literal_eval) verbatim, substituting our own live LLM client for the
NIM-hosted Qwen classifier the blueprint expects (we have no NIM/GPU
access in this environment). The blueprint's own `MAP_INTENT_TO_PIPELINE`
targets its own model roster (Nemotron/GPT-5-chat); we remap those same
intents onto our 3 real tiers since we don't have those specific backing
models.

Source: NVIDIA-AI-Blueprints/llm-router, src/nat_sfc_router/functions/
hf_intent_objective_fn.py (cloned 2026-07-03).
"""

from __future__ import annotations

import ast
import json

from router_benchmark.interfaces import RouteDecision, Router, Task
from router_benchmark.live.live_routers import LIVE_CANDIDATES
from router_benchmark.live.llm_client import LiveLLMClient

# Verbatim from the blueprint's hf_intent_objective_fn.py.
TASK_INSTRUCTION = """
You are a helpful assistant designed to find the best suited route.
You are provided with route description within <routes></routes> XML tags:
<routes>

{routes}

</routes>

<conversation>

{conversation}

</conversation>
"""

FORMAT_PROMPT = """
Your task is to decide which route is best suit with user intent on the conversation in <conversation></conversation> XML tags.  Follow the instruction:
1. If the latest intent from user is irrelevant or user intent is full filled, response with other route {"route": "other"}.
2. You must analyze the route descriptions and find the best match route for user latest intent.
3. You only response the name of the route that best matches the user's request, use the exact name in the <routes></routes>.

Based on your analysis, provide your response in the following JSON formats if you decide to match any route:
{"route": "route_name"}
"""

ROUTE_CONFIG = [
    {
        "name": "hard_question",
        "description": "A question that requires deep reasoning, or complex problem solving, or if the user asks for careful thinking or careful consideration",
    },
    {"name": "chit_chat", "description": "Any social chit chat, small talk, or casual conversation."},
    {"name": "try_again", "description": "Only if the user explicitly says the previous answer was incorrect or incomplete."},
    {"name": "image_understanding", "description": "A question that requires understanding an image."},
    {"name": "image_question", "description": "A question that requires the assistant to see the user eg a question about their appearance, environment, scene or surroundings."},
]

# The blueprint maps intents to its own model roster (Nemotron/GPT-5-chat);
# we remap onto our 3 real tiers since those specific models aren't
# available to us. image_* intents shouldn't fire on our text-only tasks.
_INTENT_TO_TIER = {
    "other": "cheap-small",
    "chit_chat": "cheap-small",
    "try_again": "strong-frontier",
    "hard_question": "strong-frontier",
    "image_understanding": "mid-general",
    "image_question": "mid-general",
}

_ROUTES_JSON = json.dumps(ROUTE_CONFIG)


def _parse_route_response(response: str) -> str:
    """Verbatim parsing logic from the blueprint (JSON, with a single-quote
    ast.literal_eval fallback for models that don't emit valid JSON)."""
    try:
        return json.loads(response)["route"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return ast.literal_eval(response)["route"]


class NVIDIABlueprintRouterLive(Router):
    name = "NVIDIA AI Blueprint LLM Router (live)"

    def __init__(self, classifier_model: str = "gpt-5.4-nano"):
        self._client = LiveLLMClient()
        self.classifier_model = classifier_model

    def route(self, task: Task, context: dict, rng) -> RouteDecision:
        prompt = task.metadata.get("prompt", "") or task.metadata.get("user_msg", "")
        if not prompt.strip():
            return RouteDecision(selected_candidate="mid-general", confidence=0.0, fallback_used=True)

        conversation = [{"role": "user", "content": prompt}]
        system_prompt = TASK_INSTRUCTION.format(routes=_ROUTES_JSON, conversation=json.dumps(conversation)) + FORMAT_PROMPT

        result = self._client.call(
            model=self.classifier_model,
            system=system_prompt,
            user="Respond with only the JSON route decision.",
            max_tokens=40,
            trace_context={"benchmark_name": task.benchmark_name, "task_id": task.task_id, "role": "nvidia_blueprint_intent_classifier"},
        )

        try:
            intent = _parse_route_response(result.text.strip())
            tier = _INTENT_TO_TIER.get(intent, "mid-general")
        except Exception:
            intent = None
            tier = "mid-general"

        return RouteDecision(
            selected_candidate=tier,
            confidence=0.6 if intent else 0.0,
            fallback_used=intent is None,
            metadata={"classified_intent": intent},
        )
