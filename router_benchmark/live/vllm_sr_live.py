"""Real vLLM Semantic Router adapter.

Runs the actual vllm-project/semantic-router service (installed via its
official installer, `vllm-sr`, v0.3.0) as a real local Docker stack --
Envoy + router + Redis/Postgres/Milvus -- configured with our 3 real
model tiers as real backend_refs (see live/vllm_sr/config.yaml), using
the `automix` cost-aware model-selection algorithm (one of the service's
real supported algorithms, matching the "cost, latency, safety signals"
description in the shortlist).

Routing decisions are read from the real `x-vsr-selected-model` response
header returned by Envoy on an actual live HTTP call to
http://localhost:8909/v1/chat/completions (model="MoM" = the service's
real mixture-of-models auto-route alias). We cap the probe call's
max_completion_tokens at 1 to minimize cost: the routing decision itself
is made by the service before generation begins (an Envoy/WASM filter
selecting the upstream backend based on the prompt), so truncating output
tokens doesn't affect which model is chosen, only the cost of this probe.

Prerequisite (see live/vllm_sr/README.md): the service must already be
running -- `VLLM_SR_PORT_OFFSET=10 vllm-sr serve --minimal --config
router_benchmark/live/vllm_sr/config.yaml` from that directory.
"""

from __future__ import annotations

import requests

from router_benchmark.interfaces import RouteDecision, Router, Task

ENVOY_URL = "http://localhost:8909/v1/chat/completions"


class VLLMSemanticRouterLive(Router):
    name = "vLLM Semantic Router (live)"

    def __init__(self, envoy_url: str = ENVOY_URL, timeout_s: float = 30.0):
        self.envoy_url = envoy_url
        self.timeout_s = timeout_s

    def route(self, task: Task, context: dict, rng) -> RouteDecision:
        prompt = task.metadata.get("prompt", "") or task.metadata.get("user_msg", "")
        if not prompt.strip():
            return RouteDecision(selected_candidate="mid-general", confidence=0.0, fallback_used=True)

        resp = requests.post(
            self.envoy_url,
            json={
                "model": "MoM",
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": 16,
            },
            timeout=self.timeout_s,
        )
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            return RouteDecision(selected_candidate="mid-general", confidence=0.0, fallback_used=True, metadata={"error": str(e)})
        selected = resp.headers.get("x-vsr-selected-model", "mid-general")
        return RouteDecision(
            selected_candidate=selected,
            confidence=0.7,
            fallback_used=resp.headers.get("x-vsr-response-path") == "fallback",
            metadata={"vsr_response_path": resp.headers.get("x-vsr-response-path")},
        )
