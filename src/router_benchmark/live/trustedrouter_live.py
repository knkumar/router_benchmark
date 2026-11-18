"""Real TrustedRouter adapter.

TrustedRouter (https://trustedrouter.com, docs: https://trustedrouter.com/docs)
is a hosted OpenAI-compatible gateway with server-side routing. Sending
model="trustedrouter/auto" lets the gateway pick the underlying model per
request; the chosen model is reported in the response's `model` field as a
vendor-prefixed id (e.g. "anthropic/claude-sonnet-4-6").

Routing decisions are read from that `model` field on a live HTTP call to
https://api.trustedrouter.com/v1/chat/completions, with
max_completion_tokens capped at 16 to keep the probe cheap (the model is
selected before generation, so truncating output does not change the
choice). TrustedRouter routes over its own catalog rather than our 3 tiers,
so the selected model is mapped onto a tier: an exact match against
CANDIDATE_TIERS first, otherwise the tier whose PRICING input rate is
nearest (log scale) to the selected model's input rate in the public model
list (https://api.trustedrouter.com/v1/models, no key needed).

Prerequisite: TRUSTEDROUTER_API_KEY must be set. Each route() is a real,
billed completion call.
"""

from __future__ import annotations

import math
import os

import requests

from router_benchmark.interfaces import RouteDecision, Router, Task
from router_benchmark.live.llm_client import CANDIDATE_TIERS, PRICING

BASE_URL = "https://api.trustedrouter.com/v1"
AUTO_MODEL = "trustedrouter/auto"


class TrustedRouterLive(Router):
    name = "TrustedRouter (live)"

    def __init__(self, base_url: str = BASE_URL, model: str = AUTO_MODEL, timeout_s: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self._input_prices: dict[str, float] | None = None

    def route(self, task: Task, context: dict, rng) -> RouteDecision:
        prompt = task.metadata.get("prompt", "") or task.metadata.get("user_msg", "")
        if not prompt.strip():
            return RouteDecision(selected_candidate="mid-general", confidence=0.0, fallback_used=True)

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['TRUSTEDROUTER_API_KEY']}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": 16,
            },
            timeout=self.timeout_s,
        )
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            return RouteDecision(selected_candidate="mid-general", confidence=0.0, fallback_used=True, metadata={"error": str(e)})

        selected_model = resp.json().get("model") or ""
        tier, mapping = self._tier_for_model(selected_model)
        if tier is None:
            return RouteDecision(
                selected_candidate="mid-general",
                confidence=0.0,
                fallback_used=True,
                metadata={"trustedrouter_model": selected_model},
            )
        return RouteDecision(
            selected_candidate=tier,
            confidence=0.7,
            fallback_used=False,
            metadata={"trustedrouter_model": selected_model, "tier_mapping": mapping},
        )

    def _tier_for_model(self, model_id: str) -> tuple[str | None, str | None]:
        bare = model_id.split("/", 1)[-1]
        for tier, model in CANDIDATE_TIERS.items():
            if bare == model:
                return tier, "exact"

        price = self._catalog_input_prices().get(model_id)
        if not price or price <= 0:
            return None, None
        tier = min(
            CANDIDATE_TIERS,
            key=lambda t: abs(math.log(PRICING[CANDIDATE_TIERS[t]][0]) - math.log(price)),
        )
        return tier, "price"

    def _catalog_input_prices(self) -> dict[str, float]:
        if self._input_prices is None:
            resp = requests.get(f"{self.base_url}/models", timeout=self.timeout_s)
            resp.raise_for_status()
            prices = {}
            for m in resp.json().get("data", []):
                try:
                    prices[m["id"]] = float(m["pricing"]["prompt"])
                except (KeyError, TypeError, ValueError):
                    continue
            self._input_prices = prices
        return self._input_prices
