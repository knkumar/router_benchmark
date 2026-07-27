"""Per-request routing helpers: reduce an OpenAI-style chat request to the
single salient string a content-based router routes on, and a mixin giving
every live router a route_request() path parallel to its route()."""
from __future__ import annotations

from typing import Any

from router_benchmark.interfaces import RouteDecision


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # OpenAI "parts" form
        parts = []
        for p in content:
            if isinstance(p, dict):
                if isinstance(p.get("text"), str):
                    parts.append(p["text"])
                elif isinstance(p.get("content"), str):
                    parts.append(p["content"])
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts)
    return str(content)


def latest_routing_text(messages: list[dict]) -> str:
    """Text a per-request router routes on: content of the most recent user
    or tool message; falls back to the last message with any non-empty text."""
    if not messages:
        return ""
    for msg in reversed(messages):
        if msg.get("role") in ("user", "tool"):
            text = _content_to_text(msg.get("content")).strip()
            if text:
                return text
    for msg in reversed(messages):
        text = _content_to_text(msg.get("content")).strip()
        if text:
            return text
    return ""


class PerRequestRouter:
    """Mixin: route_request() extracts the salient text and delegates to the
    router's own _route_on_text() -- the same method its route() uses."""

    def route_request(self, messages, tools, candidates, context, rng) -> RouteDecision:
        return self._route_on_text(latest_routing_text(messages), rng)

    def _route_on_text(self, text: str, rng) -> RouteDecision:  # pragma: no cover
        raise NotImplementedError
