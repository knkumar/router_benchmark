"""Real LLM client: makes actual API calls to OpenAI and Anthropic and
returns measured token usage, latency, and cost.

Candidate pool (mixed-provider, matching the paper's 3-tier design):
    cheap-small     -> gpt-5.4-nano        (OpenAI)
    mid-general     -> claude-sonnet-4-6   (Anthropic)
    strong-frontier -> claude-opus-4-8     (Anthropic)

Pricing is USD per token, sourced from each provider's published rate card
as of 2026-07-02 (see paper/paper.tex references / PRICING_ASOF below).
Cost is computed from the real `usage` block each API returns, not
estimated from prompt length.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import anthropic
import openai

PRICING_ASOF = "2026-07-02"

# (input_usd_per_token, output_usd_per_token)
PRICING = {
    "gpt-5.4-nano": (0.20 / 1_000_000, 1.25 / 1_000_000),
    "claude-sonnet-4-6": (3.00 / 1_000_000, 15.00 / 1_000_000),
    "claude-opus-4-8": (5.00 / 1_000_000, 25.00 / 1_000_000),
}

CANDIDATE_TIERS = {
    "cheap-small": "gpt-5.4-nano",
    "mid-general": "claude-sonnet-4-6",
    "strong-frontier": "claude-opus-4-8",
}

_PROVIDER_OF = {
    "gpt-5.4-nano": "openai",
    "claude-sonnet-4-6": "anthropic",
    "claude-opus-4-8": "anthropic",
}


@dataclass
class LLMCallResult:
    model: str
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    tool_calls: list  # list of {"name": str, "arguments": dict}


class LiveLLMClient:
    """Thin wrapper making real calls, used by every live router adapter."""

    def __init__(self):
        self._openai = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._anthropic = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def call(
        self,
        model: str,
        system: str,
        user: str,
        tools: list | None = None,
        max_tokens: int = 512,
        trace_context: dict | None = None,
    ) -> LLMCallResult:
        provider = _PROVIDER_OF[model]
        start = time.monotonic()
        error = None
        try:
            if provider == "openai":
                result = self._call_openai(model, system, user, tools, max_tokens)
            else:
                result = self._call_anthropic(model, system, user, tools, max_tokens)
        except Exception as e:
            error = e
            result = None
        latency_ms = (time.monotonic() - start) * 1000.0

        if error is not None:
            self._log_trace(model, system, user, tools, trace_context, latency_ms, error=error)
            raise error

        price_in, price_out = PRICING[model]
        cost = result[1] * price_in + result[2] * price_out
        call_result = LLMCallResult(
            model=model,
            text=result[0],
            input_tokens=result[1],
            output_tokens=result[2],
            latency_ms=latency_ms,
            cost_usd=cost,
            tool_calls=result[3],
        )
        self._log_trace(model, system, user, tools, trace_context, latency_ms, call_result=call_result)
        return call_result

    @staticmethod
    def _log_trace(model, system, user, tools, trace_context, latency_ms, call_result=None, error=None):
        from router_benchmark.live.trace_logger import get_active_trace_logger

        logger = get_active_trace_logger()
        if logger is None:
            return
        record = {
            "context": trace_context or {},
            "request": {
                "model": model,
                "system": system,
                "user": user,
                "tools": tools,
                "pricing_asof": PRICING_ASOF,
            },
            "latency_ms": latency_ms,
        }
        if error is not None:
            record["error"] = f"{type(error).__name__}: {error}"
        else:
            record["response"] = {
                "text": call_result.text,
                "tool_calls": call_result.tool_calls,
                "input_tokens": call_result.input_tokens,
                "output_tokens": call_result.output_tokens,
                "cost_usd": call_result.cost_usd,
            }
        logger.log(record)

    def _call_openai(self, model, system, user, tools, max_tokens):
        kwargs = {}
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
        resp = self._openai.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_completion_tokens=max_tokens,
            **kwargs,
        )
        choice = resp.choices[0]
        text = choice.message.content or ""
        tool_calls = []
        if choice.message.tool_calls:
            import json

            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append({"name": tc.function.name, "arguments": args})
        return text, resp.usage.prompt_tokens, resp.usage.completion_tokens, tool_calls

    def _call_anthropic(self, model, system, user, tools, max_tokens):
        kwargs = {}
        if tools:
            kwargs["tools"] = [
                {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                for t in tools
            ]
        resp = self._anthropic.messages.create(
            model=model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
            **kwargs,
        )
        text_parts = [b.text for b in resp.content if b.type == "text"]
        tool_calls = [
            {"name": b.name, "arguments": b.input} for b in resp.content if b.type == "tool_use"
        ]
        return "".join(text_parts), resp.usage.input_tokens, resp.usage.output_tokens, tool_calls
