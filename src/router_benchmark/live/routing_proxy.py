"""OpenAI-compatible routing proxy. Every agent LLM call from a vendored
harness posts here; we run the configured router on the request messages,
forward to the chosen tier's real backend via litellm, log a per-request
step trace, and return an OpenAI-normalized response (streaming or not)."""
from __future__ import annotations

import json
import os
import time
import zlib
from dataclasses import dataclass, field
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from router_benchmark.live.backend_params import sanitize_params
from router_benchmark.live.llm_client import CANDIDATE_TIERS, PRICING

_PROVIDER_PREFIX = {
    "gpt-5.4-nano": "openai/gpt-5.4-nano",
    "claude-sonnet-4-6": "anthropic/claude-sonnet-4-6",
    "claude-opus-4-8": "anthropic/claude-opus-4-8",
}


def _litellm_model_for_tier(tier: str) -> str:
    return _PROVIDER_PREFIX[CANDIDATE_TIERS[tier]]


def _cost_usd(model_id: str, input_tokens: int, output_tokens: int) -> float:
    in_rate, out_rate = PRICING[model_id]
    return input_tokens * in_rate + output_tokens * out_rate


def _seed_from(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0xFFFFFFFF


@dataclass
class _ProxyState:
    registry: dict
    trace_path: str
    forced_tier: Optional[str] = None
    active_router: Any = None
    ctx: dict = field(default_factory=dict)
    step_idx: int = 0


def build_proxy_app(router_registry, trace_path, forced_tier=None, litellm_completion=None):
    """`litellm_completion` is injectable for offline tests; defaults to the
    real litellm.completion."""
    if litellm_completion is None:
        import litellm
        litellm_completion = litellm.completion

    state = _ProxyState(registry=router_registry, trace_path=trace_path, forced_tier=forced_tier)
    app = FastAPI()

    def _log_step(row: dict) -> None:
        with open(state.trace_path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    @app.post("/begin_task")
    async def begin_task(request: Request):
        body = await request.json()
        router_name = body.get("router")
        state.ctx = {k: body.get(k) for k in ("router", "benchmark", "task_id", "trial")}
        state.active_router = state.registry.get(router_name)

        # Validate router name if not in forced_tier mode (forced_tier never uses active_router)
        if state.active_router is None and state.forced_tier is None:
            valid_routers = sorted(state.registry.keys())
            return JSONResponse(
                {"error": f"unknown router '{router_name}'; valid: {valid_routers}"},
                status_code=400
            )

        state.step_idx = 0
        return {"ok": True}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        messages = body.get("messages", [])
        tools = body.get("tools")
        stream = bool(body.get("stream", False))
        step_idx = state.step_idx
        state.step_idx += 1

        rng = _seed_from(state.ctx.get("router"), state.ctx.get("benchmark"),
                         state.ctx.get("task_id"), state.ctx.get("trial"), step_idx)

        route_start = time.monotonic()
        if state.forced_tier is not None:
            tier, confidence, fallback = state.forced_tier, 1.0, False
        else:
            decision = state.active_router.route_request(
                messages=messages, tools=tools, candidates=None, context=state.ctx, rng=rng)
            tier, confidence, fallback = (decision.selected_candidate, decision.confidence,
                                          decision.fallback_used)
        routing_latency_ms = (time.monotonic() - route_start) * 1000.0

        model = _litellm_model_for_tier(tier)
        model_id = CANDIDATE_TIERS[tier]

        call_kwargs = sanitize_params(model, {"temperature": body.get("temperature"),
                                              "max_tokens": body.get("max_tokens")})
        call_kwargs = {k: v for k, v in call_kwargs.items() if v is not None}
        if tools:
            call_kwargs["tools"] = tools
            if body.get("tool_choice") is not None:
                call_kwargs["tool_choice"] = body["tool_choice"]

        base_row = {**state.ctx, "step_idx": step_idx, "chosen_tier": tier,
                    "chosen_model": model_id, "confidence": confidence,
                    "fallback_used": fallback, "routing_latency_ms": routing_latency_ms,
                    "messages_digest": _seed_from(json.dumps(messages, sort_keys=True, default=str))}

        if not stream:
            def _is_400(exc):
                return getattr(exc, "status_code", None) == 400 or "badrequest" in type(exc).__name__.lower()

            start = time.monotonic()
            try:
                resp = litellm_completion(model=model, messages=messages, **call_kwargs)
            except Exception as exc:  # noqa: BLE001 - surface after one sanitized retry
                if not _is_400(exc):
                    _log_step({**base_row, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
                               "latency_ms": (time.monotonic() - start) * 1000.0,
                               "usage_estimated": True, "error": str(exc), "ts": time.time()})
                    return JSONResponse({"error": {"message": str(exc)}}, status_code=502)
                # one retry, SAME tier, params re-sanitized (idempotent)
                retry_kwargs = sanitize_params(model, call_kwargs)
                try:
                    resp = litellm_completion(model=model, messages=messages, **retry_kwargs)
                except Exception as exc2:  # noqa: BLE001
                    _log_step({**base_row, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
                               "latency_ms": (time.monotonic() - start) * 1000.0,
                               "usage_estimated": True, "error": str(exc2), "ts": time.time()})
                    return JSONResponse({"error": {"message": str(exc2)}}, status_code=502)
            latency_ms = (time.monotonic() - start) * 1000.0
            data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
            usage = data.get("usage")
            if usage:
                in_tok = int(usage.get("prompt_tokens", 0) or 0)
                out_tok = int(usage.get("completion_tokens", 0) or 0)
                estimated = False
            else:
                in_tok, out_tok, estimated = 0, 0, True
            _log_step({**base_row, "input_tokens": in_tok, "output_tokens": out_tok,
                       "cost_usd": _cost_usd(model_id, in_tok, out_tok),
                       "latency_ms": latency_ms, "usage_estimated": estimated, "ts": time.time()})
            return JSONResponse(data)

        return StreamingResponse(
            _stream_and_log(litellm_completion, model, messages, call_kwargs,
                            base_row, model_id, _log_step),
            media_type="text/event-stream")

    return app


def _stream_and_log(litellm_completion, model, messages, call_kwargs,
                    base_row, model_id, log_step):
    start = time.monotonic()
    stream = litellm_completion(model=model, messages=messages, stream=True,
                                stream_options={"include_usage": True}, **call_kwargs)
    usage = None
    for chunk in stream:
        data = chunk.model_dump() if hasattr(chunk, "model_dump") else dict(chunk)
        if data.get("usage"):
            usage = data["usage"]
        yield f"data: {json.dumps(data, default=str)}\n\n"
    yield "data: [DONE]\n\n"
    latency_ms = (time.monotonic() - start) * 1000.0
    if usage:
        in_tok = int(usage.get("prompt_tokens", 0) or 0)
        out_tok = int(usage.get("completion_tokens", 0) or 0)
        estimated = False
    else:
        in_tok, out_tok, estimated = 0, 0, True
    log_step({**base_row, "input_tokens": in_tok, "output_tokens": out_tok,
              "cost_usd": _cost_usd(model_id, in_tok, out_tok),
              "latency_ms": latency_ms, "usage_estimated": estimated, "ts": time.time()})


def build_router_registry() -> dict:
    """Lazily build the four live routers, keyed by their canonical `.name`
    (e.g. 'LiteLLM Router (live)') -- the SAME identity tau2's /begin_task posts
    via decision.metadata['router_name'] and the identity results.csv uses.
    Imported here (not at module top) so a proxy needing one router does not
    construct all four."""
    from router_benchmark.live.live_routers import LiteLLMRouterLive, AurelioSemanticRouterLive
    from router_benchmark.live.routellm_live import RouteLLMLive
    from router_benchmark.live.vllm_sr_live import VLLMSemanticRouterLive
    routers = [LiteLLMRouterLive(), AurelioSemanticRouterLive(), RouteLLMLive(),
               VLLMSemanticRouterLive()]
    return {r.name: r for r in routers}


def main() -> None:
    import uvicorn
    trace_path = os.environ["ROUTER_BENCHMARK_PROXY_TRACE"]
    forced = os.environ.get("ROUTER_BENCHMARK_PROXY_FORCE_TIER") or None
    port = int(os.environ.get("ROUTER_BENCHMARK_PROXY_PORT", "8010"))
    only = os.environ.get("ROUTER_BENCHMARK_PROXY_ROUTER")  # optional: canonical .name of one router
    registry = build_router_registry()
    if only:
        registry = {only: registry[only]}
    app = build_proxy_app(registry, trace_path, forced_tier=forced)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
