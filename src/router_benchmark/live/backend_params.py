"""Per-backend request-param sanitization. The proxy can pick any tier per
request, so params are adjusted for the chosen real model right before the
backend call. These two quirks are already documented in tau2_live.py /
webarena_live.py: opus rejects temperature==0.0, gpt-5* wants
max_completion_tokens."""
from __future__ import annotations


def sanitize_params(model: str, params: dict) -> dict:
    out = dict(params)
    if "claude-opus-4-8" in model and out.get("temperature") == 0.0:
        out.pop("temperature", None)
    if "gpt-5" in model and "max_tokens" in out and "max_completion_tokens" not in out:
        out["max_completion_tokens"] = out.pop("max_tokens")
    return out
