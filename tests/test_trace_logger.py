from __future__ import annotations

import json

from router_benchmark.live.trace_logger import TraceLogger


def test_trace_logger_returns_and_persists_content_digest(tmp_path) -> None:
    path = tmp_path / "traces.jsonl"
    logger = TraceLogger(path)
    digest = logger.log({"request": {"model": "fixture"}, "response": {"success": True}})
    logger.close()

    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["digest"] == digest
    assert len(digest) == 64
