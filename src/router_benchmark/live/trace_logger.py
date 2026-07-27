"""Append-only JSONL trace logger for live runs.

Every real LLM call (full prompt, full response text/tool-calls, token
counts, cost, latency, wall-clock timestamp) is appended here, tagged with
whatever router/benchmark/task/trial context the caller provides. This is
what makes a live run reproducible/auditable after the fact: results.csv
has the graded outcome per row, but traces.jsonl has the raw evidence for
every real API call that produced it.

One JSONL file per phase run, at router_benchmark/output/live/traces/.
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path


class TraceLogger:
    def __init__(self, path: Path, append: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._f = open(self.path, "a" if append else "w")

    def log(self, record: dict) -> str:
        digest = hashlib.sha256(
            json.dumps(record, default=str, sort_keys=True).encode("utf-8")
        ).hexdigest()
        record = {"ts": datetime.now(timezone.utc).isoformat(), "digest": digest, **record}
        line = json.dumps(record, default=str)
        with self._lock:
            self._f.write(line + "\n")
            self._f.flush()
        return digest

    def close(self) -> None:
        self._f.close()


_ACTIVE: TraceLogger | None = None


def set_active_trace_logger(logger: TraceLogger | None) -> None:
    global _ACTIVE
    _ACTIVE = logger


def get_active_trace_logger() -> TraceLogger | None:
    return _ACTIVE
