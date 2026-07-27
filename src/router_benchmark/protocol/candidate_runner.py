"""Provider-agnostic staging for the frozen candidate-outcome matrix.

The runner accepts injected, already-authorized execution functions.  It does
not import benchmark adapters or provider clients, which makes resume and
budget behavior testable without network calls.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CANDIDATE_COLUMNS = (
    "benchmark_id", "task_id", "candidate_id", "outcome_replicate",
    "execution_seed", "grader_version", "raw_trace_digest", "success",
    "provider_generation_usd", "fallback_generation_usd", "model_api_cost_usd",
    "generation_latency_ms", "failure_status", "cache_flag", "model_version",
    "pricing_snapshot",
)
CANDIDATE_KEY = ("benchmark_id", "task_id", "candidate_id", "outcome_replicate")
STAGE_MANIFEST = "stage_manifest.json"
EXTERNAL_LEDGER = "external_metered_spend.jsonl"


class CandidateBudgetExceeded(RuntimeError):
    """Raised before an injected execution would exceed the declared cap."""


@dataclass(frozen=True)
class CandidateCell:
    benchmark_id: str
    task_id: str
    candidate_id: str
    outcome_replicate: int

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.benchmark_id, self.task_id, self.candidate_id, str(self.outcome_replicate))


def _trace_digest(record: Mapping[str, Any]) -> str:
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _protocol_digest(protocol: Mapping[str, Any]) -> str:
    payload = json.dumps(protocol, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cell_key_string(cell: CandidateCell) -> str:
    return "|".join(cell.key)


def _key_tuple_string(key: tuple[str, str, str, str]) -> str:
    return "|".join(key)


class CandidateStageRunner:
    """Persist one candidate matrix cell at a time and resume by primary key."""

    def __init__(
        self,
        stage_dir: Path,
        protocol: Mapping[str, Any],
        *,
        budget_cap_usd: float,
        reservation_by_cell: Mapping[Any, float] | None = None,
        external_reserved_usd: float = 0.0,
        stage_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not math.isfinite(budget_cap_usd) or budget_cap_usd < 0:
            raise ValueError("budget_cap_usd must be a finite nonnegative number")
        if not math.isfinite(external_reserved_usd) or external_reserved_usd < 0:
            raise ValueError("external_reserved_usd must be a finite nonnegative number")
        self.stage_dir = Path(stage_dir)
        self.protocol = protocol
        self.budget_cap_usd = budget_cap_usd
        self.external_reserved_usd = external_reserved_usd
        self.stage_metadata = dict(stage_metadata or {})
        self.rows_path = self.stage_dir / "candidate_outcomes.csv"
        self.traces_path = self.stage_dir / "traces.jsonl"
        self.external_ledger_path = self.stage_dir / EXTERNAL_LEDGER
        self.manifest_path = self.stage_dir / STAGE_MANIFEST
        self._cells = [
            CandidateCell(benchmark, task_id, candidate, replicate)
            for benchmark, entry in self.protocol["benchmarks"].items()
            for task_id in entry["task_ids"]
            for candidate in self.protocol["candidates"]
            for replicate in range(entry["outcome_replicates_per_task_candidate"])
        ]
        self.reservation_by_cell = self._normalize_reservations(reservation_by_cell)
        reserved_total = sum(self.reservation_by_cell.values()) + self.external_reserved_usd
        if self.reservation_by_cell and reserved_total > self.budget_cap_usd + 1e-9:
            raise CandidateBudgetExceeded("reserved dry-run spend exceeds the budget cap")

    def _normalize_reservations(self, reservations: Mapping[Any, float] | None) -> dict[str, float]:
        if reservations is None:
            return {}
        normalized: dict[str, float] = {}
        for raw_key, raw_value in reservations.items():
            if isinstance(raw_key, tuple):
                key = "|".join(str(part) for part in raw_key)
            else:
                key = str(raw_key)
            value = float(raw_value)
            if not math.isfinite(value) or value < 0:
                raise ValueError("candidate reservations must be finite nonnegative numbers")
            normalized[key] = value
        expected = {cell_key_string(cell) for cell in self._cells}
        if set(normalized) != expected:
            missing = sorted(expected - set(normalized))
            extra = sorted(set(normalized) - expected)
            raise ValueError(f"candidate reservations must cover exactly the stage cells; missing={missing}; extra={extra}")
        return normalized

    def cells(self) -> list[CandidateCell]:
        return list(self._cells)

    def _manifest(self) -> dict[str, Any]:
        return {
            "manifest_version": 2,
            "protocol_sha256": _protocol_digest(self.protocol),
            "budget_cap_usd": self.budget_cap_usd,
            "external_reserved_usd": self.external_reserved_usd,
            "candidate_reservations_usd": {
                key: self.reservation_by_cell[key] for key in sorted(self.reservation_by_cell)
            },
            "stage_metadata": self.stage_metadata,
        }

    def _write_manifest(self) -> None:
        payload = json.dumps(self._manifest(), sort_keys=True, indent=2, default=str) + "\n"
        self.manifest_path.write_text(payload, encoding="utf-8")
        with self.manifest_path.open("r+", encoding="utf-8") as handle:
            handle.flush()
            os.fsync(handle.fileno())

    def _check_manifest(self, resume: bool) -> None:
        expected = self._manifest()
        if resume:
            if not self.manifest_path.exists():
                raise ValueError("staged candidate rows have no stage manifest")
            actual = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if actual != expected:
                raise ValueError("staged candidate manifest does not match this execution plan")
            return
        self.manifest_path.unlink(missing_ok=True)
        self._write_manifest()

    def _external_spend(self, resume: bool, completed: Mapping[tuple[str, str, str, str], dict[str, str]]) -> float:
        if not resume or not self.external_ledger_path.exists():
            if resume and completed and self.external_reserved_usd > 0:
                raise ValueError("staged candidate rows have no external metered spend ledger")
            return 0.0
        total = 0.0
        expected_keys = {cell_key_string(cell) for cell in self._cells}
        for line in self.external_ledger_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            digest = record.pop("digest", None)
            if not isinstance(digest, str) or digest != _trace_digest(record):
                raise ValueError("external metered spend ledger contains an invalid digest")
            cell_key = record.get("cell")
            if not isinstance(cell_key, list) or "|".join(str(part) for part in cell_key) not in expected_keys:
                raise ValueError("external metered spend ledger references an undeclared candidate cell")
            row_key = tuple(str(part) for part in cell_key)
            if row_key not in completed:
                raise ValueError("external metered spend ledger references a cell without a completed candidate row")
            value = float(record.get("external_metered_usd", 0.0))
            if not math.isfinite(value) or value < 0:
                raise ValueError("external metered spend ledger contains nonfinite or negative spend")
            total += value
        return total

    def _append_external_spend(self, cell: CandidateCell, external_cost: float) -> None:
        record = {"cell": cell.key, "external_metered_usd": external_cost}
        digest = _trace_digest(record)
        with self.external_ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"digest": digest, **record}, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _assert_reserved_total_fits(
        self,
        completed: Mapping[tuple[str, str, str, str], dict[str, str]],
        spent: float,
        external_spent: float,
    ) -> None:
        if external_spent > self.external_reserved_usd + 1e-9:
            raise CandidateBudgetExceeded("external metered spend exceeds its reserved budget")
        if not self.reservation_by_cell:
            if spent + max(self.external_reserved_usd, external_spent) > self.budget_cap_usd + 1e-9:
                raise CandidateBudgetExceeded("resumed staged spend and external reserve already exceed the budget cap")
            return
        completed_keys = {_key_tuple_string(key) for key in completed}
        pending_reserved = sum(
            reservation for key, reservation in self.reservation_by_cell.items()
            if key not in completed_keys
        )
        if spent + pending_reserved + max(self.external_reserved_usd, external_spent) > self.budget_cap_usd + 1e-9:
            raise CandidateBudgetExceeded("completed spend plus pending reservations would exceed the budget cap")

    def _completed_rows(self, resume: bool) -> dict[tuple[str, str, str, str], dict[str, str]]:
        if not resume or not self.rows_path.exists():
            return {}
        with self.rows_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if any(set(CANDIDATE_COLUMNS) - set(row) for row in rows):
            raise ValueError("staged candidate rows do not have the canonical columns")
        completed = {tuple(row[field] for field in CANDIDATE_KEY): row for row in rows}
        if len(completed) != len(rows):
            raise ValueError("staged candidate rows contain duplicate matrix cells")
        numeric_columns = (
            "provider_generation_usd", "fallback_generation_usd", "model_api_cost_usd", "generation_latency_ms",
        )
        for row in rows:
            values = [float(row[column]) for column in numeric_columns]
            if not all(math.isfinite(value) and value >= 0 for value in values):
                raise ValueError("staged candidate rows contain nonfinite or negative cost or latency")
            if abs(values[2] - values[0] - values[1]) > 1e-9:
                raise ValueError("staged candidate rows have invalid named cost components")
        if not self.traces_path.exists():
            raise ValueError("staged candidate rows have no trace file")
        trace_digests: set[str] = set()
        for line in self.traces_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            digest = record.pop("digest", None)
            if not isinstance(digest, str) or digest != _trace_digest(record):
                raise ValueError("staged traces contain an invalid digest")
            trace_digests.add(digest)
        if {row["raw_trace_digest"] for row in rows} - trace_digests:
            raise ValueError("staged candidate rows reference missing traces")
        return completed

    def run(
        self,
        execute: Callable[[CandidateCell], Mapping[str, Any]],
        estimate_cost_usd: Callable[[CandidateCell], float],
        *,
        resume: bool = False,
        stop_before_benchmark: str | None = None,
    ) -> list[dict[str, str]]:
        """Execute missing cells only; check the cap before every execution."""
        if self.stage_dir.exists() and not resume and any(self.stage_dir.iterdir()):
            raise ValueError("candidate stage directory must be empty unless --resume is used")
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        if (self.stage_dir / "checksums.sha256").exists() or (self.stage_dir / "manifest.json").exists():
            raise ValueError("candidate stage directory must not be a locked canonical bundle")
        self._check_manifest(resume)
        completed = self._completed_rows(resume)
        if not resume:
            self.rows_path.unlink(missing_ok=True)
            self.traces_path.unlink(missing_ok=True)
            self.external_ledger_path.unlink(missing_ok=True)
        spent = sum(float(row["model_api_cost_usd"]) for row in completed.values())
        external_spent = self._external_spend(resume, completed)
        if spent > self.budget_cap_usd:
            raise CandidateBudgetExceeded("resumed staged spend already exceeds the budget cap")
        self._assert_reserved_total_fits(completed, spent, external_spent)

        new_file = not self.rows_path.exists()
        with self.rows_path.open("a", newline="", encoding="utf-8") as rows_handle, self.traces_path.open("a", encoding="utf-8") as traces_handle:
            writer = csv.DictWriter(rows_handle, fieldnames=CANDIDATE_COLUMNS)
            if new_file:
                writer.writeheader()
                rows_handle.flush()
                os.fsync(rows_handle.fileno())
            for cell in self.cells():
                if cell.key in completed:
                    continue
                if stop_before_benchmark == cell.benchmark_id:
                    break
                self._assert_reserved_total_fits(completed, spent, external_spent)
                if self.reservation_by_cell:
                    estimate = self.reservation_by_cell[cell_key_string(cell)]
                else:
                    estimate = float(estimate_cost_usd(cell))
                if not math.isfinite(estimate) or estimate < 0:
                    raise ValueError("candidate cost estimate must be a finite nonnegative number")
                if spent + estimate + self.external_reserved_usd > self.budget_cap_usd + 1e-9:
                    raise CandidateBudgetExceeded(
                        f"candidate execution would exceed ${self.budget_cap_usd:.2f} cap"
                    )
                response = dict(execute(cell))
                provider_cost = float(response.get("provider_generation_usd", 0.0))
                fallback_cost = float(response.get("fallback_generation_usd", 0.0))
                model_cost = float(response.get("model_api_cost_usd", provider_cost + fallback_cost))
                if (not all(math.isfinite(value) for value in (provider_cost, fallback_cost, model_cost))
                        or min(provider_cost, fallback_cost, model_cost) < 0
                        or abs(model_cost - provider_cost - fallback_cost) > 1e-9):
                    raise ValueError("candidate execution returned invalid named cost components")
                latency_ms = float(response.get("generation_latency_ms", 0.0))
                if not math.isfinite(latency_ms) or latency_ms < 0:
                    raise ValueError("candidate execution returned invalid latency")
                external_cost = float(response.get("external_metered_usd", 0.0))
                if not math.isfinite(external_cost) or external_cost < 0:
                    raise ValueError("candidate execution returned invalid external metered spend")
                trace = {"cell": cell.key, "raw_trace": response.pop("raw_trace", {})}
                digest = _trace_digest(trace)
                traces_handle.write(json.dumps({"digest": digest, **trace}, default=str) + "\n")
                traces_handle.flush()
                os.fsync(traces_handle.fileno())
                if self.external_reserved_usd > 0 or external_cost > 0:
                    self._append_external_spend(cell, external_cost)
                if model_cost > estimate + 1e-9:
                    raise CandidateBudgetExceeded("candidate execution exceeded its reserved cost")
                if external_spent + external_cost > self.external_reserved_usd + 1e-9:
                    raise CandidateBudgetExceeded("external metered spend exceeds its reserved budget")
                if spent + model_cost + max(self.external_reserved_usd, external_spent + external_cost) > self.budget_cap_usd + 1e-9:
                    raise CandidateBudgetExceeded("candidate execution exceeded the budget cap")
                row = {
                    "benchmark_id": cell.benchmark_id,
                    "task_id": cell.task_id,
                    "candidate_id": cell.candidate_id,
                    "outcome_replicate": str(cell.outcome_replicate),
                    "execution_seed": str(response.get("execution_seed", cell.outcome_replicate)),
                    "grader_version": str(response.get("grader_version", self.protocol["grader_versions"][cell.benchmark_id])),
                    "raw_trace_digest": digest,
                    "success": str(bool(response.get("success", False))).lower(),
                    "provider_generation_usd": str(provider_cost),
                    "fallback_generation_usd": str(fallback_cost),
                    "model_api_cost_usd": str(model_cost),
                    "generation_latency_ms": str(latency_ms),
                    "failure_status": str(response.get("failure_status", "none")),
                    "cache_flag": "false",
                    "model_version": str(response.get("model_version", self.protocol["model_snapshots"][cell.candidate_id])),
                    "pricing_snapshot": str(response.get("pricing_snapshot", self.protocol["pricing"]["as_of"])),
                }
                writer.writerow(row)
                rows_handle.flush()
                os.fsync(rows_handle.fileno())
                completed[cell.key] = row
                spent += model_cost
                external_spent += external_cost
        return [completed[cell.key] for cell in self.cells() if cell.key in completed]
