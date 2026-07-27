"""Lock a completed full rebuild stage into the canonical Paper 1 bundle."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from router_benchmark.protocol.bundle_writer import write_locked_bundle
from router_benchmark.protocol.protocol_tools import load_yaml
from router_benchmark.scripts.preflight_full_run import validate_full_run_protocol


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_traces(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_full_run_bundle(
    protocol: dict,
    *,
    stage_dir: Path,
    bundle_dir: Path,
) -> Path:
    validate_full_run_protocol(protocol)
    required = [
        "candidate_outcomes.csv",
        "routes.csv",
        "router_configs.json",
        "traces.jsonl",
        "external_metered_spend.jsonl",
        "stage_manifest.json",
    ]
    missing = [name for name in required if not (stage_dir / name).exists()]
    if missing:
        raise ValueError(f"full-run stage is incomplete; missing {missing}")
    router_configs = json.loads((stage_dir / "router_configs.json").read_text(encoding="utf-8"))
    candidate_rows = _read_csv(stage_dir / "candidate_outcomes.csv")
    routes = _read_csv(stage_dir / "routes.csv")
    traces = _read_traces(stage_dir / "traces.jsonl")
    external_total = sum(
        float(json.loads(line).get("external_metered_usd", 0.0) or 0.0)
        for line in (stage_dir / "external_metered_spend.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_stage_dir": str(stage_dir),
        "diagnostic_only": False,
        "full_run_budget_cap_usd": protocol["full_run_cost_reservations"]["total_cap_usd"],
        "candidate_model_api_usd": sum(float(row["model_api_cost_usd"]) for row in candidate_rows),
        "external_metered_usd": external_total,
        "router_service_usd": sum(float(row["router_service_usd"]) for row in routes),
        "excluded_historical_artifacts": [
            {
                "path": "router_benchmark/output/live/*",
                "reason": "historical router-specific outcomes are audit-only and excluded from this full canonical bundle",
            }
        ],
    }
    repair_lineage_path = stage_dir / "repair_lineage.json"
    if repair_lineage_path.exists():
        provenance["repair_lineage"] = json.loads(repair_lineage_path.read_text(encoding="utf-8"))
    return write_locked_bundle(
        bundle_dir,
        protocol,
        router_configs=dict(router_configs),
        candidate_outcomes=candidate_rows,
        routes=routes,
        traces=traces,
        provenance=provenance,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    protocol = load_yaml(args.protocol)
    bundle_dir = write_full_run_bundle(
        protocol,
        stage_dir=args.stage_dir,
        bundle_dir=args.bundle_dir,
    )
    print(f"Locked full-run bundle written to {bundle_dir}.")


if __name__ == "__main__":
    main()
