#!/usr/bin/env python3
"""Create the replacement protocol for a repaired WebArena execution."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from router_benchmark.protocol.protocol_tools import load_yaml
from router_benchmark.scripts.preflight_full_run import validate_full_run_protocol


REPAIRED_PROTOCOL_ID = "paper1-rebuild-webarena-repair-v2"
REPAIRED_GRADER_VERSION = (
    "WebArena git dce04686a56253aefba7b18a4fa0937cf1dc987b; "
    "Playwright 1.32.1 Chromium 112.0.5615.29 (revision 1055); "
    "GitLab image sha256:cb52e8185fbc421edd9a5bef523d6504edef3c21418be2ceaf825904b3013be0; "
    "shopping image sha256:ccff8c1772be884313edad94136d2a4048020300a0fc169781c50a02aa8bd206"
)


def repaired_protocol(source: dict) -> dict:
    """Return a full-scope protocol differing only in WebArena environment provenance."""
    result = dict(source)
    result["protocol_id"] = REPAIRED_PROTOCOL_ID
    result["study_status"] = "frozen_scope_repaired_webarena_pending_execution"
    graders = dict(result["grader_versions"])
    graders["WebArena (live)"] = REPAIRED_GRADER_VERSION
    result["grader_versions"] = graders
    result["webarena_environment_repair"] = {
        "reason": "The prior run could not launch its pinned Chromium revision.",
        "browser_install": "Playwright 1.32.1 Chromium revision 1055 installed in the pinned WebArena environment.",
        "shopping_host_routing": "Chromium resolves metis.lti.cs.cmu.edu to the local Shopping service only for WebArena browser processes.",
        "replacement_scope": "Replace only WebArena candidate and route rows; retain the validated, unchanged three-benchmark rows from the approved full stage.",
    }
    validate_full_run_protocol(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite existing protocol: {args.output}")
    protocol = repaired_protocol(load_yaml(args.source))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()

