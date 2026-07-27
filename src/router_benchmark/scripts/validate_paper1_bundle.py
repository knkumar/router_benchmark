"""Validate a locked Paper 1 canonical bundle without provider calls."""

from __future__ import annotations

import argparse
from pathlib import Path

from router_benchmark.protocol.canonical import validate_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    validate_bundle(args.bundle, args.protocol)
    print(f"Canonical bundle valid: {args.bundle}")


if __name__ == "__main__":
    main()
