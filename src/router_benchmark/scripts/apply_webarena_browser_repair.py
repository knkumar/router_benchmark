#!/usr/bin/env python3
"""Apply or verify the local WebArena browser-host repair."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_WEBARENA_DIR = Path.home() / ".local" / "share" / "router_bench_vendor" / "webarena"

ENV_OLD = """import json
import re
"""
ENV_NEW = """import json
import os
import re
"""
ENV_LAUNCH_OLD = """        self.browser = self.playwright.chromium.launch(
            headless=self.headless, slow_mo=self.slow_mo
        )
"""
ENV_LAUNCH_NEW = """        self.browser = self.playwright.chromium.launch(
            headless=self.headless,
            slow_mo=self.slow_mo,
            args=_chromium_launch_args(),
        )
"""
ENV_HELPER = """

def _chromium_launch_args() -> list[str]:
    configured = json.loads(os.environ.get(\"WEBARENA_CHROMIUM_ARGS\", \"[]\"))
    if not isinstance(configured, list) or not all(isinstance(arg, str) for arg in configured):
        raise ValueError(\"WEBARENA_CHROMIUM_ARGS must be a JSON array of strings\")
    return configured
"""
AUTO_OLD = """import argparse
import glob
import os
"""
AUTO_NEW = """import argparse
import glob
import json
import os
"""
AUTO_HELPER = """

def _chromium_launch_args() -> list[str]:
    configured = json.loads(os.environ.get(\"WEBARENA_CHROMIUM_ARGS\", \"[]\"))
    if not isinstance(configured, list) or not all(isinstance(arg, str) for arg in configured):
        raise ValueError(\"WEBARENA_CHROMIUM_ARGS must be a JSON array of strings\")
    return configured
"""
AUTO_IS_EXPIRED_OLD = """    browser = playwright.chromium.launch(headless=True, slow_mo=SLOW_MO)
"""
AUTO_IS_EXPIRED_NEW = """    browser = playwright.chromium.launch(
        headless=True, slow_mo=SLOW_MO, args=_chromium_launch_args()
    )
"""
AUTO_RENEW_OLD = """    browser = playwright.chromium.launch(headless=HEADLESS)
"""
AUTO_RENEW_NEW = """    browser = playwright.chromium.launch(
        headless=HEADLESS, args=_chromium_launch_args()
    )
"""


def _replace_once(content: str, old: str, new: str, *, path: Path) -> str:
    if new in content:
        return content
    if old not in content:
        raise ValueError(f"unrecognized upstream layout in {path}")
    return content.replace(old, new, 1)


def repaired_content(path: Path, content: str) -> str:
    if path.name == "envs.py":
        content = _replace_once(content, ENV_OLD, ENV_NEW, path=path)
        content = _replace_once(content, "\n\n@dataclass\nclass PlaywrightScript", ENV_HELPER + "\n\n@dataclass\nclass PlaywrightScript", path=path)
        return _replace_once(content, ENV_LAUNCH_OLD, ENV_LAUNCH_NEW, path=path)
    content = _replace_once(content, AUTO_OLD, AUTO_NEW, path=path)
    content = _replace_once(content, "\n\ndef is_expired", AUTO_HELPER + "\n\ndef is_expired", path=path)
    content = _replace_once(content, AUTO_IS_EXPIRED_OLD, AUTO_IS_EXPIRED_NEW, path=path)
    return _replace_once(content, AUTO_RENEW_OLD, AUTO_RENEW_NEW, path=path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--webarena-dir", type=Path, default=DEFAULT_WEBARENA_DIR)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    paths = (args.webarena_dir / "browser_env" / "envs.py", args.webarena_dir / "browser_env" / "auto_login.py")
    for path in paths:
        if not path.exists():
            raise ValueError(f"missing WebArena source file: {path}")
        current = path.read_text(encoding="utf-8")
        expected = repaired_content(path, current)
        if args.verify:
            if current != expected:
                raise ValueError(f"repair is not applied: {path}")
        elif current != expected:
            path.write_text(expected, encoding="utf-8")


if __name__ == "__main__":
    main()

