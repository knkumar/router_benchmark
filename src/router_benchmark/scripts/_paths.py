"""Paths shared by command modules that read repository-owned artifacts."""

from __future__ import annotations

from pathlib import Path


def repository_root() -> Path:
    """Return the checkout root when one is available, otherwise the CWD.

    Commands are normally launched from a checkout. A test image installs the
    package into site-packages, so deriving paths solely from ``__file__``
    would point at the Python installation instead of the copied protocol and
    output directories.
    """
    working_directory = Path.cwd().resolve()
    for candidate in (working_directory, *working_directory.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "protocol").is_dir():
            return candidate
    source_candidate = Path(__file__).resolve().parents[3]
    if (source_candidate / "pyproject.toml").is_file() and (source_candidate / "protocol").is_dir():
        return source_candidate
    return working_directory
