"""Provider-free validation and ordering for prespecified task IDs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TypeVar


T = TypeVar("T")


def normalize_frozen_task_ids(task_ids: Sequence[str] | None, *, benchmark: str) -> tuple[str, ...] | None:
    """Validate a caller-supplied frozen task list without loading a provider.

    The adapters retain their historical ``n_tasks`` behavior when this is
    ``None``.  A supplied list is intentionally ordered: the resulting task
    list must have exactly that membership and order.
    """
    if task_ids is None:
        return None
    if isinstance(task_ids, (str, bytes)) or not isinstance(task_ids, Iterable):
        raise ValueError(f"{benchmark} frozen task IDs must be a sequence of strings")
    ids = tuple(task_ids)
    if not ids or any(not isinstance(task_id, str) or not task_id for task_id in ids):
        raise ValueError(f"{benchmark} frozen task IDs must be nonempty strings")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{benchmark} frozen task IDs contain duplicates")
    return ids


def select_frozen_records(
    records: Mapping[str, T], task_ids: tuple[str, ...], *, benchmark: str
) -> list[tuple[str, T]]:
    """Return records in frozen-ID order and fail closed on unavailable IDs."""
    missing = [task_id for task_id in task_ids if task_id not in records]
    if missing:
        raise ValueError(f"{benchmark} frozen task IDs are unavailable: {missing}")
    return [(task_id, records[task_id]) for task_id in task_ids]
