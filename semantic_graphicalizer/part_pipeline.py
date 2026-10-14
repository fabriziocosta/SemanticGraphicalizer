"""Modality-neutral part processing primitives."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol, TypeVar

from .types import DocumentInput, Part


_T = TypeVar("_T")


class InputPartitioner(Protocol):
    """Split an input document into located, modality-specific parts."""

    def partition(self, document: DocumentInput) -> list[Part]: ...


def analyze_parts(
    parts: Sequence[Part],
    analyzer: Callable[[Part], _T],
    *,
    max_workers: int = 1,
) -> list[_T]:
    """Analyze modality-neutral parts with bounded, order-preserving concurrency.

    The analyzer owns modality-specific prompts and evidence handling. This
    shared helper only schedules part work and preserves input order.
    """

    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError("max_workers must be a positive integer")
    if max_workers == 1 or len(parts) < 2:
        return [analyzer(part) for part in parts]
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="sg-part") as executor:
        return list(executor.map(analyzer, parts))
