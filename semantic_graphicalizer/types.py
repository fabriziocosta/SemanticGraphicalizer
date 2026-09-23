"""Typed representations used by the recursive semantic graph compiler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx


@dataclass(frozen=True)
class Chunk:
    document_id: str
    chunk_id: str
    text: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class Summary:
    chunk: Chunk
    text: str


@dataclass(frozen=True)
class NormalizedText:
    chunk: Chunk
    text: str


@dataclass(frozen=True)
class Entity:
    """The universal semantic node.

    Atomic entities have ``relation=None``. Relational entities use the same
    structure with a configured relation; their arguments live in graph edges.
    """

    id: str
    type: str
    relation: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Argument:
    role: str
    entity_id: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RelationInstance:
    """Extraction-time relation record before graph materialization."""

    id: str
    type: str
    relation: str
    arguments: tuple[Argument, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StageStat:
    """Runtime and cardinality information for one pipeline stage."""

    document_id: str
    stage: str
    elapsed_seconds: float
    input_count: int
    output_count: int
    chunk_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentTrace:
    document_id: str
    text: str
    chunks: list[Chunk]
    summaries: list[Summary]
    normalized: list[NormalizedText]
    entities: list[Entity]
    relations: list[RelationInstance]
    graph: nx.MultiDiGraph
    stats: list[StageStat] = field(default_factory=list)
