"""Typed intermediate representations used by the semantic compiler."""

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
class EntityMention:
    mention: str
    ontology_term: str
    key: str | None = None


@dataclass(frozen=True)
class Proposition:
    proposition_id: str
    chunk: Chunk
    text: str
    source_text: str
    confidence: float | None = None
    qualification: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Triple:
    triple_id: str
    proposition_id: str
    chunk: Chunk
    subject: EntityMention
    predicate: str
    object: EntityMention
    proposition: str
    confidence: float | None = None
    qualification: dict[str, Any] = field(default_factory=dict)


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
    propositions: list[Proposition]
    triples: list[Triple]
    graph: nx.MultiDiGraph
    stats: list[StageStat] = field(default_factory=list)
