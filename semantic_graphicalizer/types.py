"""Typed intermediate representations used by the semantic compiler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import networkx as nx


PropositionKind = Literal["event", "state", "statement"]
LinkCategory = Literal["temporal", "causal"]


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
    kind: PropositionKind
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
    source_text: str = ""
    source_start_char: int | None = None
    source_end_char: int | None = None


@dataclass(frozen=True)
class PropositionLink:
    source_proposition_id: str
    target_proposition_id: str
    predicate: str
    category: LinkCategory
    confidence: float | None = None
    qualification: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StateInterval:
    state_proposition_id: str
    starts_at: str | None = None
    ends_at: str | None = None


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
    links: list[PropositionLink] = field(default_factory=list)
    state_intervals: list[StateInterval] = field(default_factory=list)
    semantic_graph: nx.MultiDiGraph | None = None
