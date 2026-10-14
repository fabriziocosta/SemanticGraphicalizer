"""Typed representations used by the recursive semantic graph compiler."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any


@dataclass(frozen=True)
class DocumentInput:
    """A source document independent of its underlying data modality.

    ``content`` is intentionally not restricted to strings: image workflows
    may supply paths, bytes, or provider-specific handles without changing
    the shared graph integration layer.
    """

    document_id: str
    modality: str
    content: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Part:
    """A modality-specific piece of a document with a source locator."""

    document_id: str
    part_id: str
    modality: str
    content: Any
    source_locator: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_chunk(cls, chunk: "Chunk") -> "Part":
        return cls(
            document_id=chunk.document_id,
            part_id=chunk.chunk_id,
            modality="text",
            content=chunk.text,
            source_locator={"start_char": chunk.start_char, "end_char": chunk.end_char},
            metadata={"chunk_id": chunk.chunk_id},
        )

    def as_text_chunk(self) -> "Chunk":
        """Adapt a text part to the legacy text workflow representation."""

        if self.modality != "text" or not isinstance(self.content, str):
            raise TypeError("only text parts can be adapted to Chunk")
        start = self.source_locator.get("start_char")
        end = self.source_locator.get("end_char")
        if isinstance(start, bool) or not isinstance(start, int):
            raise ValueError("text part source_locator must contain integer start_char")
        if isinstance(end, bool) or not isinstance(end, int):
            raise ValueError("text part source_locator must contain integer end_char")
        return Chunk(self.document_id, self.part_id, self.content, start, end)


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
