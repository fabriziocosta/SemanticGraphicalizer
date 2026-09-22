"""The staged semantic compilation pipeline."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol

import networkx as nx

from .config import OntologyConfig, PromptConfig
from .exceptions import StageOutputError
from .model import ModelClient, as_model_client
from .types import (
    Chunk,
    DocumentTrace,
    EntityMention,
    NormalizedText,
    Proposition,
    PropositionLink,
    Summary,
    StateInterval,
    StageStat,
    Triple,
)


class Segmenter(Protocol):
    def segment(self, document_id: str, text: str) -> list[Chunk]:
        ...


class EntityResolver(Protocol):
    def resolve(self, mention: EntityMention) -> str:
        ...


class ParagraphWindowSegmenter:
    """Split documents on paragraph boundaries and cap long chunks."""

    def __init__(self, max_chars: int = 4000, overlap: int = 0) -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be positive")
        if overlap < 0 or overlap >= max_chars:
            raise ValueError("overlap must be between 0 and max_chars - 1")
        self.max_chars = max_chars
        self.overlap = overlap

    def segment(self, document_id: str, text: str) -> list[Chunk]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("document text must be a non-empty string")
        paragraphs = [match for match in re.finditer(r"\S.*?(?=\n\s*\n|\Z)", text, re.S)]
        if not paragraphs:
            paragraphs = [re.match(r"(?s).*", text)]  # type: ignore[list-item]

        chunks: list[Chunk] = []
        pending_start: int | None = None
        pending_end: int | None = None

        def flush() -> None:
            nonlocal pending_start, pending_end
            if pending_start is None or pending_end is None:
                return
            chunk_number = len(chunks)
            chunk_start = pending_start
            while chunk_start < pending_end and text[chunk_start].isspace():
                chunk_start += 1
            chunk_end = pending_end
            while chunk_end > chunk_start and text[chunk_end - 1].isspace():
                chunk_end -= 1
            chunks.append(Chunk(
                document_id=document_id,
                chunk_id=f"{document_id}:chunk-{chunk_number}",
                text=text[chunk_start:chunk_end],
                start_char=chunk_start,
                end_char=chunk_end,
            ))
            pending_start = pending_end = None

        for match in paragraphs:
            start, end = match.span()
            if end - start > self.max_chars:
                flush()
                window_start = start
                while window_start < end:
                    window_end = min(window_start + self.max_chars, end)
                    if window_end < end:
                        boundary = max(
                            text.rfind(" ", window_start + 1, window_end + 1),
                            text.rfind("\t", window_start + 1, window_end + 1),
                        )
                        if boundary > window_start:
                            window_end = boundary
                    chunk_start = window_start
                    while chunk_start < window_end and text[chunk_start].isspace():
                        chunk_start += 1
                    chunk_end = window_end
                    while chunk_end > chunk_start and text[chunk_end - 1].isspace():
                        chunk_end -= 1
                    chunks.append(Chunk(
                        document_id=document_id,
                        chunk_id=f"{document_id}:chunk-{len(chunks)}",
                        text=text[chunk_start:chunk_end],
                        start_char=chunk_start,
                        end_char=chunk_end,
                    ))
                    if window_end == end:
                        break
                    window_start = max(chunk_start + 1, window_end - self.overlap)
                    while window_start < end and text[window_start].isspace():
                        window_start += 1
                    if window_start < end and window_start > start and not text[window_start - 1].isspace():
                        boundary = max(
                            text.rfind(" ", start, window_start),
                            text.rfind("\t", start, window_start),
                        )
                        if boundary >= start:
                            window_start = boundary + 1
                continue
            if pending_start is None:
                pending_start, pending_end = start, end
            elif end - pending_start <= self.max_chars:
                pending_end = end
            else:
                flush()
                pending_start, pending_end = start, end
        flush()
        return chunks


class ConservativeEntityResolver:
    """Resolve only matching normalized mentions with matching ontology terms."""

    def resolve(self, mention: EntityMention) -> str:
        normalized = mention.key or mention.mention
        normalized = re.sub(r"['’]s\b", "", normalized.lower())
        normalized = re.sub(r"^\s*(?:the|a|an)\s+", "", normalized)
        normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
        normalized = normalized or "unnamed_entity"
        label = re.sub(r"[^a-z0-9]+", "_", mention.ontology_term.lower()).strip("_")
        return f"{label}::{normalized}"


_QUALIFICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "modality": {"type": ["string", "null"]},
        "negated": {"type": "boolean"},
        "attribution": {"type": ["string", "null"]},
        "temporal": {"type": ["string", "null"]},
    },
    "required": ["modality", "negated", "attribution", "temporal"],
    "additionalProperties": False,
}

_ENTITY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "mention": {"type": "string"},
        "label": {"type": "string"},
        "key": {"type": ["string", "null"]},
    },
    "required": ["mention", "label", "key"],
    "additionalProperties": False,
}

_SCHEMAS: dict[str, dict[str, Any]] = {
    "summarize": {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
        "additionalProperties": False,
    },
    "normalize": {
        "type": "object",
        "properties": {"normalized": {"type": "string"}},
        "required": ["normalized"],
        "additionalProperties": False,
    },
    "decompose": {
        "type": "object",
        "properties": {
            "propositions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "text": {"type": "string"},
                        "source_text": {"type": "string"},
                        "kind": {"type": "string", "enum": ["event", "state", "statement"]},
                        "confidence": {"type": ["number", "null"]},
                        "qualification": _QUALIFICATION_SCHEMA,
                    },
                    "required": ["id", "text", "source_text", "kind", "confidence", "qualification"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["propositions"],
        "additionalProperties": False,
    },
    "triple": {
        "type": "object",
        "properties": {
            "triples": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "proposition_id": {"type": "string"},
                        "subject": _ENTITY_SCHEMA,
                        "predicate": {"type": "string"},
                        "object": _ENTITY_SCHEMA,
                        "proposition": {"type": "string"},
                        "confidence": {"type": ["number", "null"]},
                        "qualification": _QUALIFICATION_SCHEMA,
                    },
                    "required": [
                        "id", "proposition_id", "subject", "predicate", "object",
                        "proposition", "confidence", "qualification",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["triples"],
        "additionalProperties": False,
    },
    "link": {
        "type": "object",
        "properties": {
            "links": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_proposition_id": {"type": "string"},
                        "target_proposition_id": {"type": "string"},
                        "predicate": {"type": "string"},
                        "confidence": {"type": ["number", "null"]},
                        "qualification": _QUALIFICATION_SCHEMA,
                    },
                    "required": [
                        "source_proposition_id", "target_proposition_id", "predicate",
                        "confidence", "qualification",
                    ],
                    "additionalProperties": False,
                },
            },
            "state_intervals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "state_proposition_id": {"type": "string"},
                        "starts_at": {"type": ["string", "null"]},
                        "ends_at": {"type": ["string", "null"]},
                    },
                    "required": ["state_proposition_id", "starts_at", "ends_at"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["links", "state_intervals"],
        "additionalProperties": False,
    },
}


def _format_elapsed(elapsed_seconds: float) -> str:
    """Format elapsed time using the largest practical unit."""

    if elapsed_seconds >= 60:
        return f"{elapsed_seconds / 60:.1f} min"
    if elapsed_seconds >= 1:
        return f"{elapsed_seconds:.1f} s"
    return f"{elapsed_seconds * 1000:.1f} ms"


def _as_mapping(value: Any, stage: str, document_id: str, chunk_id: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageOutputError(stage, "response must be a mapping", document_id=document_id, chunk_id=chunk_id)
    return value


def _as_text(value: Any, field: str, stage: str, document_id: str, chunk_id: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StageOutputError(stage, f"'{field}' must be a non-empty string", document_id=document_id, chunk_id=chunk_id)
    return value.strip()


def _as_float(value: Any, field: str, stage: str, document_id: str, chunk_id: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
        raise StageOutputError(stage, f"'{field}' must be a number between 0 and 1", document_id=document_id, chunk_id=chunk_id)
    return float(value)


def _as_proposition_kind(value: Any, stage: str, document_id: str, chunk_id: str) -> str:
    kind = _as_text(value, "kind", stage, document_id, chunk_id)
    if kind not in {"event", "state", "statement"}:
        raise StageOutputError(
            stage,
            "'kind' must be one of 'event', 'state', or 'statement'",
            document_id=document_id,
            chunk_id=chunk_id,
        )
    return kind


def _find_text_span(text: str, value: str) -> tuple[int, int] | None:
    """Find a case-insensitive span while allowing flexible whitespace."""

    value = value.strip()
    if not value:
        return None
    direct_start = text.casefold().find(value.casefold())
    if direct_start >= 0:
        return direct_start, direct_start + len(value)
    pattern = r"\\s+".join(re.escape(part) for part in value.split())
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.span() if match else None


def _retryable_model_error(error: Exception) -> bool:
    """Return whether a model failure is likely safe to retry."""

    if isinstance(error, (TimeoutError, ConnectionError, OSError)):
        return True
    status_code = getattr(error, "status_code", None)
    if status_code in {408, 409, 429} or isinstance(status_code, int) and status_code >= 500:
        return True
    error_name = type(error).__name__.casefold()
    return any(
        marker in error_name
        for marker in ("timeout", "connection", "ratelimit", "rate_limit", "internalserver")
    )


def _entity(value: Any, stage: str, document_id: str, chunk_id: str) -> EntityMention:
    item = _as_mapping(value, stage, document_id, chunk_id)
    mention = _as_text(item.get("mention"), "mention", stage, document_id, chunk_id)
    label = _as_text(item.get("label"), "label", stage, document_id, chunk_id)
    key = item.get("key")
    if key is not None:
        key = _as_text(key, "key", stage, document_id, chunk_id)
    return EntityMention(mention, label, key)


@dataclass
class SemanticPipeline:
    model: ModelClient
    ontology: OntologyConfig
    prompts: PromptConfig
    segmenter: Segmenter
    resolver: EntityResolver
    verbose: bool = True
    max_retries: int = 2
    retry_backoff: float = 0.25

    def __post_init__(self) -> None:
        self.model = as_model_client(self.model)
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int) or self.max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        if isinstance(self.retry_backoff, bool) or not isinstance(self.retry_backoff, (int, float)) or self.retry_backoff < 0:
            raise ValueError("retry_backoff must be non-negative")

    def _record_stat(
        self,
        stats: list[StageStat],
        *,
        document_id: str,
        stage: str,
        started: float,
        input_count: int,
        output_count: int,
        chunk_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        elapsed_seconds = time.perf_counter() - started
        stat = StageStat(
            document_id=document_id,
            stage=stage,
            elapsed_seconds=elapsed_seconds,
            input_count=input_count,
            output_count=output_count,
            chunk_id=chunk_id,
            details=details or {},
        )
        stats.append(stat)
        if not self.verbose:
            return
        scope = f" {chunk_id.rsplit(':', 1)[-1]}" if chunk_id else ""
        detail_text = ", ".join(f"{key}={value}" for key, value in stat.details.items())
        suffix = f" | {detail_text}" if detail_text else ""
        print(
            f"[{document_id}{scope}] {stage}: "
            f"{input_count} -> {output_count} | "
            f"{_format_elapsed(elapsed_seconds)}{suffix}"
        )

    def _generate(self, stage: str, prompt_values: dict[str, str], *, chunk: Chunk) -> Mapping[str, Any]:
        prompt = self.prompts.render(stage, **prompt_values)
        schema = deepcopy(_SCHEMAS[stage])
        if stage == "triple":
            triple_schema = schema["properties"]["triples"]["items"]
            all_term_ids = sorted(self.ontology.term_ids)
            relation_schemas = []
            for relation in self.ontology.relations:
                relation_schema = deepcopy(triple_schema)
                subject_schema = deepcopy(relation_schema["properties"]["subject"])
                object_schema = deepcopy(relation_schema["properties"]["object"])
                relation_schema["properties"]["predicate"] = {
                    "type": "string",
                    "enum": [relation.id],
                }
                subject_schema["properties"]["label"] = {
                    "type": "string",
                    "enum": sorted(relation.source_terms) or all_term_ids,
                }
                object_schema["properties"]["label"] = {
                    "type": "string",
                    "enum": sorted(relation.target_terms) or all_term_ids,
                }
                relation_schema["properties"]["subject"] = subject_schema
                relation_schema["properties"]["object"] = object_schema
                relation_schemas.append(relation_schema)
            schema["properties"]["triples"]["items"] = {"anyOf": relation_schemas}
        elif stage == "link":
            schema["properties"]["links"]["items"]["properties"]["predicate"] = {
                "type": "string",
                "enum": sorted(self.ontology.link_relation_ids),
            }
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.model.generate(
                    stage=stage,
                    prompt=prompt,
                    schema=schema,
                    context={"document_id": chunk.document_id, "chunk_id": chunk.chunk_id},
                )
                break
            except Exception as exc:
                last_error = exc
                if attempt == self.max_retries or not _retryable_model_error(exc):
                    raise StageOutputError(
                        stage,
                        f"model generation failed after {attempt + 1} attempt(s): {exc}",
                        document_id=chunk.document_id,
                        chunk_id=chunk.chunk_id,
                    ) from exc
                if self.retry_backoff:
                    time.sleep(self.retry_backoff * (2 ** attempt))
        else:  # pragma: no cover - loop always breaks or raises
            raise AssertionError(last_error)
        return _as_mapping(response, stage, chunk.document_id, chunk.chunk_id)

    def _summarize(self, chunk: Chunk) -> Summary:
        response = self._generate(
            "summarize", {"text": chunk.text, "ontology": self.ontology.as_prompt()}, chunk=chunk
        )
        return Summary(chunk, _as_text(response.get("summary"), "summary", "summarize", chunk.document_id, chunk.chunk_id))

    def _normalize(self, summary: Summary) -> NormalizedText:
        response = self._generate(
            "normalize", {"text": summary.text, "ontology": self.ontology.as_prompt()}, chunk=summary.chunk
        )
        return NormalizedText(summary.chunk, _as_text(response.get("normalized"), "normalized", "normalize", summary.chunk.document_id, summary.chunk.chunk_id))

    def _decompose(self, normalized: NormalizedText) -> list[Proposition]:
        response = self._generate(
            "decompose", {"text": normalized.text, "ontology": self.ontology.as_prompt()}, chunk=normalized.chunk
        )
        raw_items = response.get("propositions")
        if not isinstance(raw_items, list):
            raise StageOutputError("decompose", "'propositions' must be a list", document_id=normalized.chunk.document_id, chunk_id=normalized.chunk.chunk_id)
        propositions: list[Proposition] = []
        for index, raw_item in enumerate(raw_items):
            item = _as_mapping(raw_item, "decompose", normalized.chunk.document_id, normalized.chunk.chunk_id)
            proposition_id = item.get("id", f"p-{index}")
            proposition_id = _as_text(proposition_id, "id", "decompose", normalized.chunk.document_id, normalized.chunk.chunk_id)
            text = _as_text(item.get("text"), "text", "decompose", normalized.chunk.document_id, normalized.chunk.chunk_id)
            source_text = item.get("source_text", normalized.chunk.text)
            source_text = _as_text(source_text, "source_text", "decompose", normalized.chunk.document_id, normalized.chunk.chunk_id)
            kind = _as_proposition_kind(
                item.get("kind"),
                "decompose",
                normalized.chunk.document_id,
                normalized.chunk.chunk_id,
            )
            qualification = item.get("qualification", {})
            if not isinstance(qualification, dict):
                raise StageOutputError("decompose", "'qualification' must be a mapping", document_id=normalized.chunk.document_id, chunk_id=normalized.chunk.chunk_id)
            propositions.append(Proposition(
                proposition_id=f"{normalized.chunk.chunk_id}:{proposition_id}",
                chunk=normalized.chunk,
                text=text,
                source_text=source_text,
                kind=kind,  # type: ignore[arg-type]
                confidence=_as_float(item.get("confidence"), "confidence", "decompose", normalized.chunk.document_id, normalized.chunk.chunk_id),
                qualification=qualification,
            ))
        return propositions

    def _triples(self, normalized: NormalizedText, propositions: Sequence[Proposition]) -> list[Triple]:
        proposition_text = json.dumps([
            {
                "id": proposition.proposition_id,
                "text": proposition.text,
                "source_text": proposition.source_text,
                "kind": proposition.kind,
                "confidence": proposition.confidence,
                "qualification": proposition.qualification,
            }
            for proposition in propositions
        ], ensure_ascii=False)
        response = self._generate(
            "triple",
            {"text": proposition_text, "ontology": self.ontology.as_prompt()},
            chunk=normalized.chunk,
        )
        raw_items = response.get("triples")
        if not isinstance(raw_items, list):
            raise StageOutputError("triple", "'triples' must be a list", document_id=normalized.chunk.document_id, chunk_id=normalized.chunk.chunk_id)
        known_propositions = {proposition.proposition_id: proposition for proposition in propositions}
        triples: list[Triple] = []
        for index, raw_item in enumerate(raw_items):
            item = _as_mapping(raw_item, "triple", normalized.chunk.document_id, normalized.chunk.chunk_id)
            proposition_id = _as_text(item.get("proposition_id"), "proposition_id", "triple", normalized.chunk.document_id, normalized.chunk.chunk_id)
            proposition = known_propositions.get(proposition_id)
            if proposition is None:
                raise StageOutputError("triple", f"unknown proposition_id '{proposition_id}'", document_id=normalized.chunk.document_id, chunk_id=normalized.chunk.chunk_id)
            predicate = _as_text(item.get("predicate"), "predicate", "triple", normalized.chunk.document_id, normalized.chunk.chunk_id)
            if predicate not in self.ontology.relation_ids:
                raise StageOutputError("triple", f"unknown ontology relation '{predicate}'", document_id=normalized.chunk.document_id, chunk_id=normalized.chunk.chunk_id)
            subject = _entity(item.get("subject"), "triple", normalized.chunk.document_id, normalized.chunk.chunk_id)
            object_ = _entity(item.get("object"), "triple", normalized.chunk.document_id, normalized.chunk.chunk_id)
            unknown_labels = {
                entity.ontology_term
                for entity in (subject, object_)
                if entity.ontology_term not in self.ontology.term_ids
            }
            if unknown_labels:
                raise StageOutputError(
                    "triple",
                    f"unknown ontology term(s): {sorted(unknown_labels)}",
                    document_id=normalized.chunk.document_id,
                    chunk_id=normalized.chunk.chunk_id,
                )
            relation = next(relation for relation in self.ontology.relations if relation.id == predicate)
            if relation.source_terms and subject.ontology_term not in relation.source_terms:
                raise StageOutputError(
                    "triple",
                    f"subject label '{subject.ontology_term}' is not allowed for relation '{predicate}'",
                    document_id=normalized.chunk.document_id,
                    chunk_id=normalized.chunk.chunk_id,
                )
            if relation.target_terms and object_.ontology_term not in relation.target_terms:
                raise StageOutputError(
                    "triple",
                    f"object label '{object_.ontology_term}' is not allowed for relation '{predicate}'",
                    document_id=normalized.chunk.document_id,
                    chunk_id=normalized.chunk.chunk_id,
                )
            proposition_label = item.get("proposition", proposition.text)
            proposition_label = _as_text(proposition_label, "proposition", "triple", normalized.chunk.document_id, normalized.chunk.chunk_id)
            qualification = item.get("qualification", proposition.qualification)
            if not isinstance(qualification, dict):
                raise StageOutputError("triple", "'qualification' must be a mapping", document_id=normalized.chunk.document_id, chunk_id=normalized.chunk.chunk_id)
            source_span = _find_text_span(normalized.chunk.text, proposition.source_text)
            source_start = (
                normalized.chunk.start_char + source_span[0]
                if source_span is not None
                else normalized.chunk.start_char
            )
            source_end = (
                normalized.chunk.start_char + source_span[1]
                if source_span is not None
                else normalized.chunk.end_char
            )
            triples.append(Triple(
                triple_id=_as_text(item.get("id", f"t-{index}"), "id", "triple", normalized.chunk.document_id, normalized.chunk.chunk_id),
                proposition_id=proposition_id,
                chunk=normalized.chunk,
                subject=subject,
                predicate=predicate,
                object=object_,
                proposition=proposition_label,
                confidence=_as_float(item.get("confidence", proposition.confidence), "confidence", "triple", normalized.chunk.document_id, normalized.chunk.chunk_id),
                qualification=qualification,
                source_text=proposition.source_text,
                source_start_char=source_start,
                source_end_char=source_end,
            ))
        return triples

    def _links(
        self,
        document_id: str,
        document_text: str,
        propositions: Sequence[Proposition],
    ) -> tuple[list[PropositionLink], list[StateInterval]]:
        """Extract explicit proposition links and state intervals."""

        if not self.ontology.link_relations or "link" not in self.prompts.stages:
            return [], []
        link_chunk = Chunk(
            document_id=document_id,
            chunk_id=f"{document_id}:links",
            text=document_text,
            start_char=0,
            end_char=len(document_text),
        )
        proposition_text = json.dumps([
            {
                "id": proposition.proposition_id,
                "kind": proposition.kind,
                "text": proposition.text,
                "source_text": proposition.source_text,
            }
            for proposition in propositions
        ], ensure_ascii=False)
        response = self._generate(
            "link",
            {"text": proposition_text, "ontology": self.ontology.as_prompt()},
            chunk=link_chunk,
        )
        known = {proposition.proposition_id: proposition for proposition in propositions}
        relation_by_id = {relation.id: relation for relation in self.ontology.link_relations}
        raw_links = response.get("links")
        if not isinstance(raw_links, list):
            raise StageOutputError("link", "'links' must be a list", document_id=document_id, chunk_id=link_chunk.chunk_id)
        links: list[PropositionLink] = []
        seen_links: set[tuple[str, str, str]] = set()
        for raw_link in raw_links:
            item = _as_mapping(raw_link, "link", document_id, link_chunk.chunk_id)
            source_id = _as_text(item.get("source_proposition_id"), "source_proposition_id", "link", document_id, link_chunk.chunk_id)
            target_id = _as_text(item.get("target_proposition_id"), "target_proposition_id", "link", document_id, link_chunk.chunk_id)
            if source_id not in known or target_id not in known:
                raise StageOutputError(
                    "link",
                    "source and target proposition IDs must refer to known propositions",
                    document_id=document_id,
                    chunk_id=link_chunk.chunk_id,
                )
            predicate = _as_text(item.get("predicate"), "predicate", "link", document_id, link_chunk.chunk_id)
            relation = relation_by_id.get(predicate)
            if relation is None:
                raise StageOutputError(
                    "link",
                    f"unknown proposition link relation '{predicate}'",
                    document_id=document_id,
                    chunk_id=link_chunk.chunk_id,
                )
            signature = (source_id, target_id, predicate)
            if signature in seen_links:
                continue
            seen_links.add(signature)
            qualification = item.get("qualification", {})
            if not isinstance(qualification, dict):
                raise StageOutputError("link", "'qualification' must be a mapping", document_id=document_id, chunk_id=link_chunk.chunk_id)
            links.append(PropositionLink(
                source_proposition_id=source_id,
                target_proposition_id=target_id,
                predicate=predicate,
                category=relation.category,  # type: ignore[arg-type]
                confidence=_as_float(item.get("confidence"), "confidence", "link", document_id, link_chunk.chunk_id),
                qualification=qualification,
                provenance={
                    "document_id": document_id,
                    "proposition_id": source_id,
                    "target_proposition_id": target_id,
                },
            ))

        raw_intervals = response.get("state_intervals")
        if not isinstance(raw_intervals, list):
            raise StageOutputError("link", "'state_intervals' must be a list", document_id=document_id, chunk_id=link_chunk.chunk_id)
        intervals: list[StateInterval] = []
        seen_states: set[str] = set()
        for raw_interval in raw_intervals:
            item = _as_mapping(raw_interval, "link", document_id, link_chunk.chunk_id)
            state_id = _as_text(item.get("state_proposition_id"), "state_proposition_id", "link", document_id, link_chunk.chunk_id)
            state = known.get(state_id)
            if state is None or state.kind != "state":
                raise StageOutputError(
                    "link",
                    "state_proposition_id must refer to a state proposition",
                    document_id=document_id,
                    chunk_id=link_chunk.chunk_id,
                )
            starts_at = item.get("starts_at")
            ends_at = item.get("ends_at")
            for field, proposition_id in (("starts_at", starts_at), ("ends_at", ends_at)):
                if proposition_id is not None:
                    proposition_id = _as_text(proposition_id, field, "link", document_id, link_chunk.chunk_id)
                    target = known.get(proposition_id)
                    if target is None or target.kind != "event":
                        raise StageOutputError(
                            "link",
                            f"{field} must refer to an event proposition",
                            document_id=document_id,
                            chunk_id=link_chunk.chunk_id,
                        )
                    if field == "starts_at":
                        starts_at = proposition_id
                    else:
                        ends_at = proposition_id
            if starts_at is None and ends_at is None:
                raise StageOutputError(
                    "link",
                    "state interval must have starts_at or ends_at",
                    document_id=document_id,
                    chunk_id=link_chunk.chunk_id,
                )
            if state_id in seen_states:
                continue
            seen_states.add(state_id)
            intervals.append(StateInterval(state_id, starts_at, ends_at))
        return links, intervals

    def _integrate(
        self,
        document_id: str,
        document_text: str,
        propositions: Sequence[Proposition],
        triples: Sequence[Triple],
        links: Sequence[PropositionLink],
        state_intervals: Sequence[StateInterval],
    ) -> tuple[nx.MultiDiGraph, nx.MultiDiGraph]:
        semantic_graph = nx.MultiDiGraph(document_id=document_id, document_text=document_text)
        seen_triples: set[tuple[Any, ...]] = set()
        accepted_triples: list[Triple] = []

        def provenance(triple: Triple) -> dict[str, Any]:
            source_start = triple.source_start_char
            source_end = triple.source_end_char
            source_text = triple.source_text or triple.chunk.text
            return {
                "document_id": document_id,
                "chunk_id": triple.chunk.chunk_id,
                "start_char": source_start if source_start is not None else triple.chunk.start_char,
                "end_char": source_end if source_end is not None else triple.chunk.end_char,
                "source_text": source_text,
                "proposition_id": triple.proposition_id,
            }

        def mention_provenance(triple: Triple, entity: EntityMention) -> dict[str, Any]:
            source = triple.source_text or triple.chunk.text
            span = _find_text_span(source, entity.mention)
            source_start = triple.source_start_char
            if span is not None and source_start is not None:
                mention_start, mention_end = source_start + span[0], source_start + span[1]
            else:
                mention_start = mention_end = None
            return {
                **provenance(triple),
                "mention": entity.mention,
                "mention_start_char": mention_start,
                "mention_end_char": mention_end,
            }

        def ensure_entity(node_id: str, entity: EntityMention, triple: Triple) -> None:
            if node_id not in semantic_graph:
                semantic_graph.add_node(
                    node_id,
                    label=entity.ontology_term,
                    node_type="entity",
                    canonical_id=node_id,
                    sequence=semantic_graph.number_of_nodes(),
                    document_id=document_id,
                    document_text=document_text,
                    mentions=[],
                    provenance=[],
                )
            node = semantic_graph.nodes[node_id]
            if entity.mention not in node["mentions"]:
                node["mentions"].append(entity.mention)
            node_provenance = mention_provenance(triple, entity)
            if node_provenance not in node["provenance"]:
                node["provenance"].append(node_provenance)

        for index, triple in enumerate(triples):
            subject_id = self.resolver.resolve(triple.subject)
            object_id = self.resolver.resolve(triple.object)
            signature = (
                subject_id,
                triple.predicate,
                object_id,
                triple.source_start_char,
                triple.source_end_char,
                re.sub(r"\s+", " ", triple.source_text or triple.proposition).casefold(),
            )
            if signature in seen_triples:
                continue
            seen_triples.add(signature)
            ensure_entity(subject_id, triple.subject, triple)
            ensure_entity(object_id, triple.object, triple)
            semantic_graph.add_edge(
                subject_id,
                object_id,
                key=f"{triple.chunk.chunk_id}:{triple.triple_id}:{index}",
                label=triple.proposition,
                predicate=triple.predicate,
                edge_type="semantic",
                category="semantic",
                proposition_id=triple.proposition_id,
                confidence=triple.confidence,
                qualification=triple.qualification,
                document_id=document_id,
                document_text=document_text,
                provenance=provenance(triple),
            )
            accepted_triples.append(triple)

        graph = semantic_graph.copy()
        graph.graph["unified"] = True
        proposition_nodes: dict[str, str] = {}
        for sequence, proposition in enumerate(propositions):
            node_id = f"proposition::{proposition.proposition_id}"
            proposition_nodes[proposition.proposition_id] = node_id
            graph.add_node(
                node_id,
                label=proposition.kind,
                node_type="proposition",
                proposition_kind=proposition.kind,
                proposition_id=proposition.proposition_id,
                sequence=sequence,
                document_id=document_id,
                document_text=document_text,
                mentions=[proposition.text],
                source_text=proposition.source_text,
                provenance={
                    "document_id": document_id,
                    "chunk_id": proposition.chunk.chunk_id,
                    "start_char": proposition.chunk.start_char,
                    "end_char": proposition.chunk.end_char,
                    "source_text": proposition.source_text,
                    "proposition_id": proposition.proposition_id,
                },
                confidence=proposition.confidence,
                qualification=proposition.qualification,
            )

        role_edges: set[tuple[str, str, str]] = set()
        for index, triple in enumerate(accepted_triples):
            proposition_node = proposition_nodes.get(triple.proposition_id)
            if proposition_node is None:
                continue
            subject_id = self.resolver.resolve(triple.subject)
            object_id = self.resolver.resolve(triple.object)
            for entity_id, role in ((subject_id, "subject_of"), (object_id, "object_of")):
                signature = (entity_id, proposition_node, role)
                if signature in role_edges:
                    continue
                role_edges.add(signature)
                graph.add_edge(
                    entity_id,
                    proposition_node,
                    key=f"role:{triple.proposition_id}:{role}:{index}",
                    label=graph.nodes[proposition_node]["mentions"][0],
                    predicate=role,
                    edge_type="participation",
                    category="semantic",
                    proposition_id=triple.proposition_id,
                    document_id=document_id,
                    document_text=document_text,
                    provenance=provenance(triple),
                )

        for source, target in zip(propositions, propositions[1:]):
            graph.add_edge(
                proposition_nodes[source.proposition_id],
                proposition_nodes[target.proposition_id],
                key=f"temporal:next:{source.proposition_id}:{target.proposition_id}",
                label="",
                predicate="next_in_narrative",
                edge_type="temporal",
                category="temporal",
                proposition_id=source.proposition_id,
                target_proposition_id=target.proposition_id,
                document_id=document_id,
                document_text=document_text,
            )

        for link in links:
            graph.add_edge(
                proposition_nodes[link.source_proposition_id],
                proposition_nodes[link.target_proposition_id],
                key=f"link:{link.predicate}:{link.source_proposition_id}:{link.target_proposition_id}",
                label="",
                predicate=link.predicate,
                edge_type=link.category,
                category=link.category,
                proposition_id=link.source_proposition_id,
                target_proposition_id=link.target_proposition_id,
                confidence=link.confidence,
                qualification=link.qualification,
                document_id=document_id,
                document_text=document_text,
                provenance=link.provenance,
            )

        for interval in state_intervals:
            state_node = proposition_nodes[interval.state_proposition_id]
            for predicate, target_id in (("starts_at", interval.starts_at), ("ends_at", interval.ends_at)):
                if target_id is None:
                    continue
                graph.add_edge(
                    state_node,
                    proposition_nodes[target_id],
                    key=f"state:{predicate}:{interval.state_proposition_id}:{target_id}",
                    label="",
                    predicate=predicate,
                    edge_type="state_interval",
                    category="temporal",
                    proposition_id=interval.state_proposition_id,
                    target_proposition_id=target_id,
                    document_id=document_id,
                    document_text=document_text,
                )
        return graph, semantic_graph

    def process(self, document_id: str, text: str) -> DocumentTrace:
        total_started = time.perf_counter()
        stats: list[StageStat] = []
        if self.verbose:
            print(f"[{document_id}] processing document: chars={len(text)}")

        started = time.perf_counter()
        chunks = self.segmenter.segment(document_id, text)
        self._record_stat(
            stats,
            document_id=document_id,
            stage="segment",
            started=started,
            input_count=1,
            output_count=len(chunks),
            details={
                "input_chars": len(text),
                "chunk_chars": sum(len(chunk.text) for chunk in chunks),
            },
        )
        summaries: list[Summary] = []
        normalized: list[NormalizedText] = []
        propositions: list[Proposition] = []
        triples: list[Triple] = []
        for chunk in chunks:
            if self.verbose:
                chunk_scope = chunk.chunk_id.rsplit(":", 1)[-1]
                print(f"[{document_id} {chunk_scope}] compiling semantic stages")

            started = time.perf_counter()
            summary = self._summarize(chunk)
            self._record_stat(
                stats,
                document_id=document_id,
                stage="summarize",
                started=started,
                input_count=1,
                output_count=1,
                chunk_id=chunk.chunk_id,
                details={"input_chars": len(chunk.text), "output_chars": len(summary.text)},
            )

            started = time.perf_counter()
            normalized_text = self._normalize(summary)
            self._record_stat(
                stats,
                document_id=document_id,
                stage="normalize",
                started=started,
                input_count=1,
                output_count=1,
                chunk_id=chunk.chunk_id,
                details={"input_chars": len(summary.text), "output_chars": len(normalized_text.text)},
            )

            started = time.perf_counter()
            chunk_propositions = self._decompose(normalized_text)
            self._record_stat(
                stats,
                document_id=document_id,
                stage="decompose",
                started=started,
                input_count=1,
                output_count=len(chunk_propositions),
                chunk_id=chunk.chunk_id,
            )

            started = time.perf_counter()
            chunk_triples = self._triples(normalized_text, chunk_propositions)
            self._record_stat(
                stats,
                document_id=document_id,
                stage="triple",
                started=started,
                input_count=len(chunk_propositions),
                output_count=len(chunk_triples),
                chunk_id=chunk.chunk_id,
            )
            summaries.append(summary)
            normalized.append(normalized_text)
            propositions.extend(chunk_propositions)
            triples.extend(chunk_triples)

        started = time.perf_counter()
        links, state_intervals = self._links(document_id, text, propositions)
        self._record_stat(
            stats,
            document_id=document_id,
            stage="link",
            started=started,
            input_count=len(propositions),
            output_count=len(links),
            details={"state_intervals": len(state_intervals)},
        )

        started = time.perf_counter()
        graph, semantic_graph = self._integrate(
            document_id,
            text,
            propositions,
            triples,
            links,
            state_intervals,
        )
        self._record_stat(
            stats,
            document_id=document_id,
            stage="integrate",
            started=started,
            input_count=len(triples),
            output_count=graph.number_of_edges(),
            details={"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()},
        )
        self._record_stat(
            stats,
            document_id=document_id,
            stage="total",
            started=total_started,
            input_count=1,
            output_count=1,
            details={
                "chunks": len(chunks),
                "propositions": len(propositions),
                "triples": len(triples),
                "links": len(links),
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
            },
        )
        return DocumentTrace(
            document_id=document_id,
            text=text,
            chunks=chunks,
            summaries=summaries,
            normalized=normalized,
            propositions=propositions,
            triples=triples,
            graph=graph,
            stats=stats,
            links=links,
            state_intervals=state_intervals,
            semantic_graph=semantic_graph,
        )
