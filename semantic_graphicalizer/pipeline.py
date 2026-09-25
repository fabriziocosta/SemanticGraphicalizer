"""The staged pipeline for recursive reified semantic graphs."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import networkx as nx

from .config import OntologyConfig, PromptConfig
from .exceptions import StageOutputError
from .graph import GraphValidationError, materialize_graph
from .model import ModelClient, as_model_client
from .types import Argument, Chunk, Entity, NormalizedText, RelationInstance, StageStat, Summary


class Segmenter(Protocol):
    def segment(self, document_id: str, text: str) -> list[Chunk]: ...


class EntityResolver(Protocol):
    def resolve(self, entity_type: str, mention: str, key: str | None = None) -> str: ...


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
            start, end = pending_start, pending_end
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            chunks.append(Chunk(document_id, f"{document_id}:chunk-{len(chunks)}", text[start:end], start, end))
            pending_start = pending_end = None

        for match in paragraphs:
            start, end = match.span()
            if end - start > self.max_chars:
                flush()
                window_start = start
                while window_start < end:
                    window_end = min(window_start + self.max_chars, end)
                    if window_end < end:
                        boundary = max(text.rfind(" ", window_start + 1, window_end + 1), text.rfind("\t", window_start + 1, window_end + 1))
                        if boundary > window_start:
                            window_end = boundary
                    chunk_start, chunk_end = window_start, window_end
                    while chunk_start < chunk_end and text[chunk_start].isspace():
                        chunk_start += 1
                    while chunk_end > chunk_start and text[chunk_end - 1].isspace():
                        chunk_end -= 1
                    chunks.append(Chunk(document_id, f"{document_id}:chunk-{len(chunks)}", text[chunk_start:chunk_end], chunk_start, chunk_end))
                    if window_end == end:
                        break
                    window_start = max(chunk_start + 1, window_end - self.overlap)
                    while window_start < end and text[window_start].isspace():
                        window_start += 1
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
    """Create stable IDs from entity type and normalized surface mention."""

    def resolve(self, entity_type: str, mention: str, key: str | None = None) -> str:
        normalized = key or mention
        normalized = re.sub(r"['’]s\b", "", normalized.lower())
        normalized = re.sub(r"^\s*(?:the|a|an)\s+", "", normalized)
        normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_") or "unnamed_entity"
        label = re.sub(r"[^a-z0-9]+", "_", entity_type.lower()).strip("_")
        return f"{label}::{normalized}"


# OpenAI structured outputs require every object in a strict schema to set
# additionalProperties=false and to list all properties as required. The
# runtime Entity.attributes mapping remains open-ended; these are the
# provider-safe fields the extraction model can emit directly, while the
# pipeline continues to add arbitrary provenance and adapter metadata.
_ATTRIBUTES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "confidence": {"type": ["number", "null"]},
        "source_text": {"type": ["string", "null"]},
        "modality": {"type": ["string", "null"]},
        "negated": {"type": "boolean"},
        "attribution": {"type": ["string", "null"]},
    },
    "required": ["confidence", "source_text", "modality", "negated", "attribution"],
    "additionalProperties": False,
}
_SCHEMAS: dict[str, dict[str, Any]] = {
    "summarize": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"], "additionalProperties": False},
    "normalize": {"type": "object", "properties": {"normalized": {"type": "string"}}, "required": ["normalized"], "additionalProperties": False},
    "decompose": {
        "type": "object", "properties": {"assertions": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "text": {"type": "string"}, "source_text": {"type": "string"}, "attributes": _ATTRIBUTES_SCHEMA}, "required": ["id", "text", "source_text", "attributes"], "additionalProperties": False}}}, "required": ["assertions"], "additionalProperties": False,
    },
}


def _relation_schema(ontology: OntologyConfig) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "id": {"type": "string"}, "type": {"type": "string", "enum": sorted(ontology.term_ids)}, "relation": {"type": "string", "enum": sorted(ontology.relation_ids)},
            "arguments": {"type": "array", "items": {"type": "object", "properties": {"role": {"type": "string", "enum": sorted(ontology.argument_role_ids)}, "entity_id": {"type": "string"}, "attributes": _ATTRIBUTES_SCHEMA}, "required": ["role", "entity_id", "attributes"], "additionalProperties": False}},
            "attributes": _ATTRIBUTES_SCHEMA,
        }, "required": ["id", "type", "relation", "arguments", "attributes"], "additionalProperties": False,
    }


def _schemas(ontology: OntologyConfig) -> dict[str, dict[str, Any]]:
    entity_schema = {"type": "object", "properties": {"id": {"type": "string"}, "type": {"type": "string", "enum": sorted(ontology.term_ids)}, "mention": {"type": "string"}, "key": {"type": ["string", "null"]}, "attributes": _ATTRIBUTES_SCHEMA}, "required": ["id", "type", "mention", "key", "attributes"], "additionalProperties": False}
    relation = _relation_schema(ontology)
    return {**deepcopy(_SCHEMAS), "extract": {"type": "object", "properties": {"entities": {"type": "array", "items": entity_schema}, "relations": {"type": "array", "items": relation}}, "required": ["entities", "relations"], "additionalProperties": False}, "resolve": {"type": "object", "properties": {"relations": {"type": "array", "items": relation}}, "required": ["relations"], "additionalProperties": False}}


def _as_mapping(value: Any, stage: str, document_id: str, chunk_id: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageOutputError(stage, "response must be a mapping", document_id=document_id, chunk_id=chunk_id)
    return value


def _text(value: Any, field: str, stage: str, document_id: str, chunk_id: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StageOutputError(stage, f"'{field}' must be a non-empty string", document_id=document_id, chunk_id=chunk_id)
    return value.strip()


def _attrs(value: Any, field: str, stage: str, document_id: str, chunk_id: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StageOutputError(stage, f"'{field}' must be a mapping", document_id=document_id, chunk_id=chunk_id)
    return dict(value)


def _span(text: str, value: str) -> tuple[int, int] | None:
    start = text.casefold().find(value.casefold())
    if start >= 0:
        return start, start + len(value)
    pattern = r"\s+".join(re.escape(part) for part in value.split())
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.span() if match else None


def _retryable(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    return isinstance(error, (TimeoutError, ConnectionError, OSError)) or status in {408, 409, 429} or (isinstance(status, int) and status >= 500) or any(marker in type(error).__name__.casefold() for marker in ("timeout", "connection", "ratelimit", "internalserver"))


def _format_elapsed(elapsed_seconds: float) -> str:
    if elapsed_seconds >= 60:
        return f"{elapsed_seconds / 60:.1f} min"
    if elapsed_seconds >= 1:
        return f"{elapsed_seconds:.1f} s"
    return f"{elapsed_seconds * 1000:.1f} ms"


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

    def _stat(self, stats: list[StageStat], document_id: str, stage: str, started: float, input_count: int, output_count: int, chunk_id: str | None = None, details: dict[str, Any] | None = None) -> None:
        stat = StageStat(document_id, stage, time.perf_counter() - started, input_count, output_count, chunk_id, details or {})
        stats.append(stat)
        if self.verbose:
            scope = f" {chunk_id.rsplit(':', 1)[-1]}" if chunk_id else ""
            detail_text = ", ".join(f"{key}={value}" for key, value in stat.details.items())
            print(f"[{document_id}{scope}] {stage}: {input_count} -> {output_count} | {stat.elapsed_seconds * 1000:.1f} ms" + (f" | {detail_text}" if detail_text else ""))

    def _generate(self, stage: str, values: dict[str, str], chunk: Chunk) -> Mapping[str, Any]:
        prompt = self.prompts.render(stage, **values)
        schema = _schemas(self.ontology)[stage]
        for attempt in range(self.max_retries + 1):
            try:
                response = self.model.generate(stage=stage, prompt=prompt, schema=deepcopy(schema), context={"document_id": chunk.document_id, "chunk_id": chunk.chunk_id})
                return _as_mapping(response, stage, chunk.document_id, chunk.chunk_id)
            except Exception as exc:
                if attempt == self.max_retries or not _retryable(exc):
                    raise StageOutputError(stage, f"model generation failed after {attempt + 1} attempt(s): {exc}", document_id=chunk.document_id, chunk_id=chunk.chunk_id) from exc
                if self.retry_backoff:
                    time.sleep(self.retry_backoff * (2 ** attempt))
        raise AssertionError("unreachable")

    def _summarize(self, chunk: Chunk) -> Summary:
        response = self._generate("summarize", {"text": chunk.text, "ontology": self.ontology.as_prompt()}, chunk)
        return Summary(chunk, _text(response.get("summary"), "summary", "summarize", chunk.document_id, chunk.chunk_id))

    def _normalize(self, summary: Summary) -> NormalizedText:
        response = self._generate("normalize", {"text": summary.text, "ontology": self.ontology.as_prompt()}, summary.chunk)
        return NormalizedText(summary.chunk, _text(response.get("normalized"), "normalized", "normalize", summary.chunk.document_id, summary.chunk.chunk_id))

    def _decompose(self, normalized: NormalizedText) -> list[dict[str, Any]]:
        response = self._generate("decompose", {"text": normalized.text, "ontology": self.ontology.as_prompt()}, normalized.chunk)
        raw = response.get("assertions")
        if not isinstance(raw, list):
            raise StageOutputError("decompose", "'assertions' must be a list", document_id=normalized.chunk.document_id, chunk_id=normalized.chunk.chunk_id)
        result = []
        for index, value in enumerate(raw):
            item = _as_mapping(value, "decompose", normalized.chunk.document_id, normalized.chunk.chunk_id)
            result.append({"id": _text(item.get("id", f"assertion-{index}"), "id", "decompose", normalized.chunk.document_id, normalized.chunk.chunk_id), "text": _text(item.get("text"), "text", "decompose", normalized.chunk.document_id, normalized.chunk.chunk_id), "source_text": _text(item.get("source_text", normalized.chunk.text), "source_text", "decompose", normalized.chunk.document_id, normalized.chunk.chunk_id), "attributes": _attrs(item.get("attributes", {}), "attributes", "decompose", normalized.chunk.document_id, normalized.chunk.chunk_id)})
        return result

    def _provenance(self, chunk: Chunk, source_text: str, *, mention: str | None = None, relation_id: str | None = None) -> dict[str, Any]:
        source_span = _span(chunk.text, source_text)
        start = chunk.start_char + source_span[0] if source_span else chunk.start_char
        end = chunk.start_char + source_span[1] if source_span else chunk.end_char
        result = {"document_id": chunk.document_id, "chunk_id": chunk.chunk_id, "start_char": start, "end_char": end, "source_text": source_text}
        if mention is not None:
            mention_span = _span(source_text, mention)
            result.update({"mention": mention, "mention_start_char": start + mention_span[0] if mention_span else None, "mention_end_char": start + mention_span[1] if mention_span else None})
        if relation_id is not None:
            result["relation_id"] = relation_id
        return result

    def _extract(self, normalized: NormalizedText, assertions: Sequence[Mapping[str, Any]]) -> tuple[list[Entity], list[RelationInstance]]:
        payload = json.dumps(list(assertions), ensure_ascii=False)
        response = self._generate("extract", {"text": payload, "ontology": self.ontology.as_prompt()}, normalized.chunk)
        raw_entities, raw_relations = response.get("entities"), response.get("relations")
        if not isinstance(raw_entities, list) or not isinstance(raw_relations, list):
            raise StageOutputError("extract", "'entities' and 'relations' must be lists", document_id=normalized.chunk.document_id, chunk_id=normalized.chunk.chunk_id)
        local_entities: dict[str, str] = {}
        entities: list[Entity] = []
        for raw in raw_entities:
            item = _as_mapping(raw, "extract", normalized.chunk.document_id, normalized.chunk.chunk_id)
            local_id = _text(item.get("id"), "id", "extract", normalized.chunk.document_id, normalized.chunk.chunk_id)
            entity_type = _text(item.get("type"), "type", "extract", normalized.chunk.document_id, normalized.chunk.chunk_id)
            if entity_type not in self.ontology.term_ids:
                raise StageOutputError("extract", f"unknown Entity type '{entity_type}'", document_id=normalized.chunk.document_id, chunk_id=normalized.chunk.chunk_id)
            mention = _text(item.get("mention"), "mention", "extract", normalized.chunk.document_id, normalized.chunk.chunk_id)
            key = item.get("key")
            if key is not None:
                key = _text(key, "key", "extract", normalized.chunk.document_id, normalized.chunk.chunk_id)
            entity_id = self.resolver.resolve(entity_type, mention, key)
            local_entities[local_id] = entity_id
            attributes = _attrs(item.get("attributes", {}), "attributes", "extract", normalized.chunk.document_id, normalized.chunk.chunk_id)
            mentions = attributes.get("mentions")
            if not isinstance(mentions, list):
                mentions = []
                attributes["mentions"] = mentions
            mentions.append(mention)
            provenance = attributes.get("provenance")
            if not isinstance(provenance, list):
                provenance = []
                attributes["provenance"] = provenance
            provenance.append(self._provenance(normalized.chunk, normalized.chunk.text, mention=mention))
            entities.append(Entity(entity_id, entity_type, None, attributes))
        local_relations = {}
        for raw in raw_relations:
            item = _as_mapping(raw, "extract", normalized.chunk.document_id, normalized.chunk.chunk_id)
            local = _text(item.get("id"), "id", "extract", normalized.chunk.document_id, normalized.chunk.chunk_id)
            local_relations[local] = f"{normalized.chunk.chunk_id}:relation:{local}"
        relations: list[RelationInstance] = []
        for raw in raw_relations:
            item = _as_mapping(raw, "extract", normalized.chunk.document_id, normalized.chunk.chunk_id)
            local = _text(item.get("id"), "id", "extract", normalized.chunk.document_id, normalized.chunk.chunk_id)
            relation_id = local_relations[local]
            entity_type = _text(item.get("type"), "type", "extract", normalized.chunk.document_id, normalized.chunk.chunk_id)
            relation_name = _text(item.get("relation"), "relation", "extract", normalized.chunk.document_id, normalized.chunk.chunk_id)
            if entity_type not in self.ontology.term_ids or relation_name not in self.ontology.relation_ids:
                raise StageOutputError("extract", f"unknown relation type or relation '{entity_type}/{relation_name}'", document_id=normalized.chunk.document_id, chunk_id=normalized.chunk.chunk_id)
            raw_arguments = item.get("arguments")
            if not isinstance(raw_arguments, list):
                raise StageOutputError("extract", "relation 'arguments' must be a list", document_id=normalized.chunk.document_id, chunk_id=normalized.chunk.chunk_id)
            arguments: list[Argument] = []
            for raw_argument in raw_arguments:
                argument = _as_mapping(raw_argument, "extract", normalized.chunk.document_id, normalized.chunk.chunk_id)
                role = _text(argument.get("role"), "role", "extract", normalized.chunk.document_id, normalized.chunk.chunk_id)
                reference = _text(argument.get("entity_id"), "entity_id", "extract", normalized.chunk.document_id, normalized.chunk.chunk_id)
                target_id = local_entities.get(reference) or local_relations.get(reference)
                if target_id is None:
                    raise StageOutputError(
                        "extract",
                        f"relation '{local}' argument '{role}' references unknown object id '{reference}'; "
                        "arguments must reference an entity or relation id from this extraction response",
                        document_id=normalized.chunk.document_id,
                        chunk_id=normalized.chunk.chunk_id,
                    )
                arguments.append(Argument(role, target_id, _attrs(argument.get("attributes", {}), "attributes", "extract", normalized.chunk.document_id, normalized.chunk.chunk_id)))
            attributes = _attrs(item.get("attributes", {}), "attributes", "extract", normalized.chunk.document_id, normalized.chunk.chunk_id)
            source_text = str(attributes.get("source_text", normalized.chunk.text))
            attributes.setdefault("provenance", []).append(self._provenance(normalized.chunk, source_text, relation_id=relation_id))
            relations.append(RelationInstance(relation_id, entity_type, relation_name, tuple(arguments), attributes))
        return entities, relations

    def _resolve(self, document_id: str, text: str, entities: Sequence[Entity], relations: Sequence[RelationInstance]) -> list[RelationInstance]:
        chunk = Chunk(document_id, f"{document_id}:resolve", text, 0, len(text))
        context = {"entities": [entity.id for entity in entities], "relations": [{"id": relation.id, "type": relation.type, "relation": relation.relation, "arguments": [{"role": argument.role, "entity_id": argument.entity_id} for argument in relation.arguments]} for relation in relations]}
        response = self._generate("resolve", {"text": json.dumps(context, ensure_ascii=False), "ontology": self.ontology.as_prompt()}, chunk)
        raw_relations = response.get("relations")
        if not isinstance(raw_relations, list):
            raise StageOutputError("resolve", "'relations' must be a list", document_id=document_id, chunk_id=chunk.chunk_id)
        known = {entity.id for entity in entities} | {relation.id for relation in relations}
        raw_relation_ids = {
            _text(_as_mapping(raw, "resolve", document_id, chunk.chunk_id).get("id", f"resolved-{index}"), "id", "resolve", document_id, chunk.chunk_id): f"{document_id}:resolved:{index}"
            for index, raw in enumerate(raw_relations)
        }
        known |= set(raw_relation_ids.values())
        result: list[RelationInstance] = []
        for index, raw in enumerate(raw_relations):
            item = _as_mapping(raw, "resolve", document_id, chunk.chunk_id)
            local = _text(item.get("id", f"resolved-{index}"), "id", "resolve", document_id, chunk.chunk_id)
            relation_id = raw_relation_ids[local]
            relation_name = _text(item.get("relation"), "relation", "resolve", document_id, chunk.chunk_id)
            entity_type = _text(item.get("type"), "type", "resolve", document_id, chunk.chunk_id)
            arguments = []
            for raw_argument in item.get("arguments", []):
                argument = _as_mapping(raw_argument, "resolve", document_id, chunk.chunk_id)
                target_id = _text(argument.get("entity_id"), "entity_id", "resolve", document_id, chunk.chunk_id)
                target_id = raw_relation_ids.get(target_id, target_id)
                if target_id not in known:
                    raise StageOutputError("resolve", f"unknown Entity reference '{target_id}'", document_id=document_id, chunk_id=chunk.chunk_id)
                arguments.append(Argument(_text(argument.get("role"), "role", "resolve", document_id, chunk.chunk_id), target_id, _attrs(argument.get("attributes", {}), "attributes", "resolve", document_id, chunk.chunk_id)))
            attributes = _attrs(item.get("attributes", {}), "attributes", "resolve", document_id, chunk.chunk_id)
            attributes.setdefault("provenance", []).append(self._provenance(chunk, str(attributes.get("source_text", text)), relation_id=relation_id))
            result.append(RelationInstance(relation_id, entity_type, relation_name, tuple(arguments), attributes))
        return result

    def process(self, document_id: str, text: str) -> nx.MultiDiGraph:
        total = time.perf_counter()
        stats: list[StageStat] = []
        started = time.perf_counter()
        chunks = self.segmenter.segment(document_id, text)
        self._stat(stats, document_id, "segment", started, 1, len(chunks), details={"input_chars": len(text), "chunk_chars": sum(len(chunk.text) for chunk in chunks)})
        summaries: list[Summary] = []
        normalized: list[NormalizedText] = []
        assertions: list[dict[str, Any]] = []
        entities: list[Entity] = []
        relations: list[RelationInstance] = []
        for chunk in chunks:
            started = time.perf_counter(); summary = self._summarize(chunk); summaries.append(summary); self._stat(stats, document_id, "summarize", started, 1, 1, chunk.chunk_id)
            started = time.perf_counter(); normalized_text = self._normalize(summary); normalized.append(normalized_text); self._stat(stats, document_id, "normalize", started, 1, 1, chunk.chunk_id)
            started = time.perf_counter(); chunk_assertions = self._decompose(normalized_text); assertions.extend(chunk_assertions); self._stat(stats, document_id, "decompose", started, 1, len(chunk_assertions), chunk.chunk_id)
            started = time.perf_counter(); chunk_entities, chunk_relations = self._extract(normalized_text, chunk_assertions); entities.extend(chunk_entities); relations.extend(chunk_relations); self._stat(stats, document_id, "extract", started, len(chunk_assertions), len(chunk_entities) + len(chunk_relations), chunk.chunk_id)
        unique_entities = {entity.id: entity for entity in entities}
        started = time.perf_counter(); resolved = self._resolve(document_id, text, list(unique_entities.values()), relations); relations.extend(resolved); self._stat(stats, document_id, "resolve", started, len(entities) + len(relations), len(resolved))
        started = time.perf_counter()
        try:
            graph = materialize_graph(list(unique_entities.values()), relations, self.ontology, document_id=document_id, document_text=text)
        except GraphValidationError as exc:
            raise StageOutputError("integrate", str(exc), document_id=document_id, chunk_id=f"{document_id}:integrate") from exc
        self._stat(stats, document_id, "integrate", started, len(relations), graph.number_of_edges(), details={"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()})
        self._stat(stats, document_id, "total", total, 1, 1, details={"chunks": len(chunks), "entities": len(unique_entities), "relations": len(relations), "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()})
        graph.graph.update(
            chunks=[asdict(chunk) for chunk in chunks],
            summaries=[{"chunk": asdict(summary.chunk), "text": summary.text} for summary in summaries],
            normalized=[{"chunk": asdict(item.chunk), "text": item.text} for item in normalized],
            entities=[asdict(entity) for entity in unique_entities.values()],
            relations=[asdict(relation) for relation in relations],
            stats=[asdict(stat) for stat in stats],
        )
        return graph
