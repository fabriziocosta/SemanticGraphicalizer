"""Ontology-guided graphicalization for local scene images."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import time
from typing import Any

import networkx as nx
import yaml

from .config import OntologyConfig, PromptStage, load_ontology
from .graph import GraphValidationError, materialize_graph
from .model import (
    DEFAULT_OPENAI_IMAGE_MODEL,
    ModelContent,
    ModelRequest,
    OpenAIModelClient,
    RequestModelClient,
    as_model_client,
)
from .types import Argument, DocumentInput, Entity, Part, RelationInstance, StageStat


IMAGE_DOMAIN_ROOT = Path(__file__).resolve().parents[1] / "domains" / "image"
DEFAULT_IMAGE_ONTOLOGY = IMAGE_DOMAIN_ROOT / "configs" / "ontologies" / "visual.yaml"
DEFAULT_IMAGE_PROMPTS = IMAGE_DOMAIN_ROOT / "configs" / "prompts" / "scene_analysis.yaml"


class ImagePromptConfig:
    """Image workflow prompts, independent of the text prompt stages."""

    def __init__(self, domain: str, version: str, stages: Mapping[str, PromptStage]) -> None:
        self.domain = domain
        self.version = version
        self.stages = dict(stages)

    def render(self, stage: str, **values: str) -> str:
        if stage not in self.stages:
            raise ValueError(f"image prompt pack has no stage '{stage}'")
        template = self.stages[stage]
        try:
            system = template.system.format(**values)
            instruction = template.instruction.format(**values)
        except KeyError as exc:
            raise ValueError(f"image prompt stage '{stage}' references missing placeholder {exc}") from exc
        return f"SYSTEM:\n{system}\n\nINSTRUCTION:\n{instruction}\n\nOUTPUT:\n{template.output}"


def load_image_prompts(source: str | Path | Mapping[str, Any] | ImagePromptConfig) -> ImagePromptConfig:
    if isinstance(source, ImagePromptConfig):
        return source
    if isinstance(source, Mapping):
        raw = dict(source)
    else:
        path = Path(source)
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError(f"could not read image prompts '{path}': {exc}") from exc
    if not isinstance(raw, Mapping) or set(raw) != {"domain", "version", "stages"}:
        raise ValueError("image prompt pack must contain domain, version, and stages")
    stages_raw = raw["stages"]
    expected = {"inventory", "analyze_entity", "relate"}
    if not isinstance(stages_raw, Mapping) or set(stages_raw) != expected:
        raise ValueError(f"image prompts.stages must contain exactly {sorted(expected)}")
    stages: dict[str, PromptStage] = {}
    for name, value in stages_raw.items():
        if not isinstance(value, Mapping) or set(value) != {"system", "instruction", "output"}:
            raise ValueError(f"image prompts.stages.{name} must contain system, instruction, and output")
        if not all(isinstance(value[key], str) and value[key].strip() for key in value):
            raise ValueError(f"image prompts.stages.{name} values must be non-empty strings")
        stages[name] = PromptStage(value["system"], value["instruction"], value["output"])
    for field in ("domain", "version"):
        if not isinstance(raw[field], str) or not raw[field].strip():
            raise ValueError(f"image prompts.{field} must be a non-empty string")
    return ImagePromptConfig(raw["domain"], raw["version"], stages)


def _object(value: Any, stage: str, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"image stage '{stage}' returned invalid '{field}' (expected an object)")
    return value


def _required_text(value: Any, stage: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"image stage '{stage}' returned invalid '{field}' (expected non-empty text)")
    return value.strip()


def _confidence_schema() -> dict[str, Any]:
    return {"type": ["number", "null"], "minimum": 0, "maximum": 1}


def _inventory_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "scene_description": {"type": "string"},
            "regions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "label": {"type": "string"},
                        "bbox": {
                            "type": "array",
                            "items": {"type": "number", "minimum": 0, "maximum": 1},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "confidence": _confidence_schema(),
                    },
                    "required": ["id", "label", "bbox", "confidence"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["scene_description", "regions"],
        "additionalProperties": False,
    }


def _analysis_schema(ontology: OntologyConfig) -> dict[str, Any]:
    properties = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "value": {"type": "string"}},
        "required": ["name", "value"],
        "additionalProperties": False,
    }
    entity = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "type": {"type": "string", "enum": sorted(ontology.term_ids)},
            "mention": {"type": "string"},
            "key": {"type": ["string", "null"]},
            "description": {"type": "string"},
            "properties": {"type": "array", "items": properties},
            "confidence": _confidence_schema(),
        },
        "required": ["id", "type", "mention", "key", "description", "properties", "confidence"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"entities": {"type": "array", "items": entity}},
        "required": ["entities"],
        "additionalProperties": False,
    }


def _relation_schema(ontology: OntologyConfig, entity_ids: list[str]) -> dict[str, Any]:
    argument = {
        "type": "object",
        "properties": {
            "role": {"type": "string", "enum": sorted(ontology.argument_role_ids)},
            "entity_id": {"type": "string", "enum": entity_ids},
        },
        "required": ["role", "entity_id"],
        "additionalProperties": False,
    }
    relation = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "type": {"type": "string", "enum": sorted(ontology.term_ids)},
            "relation": {"type": "string", "enum": sorted(ontology.relation_ids)},
            "arguments": {"type": "array", "items": argument},
            "description": {"type": "string"},
            "confidence": _confidence_schema(),
        },
        "required": ["id", "type", "relation", "arguments", "description", "confidence"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"relations": {"type": "array", "items": relation}},
        "required": ["relations"],
        "additionalProperties": False,
    }


def _bbox_pixels(raw_bbox: Any, width: int, height: int) -> tuple[list[float], tuple[int, int, int, int]]:
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        raise ValueError("bbox must contain normalized [x_min, y_min, x_max, y_max]")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw_bbox):
        raise ValueError("bbox coordinates must be numeric")
    box = [float(value) for value in raw_bbox]
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in box):
        raise ValueError("bbox coordinates must be finite values between 0 and 1")
    x_min, y_min, x_max, y_max = box
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("bbox must have positive width and height")
    left = max(0, min(width, math.floor(x_min * width)))
    top = max(0, min(height, math.floor(y_min * height)))
    right = max(0, min(width, math.ceil(x_max * width)))
    bottom = max(0, min(height, math.ceil(y_max * height)))
    if right <= left or bottom <= top:
        raise ValueError("bbox produces an empty pixel crop")
    return box, (left, top, right, bottom)


class ImageGraphicalizer:
    """Extract ontology-grounded graphs from local scene images."""

    def __init__(
        self,
        ontology: str | Path | Mapping[str, Any] | OntologyConfig = DEFAULT_IMAGE_ONTOLOGY,
        prompts: str | Path | Mapping[str, Any] | ImagePromptConfig = DEFAULT_IMAGE_PROMPTS,
        model: Any = None,
        *,
        crop_dir: str | Path = Path("domains/image/data/processed/regions"),
        scene_type: str = "scene",
        max_retries: int = 2,
        retry_backoff: float = 0.25,
        verbose: bool = True,
    ) -> None:
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        if isinstance(retry_backoff, bool) or not isinstance(retry_backoff, (int, float)) or retry_backoff < 0:
            raise ValueError("retry_backoff must be non-negative")
        self.ontology = load_ontology(ontology)
        self.prompts = load_image_prompts(prompts)
        if scene_type not in self.ontology.term_ids:
            raise ValueError(f"scene_type '{scene_type}' is not defined by the ontology")
        self.scene_type = scene_type
        self.crop_dir = Path(crop_dir)
        self.max_retries = max_retries
        self.retry_backoff = float(retry_backoff)
        self.verbose = verbose
        self.model: RequestModelClient | None = as_model_client(model) if model is not None else None

    @staticmethod
    def _image_backend():
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise ImportError("ImageGraphicalizer requires Pillow; install semantic-graphicalizer[image]") from exc
        return Image

    def _run_request(self, request: ModelRequest) -> Mapping[str, Any]:
        if self.model is None:
            self.model = as_model_client(OpenAIModelClient(model=DEFAULT_OPENAI_IMAGE_MODEL))
        for attempt in range(self.max_retries + 1):
            try:
                result = self.model.generate_request(request)
                if not isinstance(result, Mapping):
                    raise ValueError(f"image stage '{request.stage}' must return an object")
                return result
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                retryable = (
                    isinstance(exc, (TimeoutError, ConnectionError, OSError))
                    or status in {408, 409, 429}
                    or (isinstance(status, int) and status >= 500)
                )
                if not retryable or attempt >= self.max_retries:
                    raise
                if self.retry_backoff:
                    time.sleep(self.retry_backoff * 2**attempt)
        raise AssertionError("unreachable")

    def _request(
        self,
        stage: str,
        prompt_values: dict[str, str],
        schema: Mapping[str, Any],
        document_id: str,
        *,
        part: Part | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        prompt = self.prompts.render(stage, **prompt_values)
        contents = () if part is None else (
            ModelContent("image", part.content, {"detail": "high", **dict(part.metadata)}),
        )
        request_context = {"document_id": document_id, **dict(context or {})}
        if part is not None:
            request_context.update({"part_id": part.part_id, "source_locator": dict(part.source_locator)})
        return self._run_request(ModelRequest(stage, prompt, schema, request_context, contents))

    def _partition(self, document: DocumentInput, image_size: tuple[int, int], regions: list[Any]) -> tuple[list[Part], list[str]]:
        Image = self._image_backend()
        width, height = image_size
        parts: list[Part] = []
        warnings: list[str] = []
        target_dir = self.crop_dir / document.document_id
        target_dir.mkdir(parents=True, exist_ok=True)
        for index, raw in enumerate(regions):
            item = _object(raw, "inventory", f"regions[{index}]")
            region_id = _required_text(item.get("id"), "inventory", f"regions[{index}].id")
            label = _required_text(item.get("label"), "inventory", f"regions[{index}].label")
            try:
                normalized, pixels = _bbox_pixels(item.get("bbox"), width, height)
            except ValueError as exc:
                warnings.append(f"region {region_id!r} skipped: {exc}")
                continue
            source = Path(document.content)
            safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", region_id).strip("_.") or f"region-{index}"
            crop_path = target_dir / f"{index:03d}-{safe_id}.png"
            with Image.open(source) as original:
                original.crop(pixels).save(crop_path, format="PNG")
            parts.append(Part(
                document.document_id,
                f"{document.document_id}:region:{index}:{safe_id}",
                "image",
                crop_path,
                source_locator={
                    "source_image": str(source),
                    "normalized_bbox": normalized,
                    "pixel_bbox": list(pixels),
                },
                metadata={"label": label, "confidence": item.get("confidence"), "crop_path": str(crop_path)},
            ))
        return parts, warnings

    def _inventory(self, document: DocumentInput, document_id: str) -> tuple[str, list[Any]]:
        ontology = self.ontology.as_prompt()
        whole_image = Part(
            document_id,
            f"{document_id}:whole-image",
            "image",
            document.content,
            source_locator={"whole_image": True},
            metadata={"mime_type": "image/png"},
        )
        response = self._request(
            "inventory",
            {"ontology": ontology},
            _inventory_schema(),
            document_id,
            part=whole_image,
            context={"modality": "image", "source_image": str(document.content)},
        )
        description = _required_text(response.get("scene_description"), "inventory", "scene_description")
        regions = response.get("regions")
        if not isinstance(regions, list):
            raise ValueError("image stage 'inventory' returned invalid 'regions' (expected a list)")
        return description, regions

    def _analyze_part(self, part: Part, scene_summary: str) -> list[Entity]:
        response = self._request(
            "analyze_entity",
            {"ontology": self.ontology.as_prompt(), "scene_summary": scene_summary, "region_label": str(part.metadata["label"])},
            _analysis_schema(self.ontology),
            part.document_id,
            part=part,
            context={"scene_summary": scene_summary},
        )
        raw_entities = response.get("entities")
        if not isinstance(raw_entities, list):
            raise ValueError("image stage 'analyze_entity' returned invalid 'entities' (expected a list)")
        entities: list[Entity] = []
        local_ids: set[str] = set()
        for index, raw in enumerate(raw_entities):
            item = _object(raw, "analyze_entity", f"entities[{index}]")
            local_id = _required_text(item.get("id"), "analyze_entity", f"entities[{index}].id")
            if local_id in local_ids:
                raise ValueError(f"image stage 'analyze_entity' returned duplicate entity id '{local_id}'")
            local_ids.add(local_id)
            entity_type = _required_text(item.get("type"), "analyze_entity", f"entities[{index}].type")
            if entity_type not in self.ontology.term_ids:
                raise ValueError(f"image stage 'analyze_entity' returned unknown ontology type '{entity_type}'")
            mention = _required_text(item.get("mention"), "analyze_entity", f"entities[{index}].mention")
            key = item.get("key")
            region_key = part.part_id.split(":region:", 1)[-1]
            entity_id = f"{part.document_id}:entity:{region_key}:{local_id}"
            properties = item.get("properties", [])
            if not isinstance(properties, list):
                raise ValueError("image entity properties must be a list")
            attributes: dict[str, Any] = {
                "mentions": [mention],
                "description": str(item.get("description") or ""),
                "properties": {},
                "confidence": item.get("confidence"),
                "provenance": [{"source_locator": dict(part.source_locator), "crop_path": str(part.content)}],
            }
            for prop_index, raw_property in enumerate(properties):
                prop = _object(raw_property, "analyze_entity", f"entities[{index}].properties[{prop_index}]")
                name = _required_text(prop.get("name"), "analyze_entity", "property name")
                value = _required_text(prop.get("value"), "analyze_entity", f"property {name}")
                attributes["properties"][name] = value
            if key is not None:
                attributes["key"] = _required_text(key, "analyze_entity", f"entities[{index}].key")
            entities.append(Entity(entity_id, entity_type, None, attributes))
        return entities

    def _relate(self, document_id: str, scene_summary: str, entities: list[Entity]) -> list[RelationInstance]:
        entity_ids = [entity.id for entity in entities]
        serialized_entities = [
            {"id": entity.id, "type": entity.type, "attributes": entity.attributes}
            for entity in entities
        ]
        response = self._request(
            "relate",
            {"ontology": self.ontology.as_prompt(), "scene_summary": scene_summary, "entities": json.dumps(serialized_entities, ensure_ascii=False)},
            _relation_schema(self.ontology, entity_ids),
            document_id,
            context={"entity_ids": entity_ids},
        )
        raw_relations = response.get("relations")
        if not isinstance(raw_relations, list):
            raise ValueError("image stage 'relate' returned invalid 'relations' (expected a list)")
        relations: list[RelationInstance] = []
        for index, raw in enumerate(raw_relations):
            item = _object(raw, "relate", f"relations[{index}]")
            local_id = _required_text(item.get("id"), "relate", f"relations[{index}].id")
            relation_type = _required_text(item.get("type"), "relate", f"relations[{index}].type")
            if relation_type not in self.ontology.term_ids:
                raise ValueError(f"image stage 'relate' returned unknown ontology type '{relation_type}'")
            relation_name = _required_text(item.get("relation"), "relate", f"relations[{index}].relation")
            if self.ontology.relation(relation_name) is None:
                raise ValueError(f"image stage 'relate' returned unknown relation '{relation_name}'")
            raw_arguments = item.get("arguments")
            if not isinstance(raw_arguments, list):
                raise ValueError("image relation arguments must be a list")
            arguments: list[Argument] = []
            for argument_index, raw_argument in enumerate(raw_arguments):
                argument = _object(raw_argument, "relate", f"relations[{index}].arguments[{argument_index}]")
                role = _required_text(argument.get("role"), "relate", "argument role")
                entity_id = _required_text(argument.get("entity_id"), "relate", "argument entity_id")
                if role not in self.ontology.argument_role_ids:
                    raise ValueError(f"image stage 'relate' returned unknown argument role '{role}'")
                if entity_id not in entity_ids:
                    raise ValueError(f"image stage 'relate' referenced unknown entity '{entity_id}'")
                arguments.append(Argument(role, entity_id))
            attributes = {
                "description": str(item.get("description") or ""),
                "confidence": item.get("confidence"),
                "provenance": [{"stage": "relate", "document_id": document_id}],
            }
            relations.append(RelationInstance(
                f"{document_id}:relation:{index}:{local_id}",
                relation_type,
                relation_name,
                tuple(arguments),
                attributes,
            ))
        return relations

    @staticmethod
    def _document_id(payload: bytes, seen: dict[str, int]) -> str:
        digest = sha256(payload).hexdigest()[:12]
        base = f"image-{digest}"
        duplicate = seen.get(base, 0)
        seen[base] = duplicate + 1
        return base if duplicate == 0 else f"{base}-{duplicate}"

    def _process(self, path: Path, seen: dict[str, int]) -> nx.MultiDiGraph:
        if not path.is_file():
            raise ValueError(f"image input must be an existing local file: {path}")
        Image = self._image_backend()
        payload = path.read_bytes()
        document_id = self._document_id(payload, seen)
        with Image.open(path) as image:
            image.load()
            width, height = image.size
        document = DocumentInput(
            document_id,
            "image",
            path,
            metadata={"source_image": str(path), "sha256": sha256(payload).hexdigest(), "width": width, "height": height},
        )
        started = time.perf_counter()
        scene_summary, regions = self._inventory(document, document_id)
        stats = [StageStat(document_id, "inventory", time.perf_counter() - started, 1, len(regions))]
        parts, warnings = self._partition(document, (width, height), regions)
        entities = [Entity(
            f"{document_id}:scene",
            self.scene_type,
            None,
            {"mentions": ["scene"], "description": scene_summary, "provenance": [{"source_image": str(path), "source_locator": {"whole_image": True}}]},
        )]
        for part in parts:
            started = time.perf_counter()
            extracted = self._analyze_part(part, scene_summary)
            entities.extend(extracted)
            stats.append(StageStat(document_id, "analyze_entity", time.perf_counter() - started, 1, len(extracted), part.part_id))
        started = time.perf_counter()
        relations = self._relate(document_id, scene_summary, entities)
        stats.append(StageStat(document_id, "relate", time.perf_counter() - started, len(entities), len(relations)))
        started = time.perf_counter()
        try:
            graph = materialize_graph(entities, relations, self.ontology, document_id=document_id, document_text=scene_summary)
        except GraphValidationError:
            raise
        stats.append(StageStat(document_id, "integrate", time.perf_counter() - started, len(entities) + len(relations), graph.number_of_edges()))
        graph.graph.update(
            modality="image",
            source_image=str(path),
            image_sha256=document.metadata["sha256"],
            image_size={"width": width, "height": height},
            scene_summary=scene_summary,
            parts=[{**asdict(part), "content": str(part.content)} for part in parts],
            warnings=warnings,
            stats=[asdict(stat) for stat in stats],
        )
        if self.verbose:
            print(f"[ImageGraphicalizer] {path.name}: {len(parts)} regions, {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        return graph

    def transform(self, X: str | Path | Iterable[str | Path]) -> list[nx.MultiDiGraph]:
        """Return one canonical graph for each local image path."""

        if isinstance(X, (str, Path)):
            paths = [Path(X)]
        else:
            try:
                paths = [Path(value) for value in X]
            except TypeError as exc:
                raise ValueError("X must be a local image path or iterable of paths") from exc
        if not paths:
            return []
        seen: dict[str, int] = {}
        return [self._process(path, seen) for path in paths]

    def fit(self, X: Any = None, y: Any = None) -> "ImageGraphicalizer":
        """Keep scikit-learn transformer compatibility; there is no fitted state."""

        del X, y
        return self

    def fit_transform(self, X: str | Path | Iterable[str | Path], y: Any = None) -> list[nx.MultiDiGraph]:
        del y
        self.fit()
        return self.transform(X)
