"""Ontology-guided graphicalization for local scene images."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
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
from .embedding_utils import project_embedding_vectors
from .graph import GraphValidationError, materialize_graph
from .image_embeddings import (
    DEFAULT_IMAGE_EMBEDDING_MODEL,
    IMAGE_EMBEDDING_DIMENSIONS,
    SUPPORTED_IMAGE_EMBEDDING_DIMENSIONS,
    EmbeddingGemma2ImageClient,
    ImageEmbeddingClient,
    as_image_embedding_client,
)
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
            "bbox": {
                "type": ["array", "null"],
                "items": {"type": "number", "minimum": 0, "maximum": 1},
                "minItems": 4,
                "maxItems": 4,
            },
            "description": {"type": "string"},
            "properties": {"type": "array", "items": properties},
            "confidence": _confidence_schema(),
        },
        "required": ["id", "type", "mention", "key", "bbox", "description", "properties", "confidence"],
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


def _bbox_overlap(first: list[float], second: list[float]) -> tuple[float, float, float]:
    """Return IoU, intersection over smaller box, and center distance."""

    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    smaller_area = min(first_area, second_area)
    iou = intersection / union if union else 0.0
    smaller_coverage = intersection / smaller_area if smaller_area else 0.0
    first_center = ((first[0] + first[2]) / 2, (first[1] + first[3]) / 2)
    second_center = ((second[0] + second[2]) / 2, (second[1] + second[3]) / 2)
    center_distance = math.dist(first_center, second_center)
    return iou, smaller_coverage, center_distance


def _mention_similarity(first: Any, second: Any) -> float:
    """Measure token overlap while tolerating extra visual adjectives."""

    def tokens(value: Any) -> set[str]:
        result = set()
        for token in re.findall(r"[a-z0-9]+", str(value).casefold()):
            # Singularize ordinary plurals so "garden shrubs" and
            # "flowering garden shrub" compare as the same visual mention.
            if len(token) > 4 and token.endswith("ies"):
                token = token[:-3] + "y"
            elif len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
                token = token[:-1]
            result.add(token)
        return result

    first_tokens = tokens(first)
    second_tokens = tokens(second)
    if not first_tokens or not second_tokens:
        return 0.0
    return len(first_tokens & second_tokens) / min(len(first_tokens), len(second_tokens))


class SemanticImageGraphicalizer:
    """Extract ontology-grounded graphs from local scene images."""

    def __init__(
        self,
        ontology: str | Path | Mapping[str, Any] | OntologyConfig = DEFAULT_IMAGE_ONTOLOGY,
        prompts: str | Path | Mapping[str, Any] | ImagePromptConfig = DEFAULT_IMAGE_PROMPTS,
        model: Any = None,
        *,
        vlm_model: str = DEFAULT_OPENAI_IMAGE_MODEL,
        crop_dir: str | Path = Path("domains/image/data/processed/regions"),
        image_embedder: ImageEmbeddingClient | Callable[[Sequence[str | Path], int], Sequence[Sequence[float]]] | None = None,
        image_embedding_model: str = DEFAULT_IMAGE_EMBEDDING_MODEL,
        image_embedding_dim: int = IMAGE_EMBEDDING_DIMENSIONS,
        image_embedding_source_dim: int | None = None,
        image_embedding_projection_seed: int = 17,
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
        if not isinstance(vlm_model, str) or not vlm_model.strip():
            raise ValueError("vlm_model must be a non-empty string")
        if not isinstance(image_embedding_model, str) or not image_embedding_model.strip():
            raise ValueError("image_embedding_model must be a non-empty string")
        if (
            isinstance(image_embedding_dim, bool)
            or not isinstance(image_embedding_dim, int)
            or image_embedding_dim < 1
        ):
            raise ValueError("image_embedding_dim must be a positive integer")
        if image_embedding_source_dim is not None and (
            isinstance(image_embedding_source_dim, bool)
            or not isinstance(image_embedding_source_dim, int)
            or image_embedding_source_dim not in SUPPORTED_IMAGE_EMBEDDING_DIMENSIONS
        ):
            raise ValueError(
                f"image_embedding_source_dim must be None or one of {SUPPORTED_IMAGE_EMBEDDING_DIMENSIONS}"
            )
        configured_source_dim = image_embedding_source_dim or (
            image_embedding_dim
            if image_embedding_dim in SUPPORTED_IMAGE_EMBEDDING_DIMENSIONS
            else IMAGE_EMBEDDING_DIMENSIONS
        )
        if image_embedding_dim > configured_source_dim:
            raise ValueError("image_embedding_dim cannot exceed image_embedding_source_dim")
        if (
            isinstance(image_embedding_projection_seed, bool)
            or not isinstance(image_embedding_projection_seed, int)
            or image_embedding_projection_seed < 0
        ):
            raise ValueError("image_embedding_projection_seed must be a non-negative integer")
        if scene_type not in self.ontology.term_ids:
            raise ValueError(f"scene_type '{scene_type}' is not defined by the ontology")
        self.scene_type = scene_type
        self.crop_dir = Path(crop_dir)
        self.vlm_model = vlm_model
        self.image_embedder = image_embedder
        self.image_embedding_model = image_embedding_model
        self.image_embedding_dim = image_embedding_dim
        self.image_embedding_source_dim = image_embedding_source_dim
        self.image_embedding_projection_seed = image_embedding_projection_seed
        self.max_retries = max_retries
        self.retry_backoff = float(retry_backoff)
        self.verbose = verbose
        self.model: RequestModelClient | None = as_model_client(model) if model is not None else None

    @staticmethod
    def _image_backend():
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise ImportError("SemanticImageGraphicalizer requires Pillow; install semantic-graphicalizer[image]") from exc
        return Image

    def _run_request(self, request: ModelRequest) -> Mapping[str, Any]:
        if self.model is None:
            self.model = as_model_client(OpenAIModelClient(model=self.vlm_model))
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

    def _analyze_part(self, part: Part, scene_summary: str) -> tuple[list[Entity], list[str]]:
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
        warnings: list[str] = []
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
            local_bbox: list[float] | None = None
            entity_bbox: list[float] | None = None
            if item.get("bbox") is not None:
                pixel_bbox = part.source_locator.get("pixel_bbox")
                if not isinstance(pixel_bbox, (list, tuple)) or len(pixel_bbox) != 4:
                    raise ValueError("image part source_locator must contain pixel_bbox")
                crop_width = int(pixel_bbox[2]) - int(pixel_bbox[0])
                crop_height = int(pixel_bbox[3]) - int(pixel_bbox[1])
                try:
                    local_bbox, _ = _bbox_pixels(item["bbox"], crop_width, crop_height)
                except ValueError as exc:
                    warnings.append(
                        f"entity {local_id!r} in part {part.part_id!r} retained without "
                        f"entity-level bbox: {exc}"
                    )
                else:
                    crop_bbox = part.source_locator.get("normalized_bbox")
                    if not isinstance(crop_bbox, (list, tuple)) or len(crop_bbox) != 4:
                        raise ValueError("image part source_locator must contain normalized_bbox")
                    crop_x0, crop_y0, crop_x1, crop_y1 = (float(value) for value in crop_bbox)
                    local_x0, local_y0, local_x1, local_y1 = local_bbox
                    entity_bbox = [
                        crop_x0 + local_x0 * (crop_x1 - crop_x0),
                        crop_y0 + local_y0 * (crop_y1 - crop_y0),
                        crop_x0 + local_x1 * (crop_x1 - crop_x0),
                        crop_y0 + local_y1 * (crop_y1 - crop_y0),
                    ]
            properties = item.get("properties", [])
            if not isinstance(properties, list):
                raise ValueError("image entity properties must be a list")
            attributes: dict[str, Any] = {
                "mentions": [mention],
                "description": str(item.get("description") or ""),
                "properties": {},
                "confidence": item.get("confidence"),
                "provenance": [{
                    "part_id": part.part_id,
                    "source_locator": {
                        **dict(part.source_locator),
                        **({"entity_bbox": entity_bbox, "entity_bbox_crop": local_bbox} if entity_bbox else {}),
                    },
                    "crop_path": str(part.content),
                    "mention": mention,
                    "description": str(item.get("description") or ""),
                    "key": _required_text(key, "analyze_entity", f"entities[{index}].key") if key is not None else None,
                    "properties": {str(prop.get("name")): str(prop.get("value")) for prop in properties if isinstance(prop, Mapping)},
                    "confidence": item.get("confidence"),
                }],
            }
            for prop_index, raw_property in enumerate(properties):
                prop = _object(raw_property, "analyze_entity", f"entities[{index}].properties[{prop_index}]")
                name = _required_text(prop.get("name"), "analyze_entity", "property name")
                value = _required_text(prop.get("value"), "analyze_entity", f"property {name}")
                attributes["properties"][name] = value
            if key is not None:
                attributes["key"] = _required_text(key, "analyze_entity", f"entities[{index}].key")
            entities.append(Entity(entity_id, entity_type, None, attributes))
        return entities, warnings

    @staticmethod
    def _same_visual_entity(first: Entity, second: Entity) -> bool:
        """Match the same typed entity only when crop geometry also agrees."""

        if first.type != second.type:
            return False
        first_attributes = first.attributes
        second_attributes = second.attributes
        first_provenance = first_attributes.get("provenance", [])
        second_provenance = second_attributes.get("provenance", [])
        if not isinstance(first_provenance, list) or not isinstance(second_provenance, list):
            return False
        first_parts = {record.get("part_id") for record in first_provenance if isinstance(record, Mapping)}
        second_parts = {record.get("part_id") for record in second_provenance if isinstance(record, Mapping)}
        first_parts.discard(None)
        second_parts.discard(None)
        if not first_parts or not second_parts:
            return False
        for first_record in first_provenance:
            if not isinstance(first_record, Mapping):
                continue
            first_locator = first_record.get("source_locator", {})
            if not isinstance(first_locator, Mapping):
                continue
            first_box = first_locator.get("entity_bbox", first_locator.get("normalized_bbox"))
            for second_record in second_provenance:
                if not isinstance(second_record, Mapping):
                    continue
                second_locator = second_record.get("source_locator", {})
                if not isinstance(second_locator, Mapping):
                    continue
                first_precise = first_locator.get("entity_bbox") is not None
                second_precise = second_locator.get("entity_bbox") is not None
                same_part = first_record.get("part_id") == second_record.get("part_id")
                second_box = second_locator.get("entity_bbox", second_locator.get("normalized_bbox"))
                if not (
                    isinstance(first_box, (list, tuple)) and len(first_box) == 4
                    and isinstance(second_box, (list, tuple)) and len(second_box) == 4
                ):
                    continue
                try:
                    iou, smaller_coverage, center_distance = _bbox_overlap(
                        [float(value) for value in first_box],
                        [float(value) for value in second_box],
                    )
                except (TypeError, ValueError):
                    continue
                first_mentions = first_attributes.get("mentions", [])
                second_mentions = second_attributes.get("mentions", [])
                similarity = max(
                    (
                        _mention_similarity(first_mention, second_mention)
                        for first_mention in first_mentions
                        for second_mention in second_mentions
                    ),
                    default=0.0,
                )
                first_key = str(first_record.get("key") or first_attributes.get("key") or "").casefold().strip()
                second_key = str(second_record.get("key") or second_attributes.get("key") or "").casefold().strip()
                if first_precise and second_precise:
                    same_location = iou >= 0.3 or (smaller_coverage >= 0.8 and center_distance <= 0.1)
                    strong_spatial_match = iou >= 0.7 or (
                        smaller_coverage >= 0.95 and center_distance <= 0.05
                    )
                else:
                    # Crop boxes are only a fallback. Require a close match to
                    # avoid merging similar objects from wide regions.
                    same_location = iou >= 0.55
                    strong_spatial_match = False
                same_identity_hint = (
                    bool(first_key and first_key == second_key)
                    or similarity >= 0.5
                    or strong_spatial_match
                )
                # Multiple entities can be returned for one broad crop. If
                # precise boxes are missing, merge only near-identical names
                # from that same crop; this avoids treating "garden shrubs"
                # and "flowering garden shrub" as two nodes while preserving
                # distinct entities such as a tree and a shrub.
                if same_part and not (first_precise and second_precise):
                    if not (first_key and first_key == second_key) and similarity < 0.8:
                        continue
                if same_identity_hint and same_location:
                    return True
        return False

    @classmethod
    def _deduplicate_entities(cls, entities: list[Entity]) -> list[Entity]:
        """Merge repeated detections across crops and retain all provenance."""

        merged: list[Entity] = []
        for entity in entities:
            match_index = next(
                (index for index, current in enumerate(merged) if cls._same_visual_entity(current, entity)),
                None,
            )
            if match_index is None:
                merged.append(entity)
                continue

            current = merged[match_index]
            attributes = {**current.attributes}
            mentions = list(attributes.get("mentions", []))
            for mention in entity.attributes.get("mentions", []):
                if not any(str(mention).casefold() == str(existing).casefold() for existing in mentions):
                    mentions.append(mention)
            attributes["mentions"] = mentions

            provenance = list(attributes.get("provenance", []))
            provenance.extend(entity.attributes.get("provenance", []))
            attributes["provenance"] = provenance

            descriptions = [
                value for value in (
                    attributes.get("description"),
                    entity.attributes.get("description"),
                ) if value
            ]
            attributes["descriptions"] = list(dict.fromkeys(descriptions))
            if descriptions:
                # Prefer the more informative merged label for graph display.
                attributes["description"] = max(
                    descriptions,
                    key=lambda value: len(set(re.findall(r"[a-z0-9]+", str(value).casefold()))),
                )
            properties = dict(attributes.get("properties", {}))
            for name, value in entity.attributes.get("properties", {}).items():
                properties.setdefault(name, value)
            attributes["properties"] = properties
            confidences = [
                value for value in (attributes.get("confidence"), entity.attributes.get("confidence"))
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            ]
            if confidences:
                attributes["confidence"] = max(confidences)
            if not attributes.get("key") and entity.attributes.get("key"):
                attributes["key"] = entity.attributes["key"]
            merged[match_index] = Entity(current.id, current.type, current.relation, attributes)
        return merged

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
            extracted, analysis_warnings = self._analyze_part(part, scene_summary)
            entities.extend(extracted)
            warnings.extend(analysis_warnings)
            stats.append(StageStat(document_id, "analyze_entity", time.perf_counter() - started, 1, len(extracted), part.part_id))
        started = time.perf_counter()
        extracted_entity_count = len(entities)
        entities = self._deduplicate_entities(entities)
        duplicate_entity_count = extracted_entity_count - len(entities)
        stats.append(StageStat(
            document_id,
            "deduplicate_entities",
            time.perf_counter() - started,
            extracted_entity_count,
            len(entities),
        ))
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
            duplicate_entities_merged=duplicate_entity_count,
            parts=[{**asdict(part), "content": str(part.content)} for part in parts],
            warnings=warnings,
            stats=[asdict(stat) for stat in stats],
        )
        if self.verbose:
            print(
                f"[SemanticImageGraphicalizer] {path.name}: {len(parts)} regions, "
                f"merged {duplicate_entity_count} duplicate detections, "
                f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
            )
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

    def compute_part_embeddings(
        self,
        values: nx.MultiDiGraph | Iterable[nx.MultiDiGraph],
        *,
        batch_size: int = 32,
        embedding_dim: int | None = None,
        source_embedding_dim: int | None = None,
        skip_matching: bool = True,
    ) -> list[nx.MultiDiGraph]:
        """Embed saved image parts and attach vectors to graph part records.

        Graphs are updated in place and returned in input order. The configured
        image embedder and output dimension from ``__init__`` are used unless
        overridden for this call. Output dimensions supported natively by
        EmbeddingGemma 2 use its Matryoshka truncation. Other smaller dimensions
        are produced with a seeded Gaussian random projection from
        ``source_embedding_dim`` (or 768 by default). The default client and
        model weights load only if a part needs an embedding. Entity provenance
        references vectors through each part's existing ``part_id``.
        """

        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(skip_matching, bool):
            raise TypeError("skip_matching must be a bool")
        if embedding_dim is None:
            embedding_dim = self.image_embedding_dim
        if (
            isinstance(embedding_dim, bool)
            or not isinstance(embedding_dim, int)
            or embedding_dim < 1
        ):
            raise ValueError("embedding_dim must be a positive integer")
        if source_embedding_dim is None:
            source_embedding_dim = self.image_embedding_source_dim
        if source_embedding_dim is None:
            source_embedding_dim = (
                embedding_dim
                if embedding_dim in SUPPORTED_IMAGE_EMBEDDING_DIMENSIONS
                else IMAGE_EMBEDDING_DIMENSIONS
            )
        if (
            isinstance(source_embedding_dim, bool)
            or not isinstance(source_embedding_dim, int)
            or source_embedding_dim not in SUPPORTED_IMAGE_EMBEDDING_DIMENSIONS
        ):
            raise ValueError(
                f"source_embedding_dim must be one of {SUPPORTED_IMAGE_EMBEDDING_DIMENSIONS}"
            )
        if embedding_dim > source_embedding_dim:
            raise ValueError("embedding_dim cannot exceed source_embedding_dim")
        if isinstance(values, nx.MultiDiGraph):
            graphs = [values]
        elif isinstance(values, (str, bytes)):
            raise TypeError("values must be image graphs or an iterable of image graphs")
        else:
            try:
                graphs = list(values)
            except TypeError as exc:
                raise TypeError("values must be image graphs or an iterable of image graphs") from exc
        if not all(isinstance(graph, nx.MultiDiGraph) for graph in graphs):
            raise TypeError("values must contain only NetworkX MultiDiGraph objects")

        part_records: list[tuple[nx.MultiDiGraph, dict[str, Any], str, str]] = []
        for graph in graphs:
            if graph.graph.get("modality") != "image":
                raise ValueError("compute_part_embeddings accepts only image graphs")
            parts = graph.graph.get("parts", [])
            if not isinstance(parts, list):
                raise ValueError("image graph metadata 'parts' must be a list")
            for index, raw_part in enumerate(parts):
                if not isinstance(raw_part, Mapping):
                    raise ValueError(f"image graph part {index} must be an object")
                part = raw_part if isinstance(raw_part, dict) else dict(raw_part)
                if part is not raw_part:
                    parts[index] = part
                part_id = part.get("part_id")
                if not isinstance(part_id, str) or not part_id:
                    raise ValueError(f"image graph part {index} is missing a valid part_id")
                raw_path = part.get("content")
                if raw_path is None:
                    metadata = part.get("metadata", {})
                    raw_path = metadata.get("crop_path") if isinstance(metadata, Mapping) else None
                if not isinstance(raw_path, (str, Path)) or not str(raw_path):
                    raise ValueError(f"image graph part {part_id!r} has no saved crop path")
                crop_path = Path(raw_path)
                if not crop_path.is_file():
                    raise ValueError(f"image crop for part {part_id!r} does not exist: {crop_path}")
                digest = sha256()
                with crop_path.open("rb") as crop_file:
                    for chunk in iter(lambda: crop_file.read(1024 * 1024), b""):
                        digest.update(chunk)
                part_records.append((graph, part, str(crop_path), digest.hexdigest()))

        if not part_records:
            return graphs

        raw_client = self.image_embedder
        if raw_client is None:
            raw_client = EmbeddingGemma2ImageClient(self.image_embedding_model)
            self.image_embedder = raw_client
        client = as_image_embedding_client(raw_client)
        configured_model_id = getattr(client, "model_id", None)
        model_id = (
            configured_model_id
            if isinstance(configured_model_id, str) and configured_model_id.strip()
            else f"{type(client).__module__}.{type(client).__qualname__}"
        )
        config = {
            "model": model_id,
            "dimension": embedding_dim,
            "source_dimension": source_embedding_dim,
            "normalized": True,
            "precision_policy": getattr(client, "precision_policy", "client-managed"),
            "input": "image-part-v1",
            "projection_method": (
                "GaussianRandomProjection"
                if embedding_dim != source_embedding_dim
                else None
            ),
            "projection_random_seed": (
                self.image_embedding_projection_seed
                if embedding_dim != source_embedding_dim
                else None
            ),
        }
        config_hash = sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()

        pending: list[tuple[nx.MultiDiGraph, dict[str, Any], str, str]] = []
        for graph, part, crop_path, crop_sha256 in part_records:
            if skip_matching:
                vector = part.get("embedding")
                metadata = part.get("embedding_metadata")
                if (
                    isinstance(vector, list)
                    and len(vector) == embedding_dim
                    and isinstance(metadata, Mapping)
                    and metadata.get("crop_sha256") == crop_sha256
                    and metadata.get("config_hash") == config_hash
                    and all(isinstance(value, (int, float)) and math.isfinite(value) for value in vector)
                ):
                    continue
            pending.append((graph, part, crop_path, crop_sha256))

        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            vectors = client.embed_images(
                [crop_path for _graph, _part, crop_path, _digest in batch],
                batch_size=batch_size,
                embedding_dim=source_embedding_dim,
            )
            if hasattr(vectors, "tolist"):
                vectors = vectors.tolist()
            if not isinstance(vectors, (list, tuple)) or len(vectors) != len(batch):
                raise ValueError("image_embedder returned a different number of vectors than image parts")
            validated: list[list[float]] = []
            for index, raw_vector in enumerate(vectors):
                if hasattr(raw_vector, "tolist"):
                    raw_vector = raw_vector.tolist()
                if not isinstance(raw_vector, (list, tuple)):
                    raise ValueError(f"image_embedder returned an invalid vector for part '{batch[index][1]['part_id']}'")
                try:
                    vector = [float(value) for value in raw_vector]
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"image_embedder returned an invalid vector for part '{batch[index][1]['part_id']}'"
                    ) from exc
                if len(vector) != source_embedding_dim or not all(math.isfinite(value) for value in vector):
                    raise ValueError(
                        f"image_embedder vectors must contain {source_embedding_dim} finite values; "
                        f"part '{batch[index][1]['part_id']}' returned {len(vector)}"
                    )
                norm = math.hypot(*vector)
                if norm == 0:
                    raise ValueError(f"image_embedder returned a zero vector for part '{batch[index][1]['part_id']}'")
                validated.append([value / norm for value in vector])

            if embedding_dim != source_embedding_dim:
                validated = project_embedding_vectors(
                    validated,
                    output_dim=embedding_dim,
                    random_seed=self.image_embedding_projection_seed,
                )

            for (graph, part, _crop_path, crop_sha256), vector in zip(batch, validated):
                part["embedding"] = vector
                part["embedding_metadata"] = {
                    **config,
                    "config_hash": config_hash,
                    "crop_sha256": crop_sha256,
                }
        return graphs

    def display_on_image(self, graph: nx.MultiDiGraph, image: str | Path | Any | None = None, **kwargs: Any) -> Any:
        """Render graph nodes at entity box centers over the source image."""

        from .visualization import display_image_graph

        return display_image_graph(graph, image=image, **kwargs)

    def fit(self, X: Any = None, y: Any = None) -> "SemanticImageGraphicalizer":
        """Keep scikit-learn transformer compatibility; there is no fitted state."""

        del X, y
        return self

    def fit_transform(self, X: str | Path | Iterable[str | Path], y: Any = None) -> list[nx.MultiDiGraph]:
        del y
        self.fit()
        return self.transform(X)

# Backwards-compatible alias; use SemanticImageGraphicalizer in new code.
ImageGraphicalizer = SemanticImageGraphicalizer
