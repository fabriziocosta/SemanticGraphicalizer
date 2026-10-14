"""Image-domain prompt configuration and request schemas."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .config import OntologyConfig, PromptStage


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


def _inventory_schema(max_regions: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "scene_description": {"type": "string"},
            "regions": {
                "type": "array",
                "maxItems": max_regions,
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


