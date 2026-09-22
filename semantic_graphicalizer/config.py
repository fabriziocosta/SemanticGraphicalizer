"""Strict YAML configuration loaders for ontologies and prompt packs."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .exceptions import ConfigurationError


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{name} must be a mapping")
    return value


def _required(mapping: dict[str, Any], key: str, name: str) -> Any:
    if key not in mapping:
        raise ConfigurationError(f"{name} is missing required field '{key}'")
    return mapping[key]


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} must be a non-empty string")
    return value.strip()


def _string_list(value: Any, name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigurationError(f"{name} must be a list of strings")
    return [item.strip() for item in value]


@dataclass(frozen=True)
class OntologyTerm:
    id: str
    description: str


@dataclass(frozen=True)
class OntologyRelation:
    id: str
    description: str
    source_terms: tuple[str, ...] = ()
    target_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class OntologyLinkRelation:
    id: str
    category: str
    description: str


@dataclass(frozen=True)
class OntologyConfig:
    name: str
    version: str
    terms: tuple[OntologyTerm, ...]
    relations: tuple[OntologyRelation, ...]
    link_relations: tuple[OntologyLinkRelation, ...] = ()

    @property
    def term_ids(self) -> set[str]:
        return {term.id for term in self.terms}

    @property
    def relation_ids(self) -> set[str]:
        return {relation.id for relation in self.relations}

    @property
    def link_relation_ids(self) -> set[str]:
        return {relation.id for relation in self.link_relations}

    def as_prompt(self) -> str:
        terms = "\n".join(f"- {term.id}: {term.description}" for term in self.terms)
        relations = "\n".join(
            f"- {relation.id}: {relation.description}"
            f" (source: {', '.join(relation.source_terms) or 'any'};"
            f" target: {', '.join(relation.target_terms) or 'any'})"
            for relation in self.relations
        )
        link_relations = "\n".join(
            f"- {relation.id}: {relation.description} (category: {relation.category})"
            for relation in self.link_relations
        )
        link_section = f"\nProposition links:\n{link_relations}" if link_relations else ""
        return f"Ontology: {self.name} (version {self.version})\nTerms:\n{terms}\nRelations:\n{relations}{link_section}"


@dataclass(frozen=True)
class PromptStage:
    system: str
    instruction: str
    output: str


@dataclass(frozen=True)
class PromptConfig:
    domain: str
    version: str
    stages: dict[str, PromptStage]

    def render(self, stage: str, **values: str) -> str:
        if stage not in self.stages:
            raise ConfigurationError(f"prompt pack has no stage '{stage}'")
        template = self.stages[stage]
        try:
            system = template.system.format(**values)
            instruction = template.instruction.format(**values)
        except KeyError as exc:
            raise ConfigurationError(
                f"prompt stage '{stage}' references missing placeholder {exc}"
            ) from exc
        return f"SYSTEM:\n{system}\n\nINSTRUCTION:\n{instruction}\n\nOUTPUT:\n{template.output}"


def _read_yaml(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source)
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
    except OSError as exc:
        raise ConfigurationError(f"could not read YAML file '{path}': {exc}") from exc
    return _mapping(value, str(path))


def load_ontology(source: str | Path | Mapping[str, Any] | OntologyConfig) -> OntologyConfig:
    if isinstance(source, OntologyConfig):
        return source
    raw = _read_yaml(source)
    required = {"name", "version", "terms", "relations"}
    expected = required | {"link_relations"}
    unexpected = set(raw) - expected
    missing = required - set(raw)
    if missing or unexpected:
        raise ConfigurationError(
            f"ontology must contain required fields {sorted(required)} and optional link_relations; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    name = _string(raw["name"], "ontology.name")
    version = _string(raw["version"], "ontology.version")
    raw_terms = raw["terms"]
    raw_relations = raw["relations"]
    if not isinstance(raw_terms, list) or not isinstance(raw_relations, list):
        raise ConfigurationError("ontology.terms and ontology.relations must be lists")

    terms: list[OntologyTerm] = []
    for index, raw_term in enumerate(raw_terms):
        item = _mapping(raw_term, f"ontology.terms[{index}]")
        if set(item) != {"id", "description"}:
            raise ConfigurationError(
                f"ontology.terms[{index}] must contain exactly id and description"
            )
        terms.append(OntologyTerm(_string(item["id"], f"ontology.terms[{index}].id"),
                                  _string(item["description"], f"ontology.terms[{index}].description")))
    term_ids = [term.id for term in terms]
    if len(term_ids) != len(set(term_ids)):
        raise ConfigurationError("ontology term IDs must be unique")

    relations: list[OntologyRelation] = []
    for index, raw_relation in enumerate(raw_relations):
        item = _mapping(raw_relation, f"ontology.relations[{index}]")
        allowed = {"id", "description", "source_terms", "target_terms"}
        if set(item) - allowed or not {"id", "description"}.issubset(item):
            raise ConfigurationError(
                f"ontology.relations[{index}] requires id and description and only allows "
                "source_terms and target_terms as optional fields"
            )
        source_terms = _string_list(item.get("source_terms"), f"ontology.relations[{index}].source_terms")
        target_terms = _string_list(item.get("target_terms"), f"ontology.relations[{index}].target_terms")
        unknown = set(source_terms + target_terms) - set(term_ids)
        if unknown:
            raise ConfigurationError(
                f"ontology.relations[{index}] references unknown terms: {sorted(unknown)}"
            )
        relations.append(OntologyRelation(
            _string(item["id"], f"ontology.relations[{index}].id"),
            _string(item["description"], f"ontology.relations[{index}].description"),
            tuple(source_terms), tuple(target_terms),
        ))
    relation_ids = [relation.id for relation in relations]
    if len(relation_ids) != len(set(relation_ids)):
        raise ConfigurationError("ontology relation IDs must be unique")

    raw_link_relations = raw.get("link_relations", [])
    if not isinstance(raw_link_relations, list):
        raise ConfigurationError("ontology.link_relations must be a list")
    link_relations: list[OntologyLinkRelation] = []
    for index, raw_link_relation in enumerate(raw_link_relations):
        item = _mapping(raw_link_relation, f"ontology.link_relations[{index}]")
        if set(item) != {"id", "category", "description"}:
            raise ConfigurationError(
                f"ontology.link_relations[{index}] must contain exactly id, category, and description"
            )
        category = _string(item["category"], f"ontology.link_relations[{index}].category")
        if category not in {"temporal", "causal"}:
            raise ConfigurationError(
                f"ontology.link_relations[{index}].category must be 'temporal' or 'causal'"
            )
        link_relations.append(OntologyLinkRelation(
            _string(item["id"], f"ontology.link_relations[{index}].id"),
            category,
            _string(item["description"], f"ontology.link_relations[{index}].description"),
        ))
    link_relation_ids = [relation.id for relation in link_relations]
    if len(link_relation_ids) != len(set(link_relation_ids)):
        raise ConfigurationError("ontology link relation IDs must be unique")
    return OntologyConfig(name, version, tuple(terms), tuple(relations), tuple(link_relations))


def load_prompts(source: str | Path | Mapping[str, Any] | PromptConfig) -> PromptConfig:
    if isinstance(source, PromptConfig):
        return source
    raw = _read_yaml(source)
    expected = {"domain", "version", "stages"}
    unexpected = set(raw) - expected
    missing = expected - set(raw)
    if missing or unexpected:
        raise ConfigurationError(
            f"prompt pack must contain exactly {sorted(expected)}; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    domain = _string(raw["domain"], "prompts.domain")
    version = _string(raw["version"], "prompts.version")
    raw_stages = _mapping(raw["stages"], "prompts.stages")
    required_stages = {"summarize", "normalize", "decompose", "triple"}
    allowed_stages = required_stages | {"link"}
    if not set(raw_stages).issubset(allowed_stages) or not required_stages.issubset(raw_stages):
        raise ConfigurationError(
            f"prompts.stages must contain {sorted(required_stages)} and may optionally include 'link'"
        )
    stages: dict[str, PromptStage] = {}
    for stage, raw_stage in raw_stages.items():
        item = _mapping(raw_stage, f"prompts.stages.{stage}")
        if set(item) != {"system", "instruction", "output"}:
            raise ConfigurationError(
                f"prompts.stages.{stage} must contain exactly system, instruction, output"
            )
        stages[stage] = PromptStage(
            _string(item["system"], f"prompts.stages.{stage}.system"),
            _string(item["instruction"], f"prompts.stages.{stage}.instruction"),
            _string(item["output"], f"prompts.stages.{stage}.output"),
        )
    return PromptConfig(domain, version, stages)
