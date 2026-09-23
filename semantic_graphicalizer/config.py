"""Strict YAML configuration loaders for ontologies and prompt packs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .exceptions import ConfigurationError


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must be a mapping")
    return dict(value)


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} must be a non-empty string")
    return value.strip()


def _string_list(value: Any, name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigurationError(f"{name} must be a list of strings")
    return [_string(item, f"{name}[{index}]") for index, item in enumerate(value)]


@dataclass(frozen=True)
class OntologyTerm:
    id: str
    description: str


@dataclass(frozen=True)
class OntologyArgument:
    role: str
    cardinality: str = "0..n"
    allowed_types: tuple[str, ...] = ()

    @property
    def minimum(self) -> int:
        return int(self.cardinality.split("..", 1)[0])

    @property
    def maximum(self) -> int | None:
        upper = self.cardinality.split("..", 1)[1]
        return None if upper == "n" else int(upper)


@dataclass(frozen=True)
class OntologyRelation:
    id: str
    description: str
    arguments: tuple[OntologyArgument, ...] = ()
    projection: tuple[str, str] | None = None
    category: str | None = None

    @property
    def argument_roles(self) -> tuple[str, ...]:
        return tuple(argument.role for argument in self.arguments)


@dataclass(frozen=True)
class ArgumentRole:
    id: str
    description: str = ""


@dataclass(frozen=True)
class OntologyConfig:
    name: str
    version: str
    terms: tuple[OntologyTerm, ...]
    relations: tuple[OntologyRelation, ...]
    argument_roles: tuple[ArgumentRole, ...] = ()

    @property
    def term_ids(self) -> set[str]:
        return {term.id for term in self.terms}

    @property
    def relation_ids(self) -> set[str]:
        return {relation.id for relation in self.relations}

    @property
    def argument_role_ids(self) -> set[str]:
        return {role.id for role in self.argument_roles} | {
            role for relation in self.relations for role in relation.argument_roles
        }

    def relation(self, relation_id: str) -> OntologyRelation | None:
        return next((item for item in self.relations if item.id == relation_id), None)

    def as_prompt(self) -> str:
        terms = "\n".join(f"- {term.id}: {term.description}" for term in self.terms)
        roles = "\n".join(
            f"- {role.id}: {role.description}" for role in self.argument_roles
        ) or "- (roles may be declared inline on relations)"
        relations = []
        for relation in self.relations:
            arguments = ", ".join(
                f"{argument.role} [{argument.cardinality}; "
                f"types: {', '.join(argument.allowed_types) or 'any'}]"
                for argument in relation.arguments
            ) or "any named arguments"
            projection = (
                f"; projection: {relation.projection[0]} -> {relation.projection[1]}"
                if relation.projection else ""
            )
            category = f"; category: {relation.category}" if relation.category else ""
            relations.append(f"- {relation.id}: {relation.description} ({arguments}{projection}{category})")
        return (
            f"Ontology: {self.name} (version {self.version})\n"
            f"Entity types:\n{terms}\nArgument roles:\n{roles}\n"
            f"Relations:\n{'\n'.join(relations)}"
        )


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


def _cardinality(value: Any, name: str) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        value = f"{value}..{value}"
    value = _string(value, name)
    if ".." not in value:
        raise ConfigurationError(f"{name} must use N or N..M/N..n cardinality syntax")
    lower, upper = value.split("..", 1)
    if not lower.isdigit() or (upper != "n" and not upper.isdigit()):
        raise ConfigurationError(f"{name} has invalid cardinality '{value}'")
    if upper != "n" and int(upper) < int(lower):
        raise ConfigurationError(f"{name} upper bound must not be below lower bound")
    return value


def load_ontology(source: str | Path | Mapping[str, Any] | OntologyConfig) -> OntologyConfig:
    if isinstance(source, OntologyConfig):
        return source
    raw = _read_yaml(source)
    required = {"name", "version", "terms", "relations"}
    unexpected = set(raw) - (required | {"argument_roles"})
    missing = required - set(raw)
    if missing or unexpected:
        raise ConfigurationError(
            f"ontology must contain required fields {sorted(required)} and optional argument_roles; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    terms_raw = raw["terms"]
    if not isinstance(terms_raw, list):
        raise ConfigurationError("ontology.terms must be a list")
    terms: list[OntologyTerm] = []
    for index, raw_term in enumerate(terms_raw):
        item = _mapping(raw_term, f"ontology.terms[{index}]")
        if set(item) != {"id", "description"}:
            raise ConfigurationError(f"ontology.terms[{index}] must contain exactly id and description")
        terms.append(OntologyTerm(_string(item["id"], f"ontology.terms[{index}].id"), _string(item["description"], f"ontology.terms[{index}].description")))
    term_ids = {term.id for term in terms}
    if len(term_ids) != len(terms):
        raise ConfigurationError("ontology term IDs must be unique")

    roles: list[ArgumentRole] = []
    raw_roles = raw.get("argument_roles", [])
    if isinstance(raw_roles, Mapping):
        raw_roles = [{"id": role_id, "description": description} for role_id, description in raw_roles.items()]
    if not isinstance(raw_roles, list):
        raise ConfigurationError("ontology.argument_roles must be a list or mapping")
    for index, raw_role in enumerate(raw_roles):
        if isinstance(raw_role, str):
            roles.append(ArgumentRole(_string(raw_role, f"ontology.argument_roles[{index}]")))
            continue
        item = _mapping(raw_role, f"ontology.argument_roles[{index}]")
        if set(item) - {"id", "description"} or "id" not in item:
            raise ConfigurationError(f"ontology.argument_roles[{index}] requires id and optional description")
        roles.append(ArgumentRole(_string(item["id"], f"ontology.argument_roles[{index}].id"), str(item.get("description", ""))))
    if len({role.id for role in roles}) != len(roles):
        raise ConfigurationError("ontology argument role IDs must be unique")

    raw_relations = raw["relations"]
    if not isinstance(raw_relations, Mapping):
        raise ConfigurationError("ontology.relations must be a mapping of relation IDs to definitions")
    relations: list[OntologyRelation] = []
    role_ids = {role.id for role in roles}
    for relation_id, raw_relation in raw_relations.items():
        rid = _string(relation_id, "ontology.relations relation ID")
        item = _mapping(raw_relation, f"ontology.relations.{rid}")
        allowed = {"description", "arguments", "projection", "category"}
        if set(item) - allowed or "description" not in item:
            raise ConfigurationError(f"ontology.relations.{rid} requires description and allows arguments/projection")
        raw_arguments = item.get("arguments", {})
        if not isinstance(raw_arguments, Mapping):
            raise ConfigurationError(f"ontology.relations.{rid}.arguments must be a mapping")
        arguments: list[OntologyArgument] = []
        for role_id, raw_argument in raw_arguments.items():
            role = _string(role_id, f"ontology.relations.{rid}.arguments role")
            argument = _mapping(raw_argument, f"ontology.relations.{rid}.arguments.{role}")
            if set(argument) - {"cardinality", "allowed_types"}:
                raise ConfigurationError(f"ontology.relations.{rid}.arguments.{role} has unsupported fields")
            cardinality = _cardinality(argument.get("cardinality", "0..n"), f"ontology.relations.{rid}.arguments.{role}.cardinality")
            allowed_types = tuple(_string_list(argument.get("allowed_types"), f"ontology.relations.{rid}.arguments.{role}.allowed_types"))
            unknown_types = set(allowed_types) - term_ids
            if unknown_types:
                raise ConfigurationError(f"ontology.relations.{rid}.arguments.{role} references unknown types: {sorted(unknown_types)}")
            arguments.append(OntologyArgument(role, cardinality, allowed_types))
            role_ids.add(role)
        projection = item.get("projection")
        if projection is not None:
            projection_items = _string_list(projection, f"ontology.relations.{rid}.projection")
            if len(projection_items) != 2:
                raise ConfigurationError(f"ontology.relations.{rid}.projection must contain exactly two roles")
            if any(role not in {argument.role for argument in arguments} for role in projection_items):
                raise ConfigurationError(f"ontology.relations.{rid}.projection roles must be declared arguments")
            projection_tuple: tuple[str, str] | None = (projection_items[0], projection_items[1])
        else:
            projection_tuple = None
        category = item.get("category")
        if category is not None:
            category = _string(category, f"ontology.relations.{rid}.category")
            if category not in {"temporal", "causal"}:
                raise ConfigurationError(
                    f"ontology.relations.{rid}.category must be 'temporal' or 'causal'"
                )
        relations.append(OntologyRelation(rid, _string(item["description"], f"ontology.relations.{rid}.description"), tuple(arguments), projection_tuple, category))
    if len({relation.id for relation in relations}) != len(relations):
        raise ConfigurationError("ontology relation IDs must be unique")
    return OntologyConfig(_string(raw["name"], "ontology.name"), _string(raw["version"], "ontology.version"), tuple(terms), tuple(relations), tuple(roles))


def load_prompts(source: str | Path | Mapping[str, Any] | PromptConfig) -> PromptConfig:
    if isinstance(source, PromptConfig):
        return source
    raw = _read_yaml(source)
    expected = {"domain", "version", "stages"}
    if set(raw) != expected:
        raise ConfigurationError(f"prompt pack must contain exactly {sorted(expected)}")
    raw_stages = _mapping(raw["stages"], "prompts.stages")
    required_stages = {"summarize", "normalize", "decompose", "extract", "resolve"}
    if set(raw_stages) != required_stages:
        raise ConfigurationError(f"prompts.stages must contain exactly {sorted(required_stages)}")
    stages: dict[str, PromptStage] = {}
    for stage, raw_stage in raw_stages.items():
        item = _mapping(raw_stage, f"prompts.stages.{stage}")
        if set(item) != {"system", "instruction", "output"}:
            raise ConfigurationError(f"prompts.stages.{stage} must contain exactly system, instruction, output")
        stages[stage] = PromptStage(_string(item["system"], f"prompts.stages.{stage}.system"), _string(item["instruction"], f"prompts.stages.{stage}.instruction"), _string(item["output"], f"prompts.stages.{stage}.output"))
    return PromptConfig(_string(raw["domain"], "prompts.domain"), _string(raw["version"], "prompts.version"), stages)
