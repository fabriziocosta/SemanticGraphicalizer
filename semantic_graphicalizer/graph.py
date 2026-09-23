"""Canonical graph integration, validation, projection, and serialization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any

import networkx as nx

from .config import OntologyConfig
from .types import Entity, RelationInstance


class GraphValidationError(ValueError):
    """Raised when a reified graph violates its ontology contract."""


def _relation_nodes(graph: nx.MultiDiGraph) -> list[tuple[Any, Mapping[str, Any]]]:
    return [
        (node_id, data)
        for node_id, data in graph.nodes(data=True)
        if data.get("relation") is not None
    ]


def validate_graph(graph: nx.MultiDiGraph, ontology: OntologyConfig) -> None:
    """Validate canonical Entity nodes and ontology-constrained arguments."""

    if not isinstance(graph, nx.MultiDiGraph):
        raise GraphValidationError("canonical graph must be a NetworkX MultiDiGraph")
    for node_id, data in graph.nodes(data=True):
        for field in ("id", "type", "relation", "attributes"):
            if field not in data:
                raise GraphValidationError(f"node '{node_id}' is missing '{field}'")
        if not isinstance(node_id, str) or not isinstance(data["id"], str) or not isinstance(data["type"], str):
            raise GraphValidationError(f"node '{node_id}' must have string id and type")
        if data["relation"] is not None and not isinstance(data["relation"], str):
            raise GraphValidationError(f"node '{node_id}' relation must be a string or None")
        if not isinstance(data["attributes"], Mapping):
            raise GraphValidationError(f"node '{node_id}' attributes must be a mapping")
        if data["id"] != node_id:
            raise GraphValidationError(f"node '{node_id}' has mismatched Entity id '{data['id']}'")
        if data["type"] not in ontology.term_ids:
            raise GraphValidationError(f"node '{node_id}' has unknown Entity type '{data['type']}'")
        relation_id = data["relation"]
        if relation_id is None:
            if graph.out_edges(node_id, data=True, keys=True):
                raise GraphValidationError(f"atomic Entity '{node_id}' cannot have argument edges")
            continue
        relation = ontology.relation(relation_id)
        if relation is None:
            raise GraphValidationError(f"node '{node_id}' has unknown relation '{relation_id}'")
        edges = list(graph.out_edges(node_id, data=True, keys=True))
        if not edges:
            raise GraphValidationError(f"relational Entity '{node_id}' must have argument edges")
        counts: dict[str, int] = {}
        for _source, target, _key, edge in edges:
            if edge.get("edge_type") != "argument":
                raise GraphValidationError(f"edge from relation '{node_id}' is not an argument edge")
            role = edge.get("role")
            if not isinstance(role, str) or role not in ontology.argument_role_ids:
                raise GraphValidationError(f"unknown argument role '{role}' on relation '{node_id}'")
            if not isinstance(edge.get("attributes", {}), Mapping):
                raise GraphValidationError(f"argument edge from '{node_id}' attributes must be a mapping")
            if target not in graph:
                raise GraphValidationError(f"argument edge from '{node_id}' references missing Entity '{target}'")
            counts[role] = counts.get(role, 0) + 1
            definition = next((item for item in relation.arguments if item.role == role), None)
            if relation.arguments and definition is None:
                raise GraphValidationError(f"invalid argument role '{role}' for relation '{relation_id}'")
            target_type = graph.nodes[target]["type"]
            if definition is not None and definition.allowed_types and target_type not in definition.allowed_types:
                raise GraphValidationError(
                    f"argument '{role}' of relation '{relation_id}' cannot target type '{target_type}'"
                )
        for definition in relation.arguments:
            count = counts.get(definition.role, 0)
            if count < definition.minimum:
                raise GraphValidationError(
                    f"relation '{relation_id}' is missing required argument role '{definition.role}'"
                )
            if definition.maximum is not None and count > definition.maximum:
                raise GraphValidationError(
                    f"relation '{relation_id}' exceeds cardinality for role '{definition.role}'"
                )


def materialize_graph(
    entities: Sequence[Entity],
    relations: Sequence[RelationInstance],
    ontology: OntologyConfig,
    *,
    document_id: str,
    document_text: str,
) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(document_id=document_id, document_text=document_text, unified=True)
    all_entities: dict[str, Entity] = {entity.id: entity for entity in entities}
    for relation in relations:
        if relation.id in all_entities:
            raise GraphValidationError(f"duplicate Entity id '{relation.id}'")
        all_entities[relation.id] = Entity(
            id=relation.id,
            type=relation.type,
            relation=relation.relation,
            attributes=dict(relation.attributes),
        )
    for sequence, entity in enumerate(all_entities.values()):
        attributes = dict(entity.attributes)
        relation_definition = ontology.relation(entity.relation) if entity.relation is not None else None
        graph.add_node(
            entity.id,
            id=entity.id,
            type=entity.type,
            relation=entity.relation,
            relation_category=relation_definition.category if relation_definition else None,
            projection_roles=list(relation_definition.projection) if relation_definition and relation_definition.projection else None,
            attributes=attributes,
            label=entity.type,
            node_type="entity",
            sequence=sequence,
            **{key: value for key, value in attributes.items() if key not in {"id", "type", "relation", "relation_category", "projection_roles", "attributes"}},
        )
    known = set(all_entities)
    for relation in relations:
        for index, argument in enumerate(relation.arguments):
            if argument.entity_id not in known:
                raise GraphValidationError(
                    f"relation '{relation.id}' references missing Entity '{argument.entity_id}'"
                )
            graph.add_edge(
                relation.id,
                argument.entity_id,
                key=f"argument:{relation.id}:{index}",
                edge_type="argument",
                role=argument.role,
                attributes=dict(argument.attributes),
                **{key: value for key, value in argument.attributes.items() if key not in {"edge_type", "role", "attributes"}},
            )
    validate_graph(graph, ontology)
    return graph


def project_binary_relations(graph: nx.MultiDiGraph, ontology: OntologyConfig) -> nx.MultiDiGraph:
    """Project explicitly ordered binary relation Entities into direct edges."""

    projected = nx.MultiDiGraph()
    projected.graph.update(dict(graph.graph))
    for node_id, data in graph.nodes(data=True):
        projected.add_node(node_id, **dict(data))
    for relation_id, data in _relation_nodes(graph):
        definition = ontology.relation(data["relation"])
        if definition is None or definition.projection is None:
            continue
        source_role, target_role = definition.projection
        source_targets = [target for _s, target, edge in graph.out_edges(relation_id, data=True) if edge.get("role") == source_role]
        target_targets = [target for _s, target, edge in graph.out_edges(relation_id, data=True) if edge.get("role") == target_role]
        if len(source_targets) != 1 or len(target_targets) != 1:
            continue
        relation_attrs = dict(data.get("attributes", {}))
        projected.add_edge(
            source_targets[0],
            target_targets[0],
            key=f"projection:{relation_id}",
            edge_type="projection",
            category=definition.category or "semantic",
            predicate=data["relation"],
            relation_entity_id=relation_id,
            attributes=relation_attrs,
            **{key: value for key, value in relation_attrs.items() if key not in {"edge_type", "predicate", "attributes"}},
        )
    projected.graph["projection"] = "binary_relations"
    return projected


def graph_to_dict(graph: nx.MultiDiGraph) -> dict[str, Any]:
    """Return a JSON-compatible node-link representation of a canonical graph."""

    return json.loads(json.dumps(nx.node_link_data(graph, edges="links"), ensure_ascii=False))


def graph_from_dict(payload: Mapping[str, Any]) -> nx.MultiDiGraph:
    """Reconstruct a canonical MultiDiGraph from :func:`graph_to_dict`."""

    if not isinstance(payload, Mapping):
        raise TypeError("graph payload must be a mapping")
    graph = nx.node_link_graph(dict(payload), directed=True, multigraph=True, edges="links")
    if not isinstance(graph, nx.MultiDiGraph):
        graph = nx.MultiDiGraph(graph)
    return graph
