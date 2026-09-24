"""Optional conversion from semantic graphs to AbstractGraph."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import networkx as nx
import numpy as np

from .types import DocumentTrace


ParallelEdgePolicy = Literal["combine", "error"]


def _abstractgraph_types() -> tuple[type, Any]:
    try:
        from abstractgraph import AbstractGraph, sum_attribute_function
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "AbstractGraph support requires the optional 'abstractgraph' package. "
            "Install SemanticGraphicalizer with the 'abstractgraph' extra."
        ) from exc
    return AbstractGraph, sum_attribute_function


def _document_id(graph: nx.MultiDiGraph, supplied: str | None) -> str | None:
    if supplied is not None:
        return supplied
    value = graph.graph.get("document_id")
    if isinstance(value, str) and value:
        return value
    for _node_id, data in graph.nodes(data=True):
        for item in _provenance_records(data):
            candidate = item.get("document_id")
            if isinstance(candidate, str) and candidate:
                return candidate
    return None


def _provenance_records(data: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    attrs = data.get("attributes")
    value = data.get("provenance")
    if value is None and isinstance(attrs, Mapping):
        value = attrs.get("provenance")
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, (list, tuple)):
        return [record for record in value if isinstance(record, Mapping)]
    return []


def _embedding(value: Any, node_id: Any) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"embedding for node '{node_id}' must be a numeric vector") from exc
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"embedding for node '{node_id}' must be a non-empty one-dimensional vector")
    if not np.isfinite(vector).all():
        raise ValueError(f"embedding for node '{node_id}' must contain only finite values")
    return vector


def _base_graph(
    graph: nx.MultiDiGraph,
    *,
    embedding_key: str,
    parallel_edge_policy: ParallelEdgePolicy,
    preserve_direction: bool,
) -> nx.DiGraph | nx.Graph:
    base = nx.DiGraph() if preserve_direction else nx.Graph()
    base.graph.update(dict(graph.graph))

    vectors: dict[Any, np.ndarray] = {}
    for node_id, data in graph.nodes(data=True):
        semantic_attributes = data.get("attributes", {})
        if not isinstance(semantic_attributes, Mapping):
            semantic_attributes = {}
        node_data = dict(data)
        value = data.get(embedding_key, semantic_attributes.get(embedding_key))
        if value is not None:
            vector = _embedding(value, node_id)
            vectors[node_id] = vector
        node_data.update(
            {
                "id": data.get("id", node_id),
                "type": data.get("type"),
                "relation": data.get("relation"),
                "label": data.get("relation") or "",
                "semantic_attributes": dict(semantic_attributes),
            }
        )
        base.add_node(node_id, **node_data)

    dimensions = {vector.shape[0] for vector in vectors.values()}
    if len(dimensions) > 1:
        raise ValueError(f"node embeddings must have one consistent dimension; found {sorted(dimensions)}")
    if dimensions:
        dimension = next(iter(dimensions))
        for node_id in base.nodes:
            base.nodes[node_id]["attribute"] = vectors.get(node_id, np.zeros(dimension, dtype=float))

    grouped_edges: dict[Any, list[tuple[Any, Any, Any, dict[str, Any]]]] = {}
    for source, target, key, data in graph.edges(keys=True, data=True):
        edge_group = (source, target) if preserve_direction else frozenset((source, target))
        grouped_edges.setdefault(edge_group, []).append((source, target, key, dict(data)))

    for records in grouped_edges.values():
        if len(records) > 1 and parallel_edge_policy == "error":
            source, target = records[0][:2]
            raise ValueError(f"parallel semantic edges found between '{source}' and '{target}'")
        source, target = records[0][:2]
        roles = [record.get("role") for _source, _target, _key, record in records]
        distinct_roles = sorted(set(roles), key=lambda role: (role is None, str(role)))
        edge_data = dict(records[0][3])
        edge_data["label"] = distinct_roles[0] if len(distinct_roles) == 1 else tuple(distinct_roles)
        semantic_edges = []
        for edge_source, edge_target, key, record in records:
            semantic_edge = {"key": key, **record}
            if not preserve_direction:
                semantic_edge["source"] = edge_source
                semantic_edge["target"] = edge_target
            semantic_edges.append(semantic_edge)
        edge_data["semantic_edges"] = semantic_edges
        if len(records) > 1:
            edge_data["semantic_attributes"] = [
                dict(record.get("attributes", {}))
                for _source, _target, _key, record in records
            ]
        else:
            edge_data["semantic_attributes"] = dict(records[0][3].get("attributes", {}))
        base.add_edge(source, target, **edge_data)
    return base


def semantic_graph_to_abstract_graph(
    graph: nx.MultiDiGraph,
    *,
    embedding_key: str = "embedding",
    chunk_key: str = "chunk_id",
    parallel_edge_policy: ParallelEdgePolicy = "combine",
    nbits: int = 14,
    document_id: str | None = None,
    preserve_direction: bool = True,
) -> Any:
    """Convert a canonical semantic ``MultiDiGraph`` to an ``AbstractGraph``.

    The base graph retains the reified semantic nodes and directed argument
    edges. Since AbstractGraph accepts only simple graphs, parallel edges with
    the same direction and endpoints are combined; their original records are
    retained in the edge's ``semantic_edges`` attribute.

    Existing node embeddings are mapped to base-node real-valued attributes.
    This low-level converter never requests embeddings; use
    ``SemanticGraphicalizer.to_abstract_graph(..., embed_nodes=True)`` to
    compute missing or stale vectors as part of conversion.
    """

    if not isinstance(graph, nx.MultiDiGraph):
        raise TypeError("graph must be a NetworkX MultiDiGraph")
    if parallel_edge_policy not in {"combine", "error"}:
        raise ValueError("parallel_edge_policy must be 'combine' or 'error'")
    if not isinstance(embedding_key, str) or not embedding_key:
        raise ValueError("embedding_key must be a non-empty string")
    if not isinstance(chunk_key, str) or not chunk_key:
        raise ValueError("chunk_key must be a non-empty string")
    if isinstance(nbits, bool) or not isinstance(nbits, int) or nbits < 1:
        raise ValueError("nbits must be a positive integer")
    if not isinstance(preserve_direction, bool):
        raise TypeError("preserve_direction must be a bool")

    AbstractGraph, sum_attribute_function = _abstractgraph_types()
    base = _base_graph(
        graph,
        embedding_key=embedding_key,
        parallel_edge_policy=parallel_edge_policy,
        preserve_direction=preserve_direction,
    )
    abstract = AbstractGraph(graph=base, nbits=nbits, attribute_function=sum_attribute_function)
    resolved_document_id = _document_id(graph, document_id)

    groups: dict[tuple[str, str | None, str | None], list[Any]] = {}
    for node_id, data in graph.nodes(data=True):
        entity_type = data.get("type")
        if not isinstance(entity_type, str) or not entity_type:
            raise ValueError(f"semantic node '{node_id}' must have a non-empty string type")
        chunks: list[str] = []
        for provenance in _provenance_records(data):
            chunk_id = provenance.get(chunk_key)
            if isinstance(chunk_id, str) and chunk_id and chunk_id not in chunks:
                chunks.append(chunk_id)
        if chunks:
            for chunk_id in chunks:
                groups.setdefault(("chunk", chunk_id, entity_type), []).append(node_id)
        else:
            groups.setdefault(("document", resolved_document_id, entity_type), []).append(node_id)

    for (scope, scope_id, entity_type), node_ids in groups.items():
        chunk_id = scope_id if scope == "chunk" else None
        interpretation_node_id = abstract.interpretation_graph.number_of_nodes()
        abstract.create_interpretation_node_with_subgraph_from_nodes(
            node_ids,
            meta={
                "source_function": "semantic_entity_type",
                "entity_type": entity_type,
                "chunk_id": chunk_id,
                "document_id": resolved_document_id,
            },
        )
        interpretation_data = abstract.interpretation_graph.nodes[interpretation_node_id]
        # Keep the semantic label stable across documents and chunks. Scope
        # identifiers remain available in ``meta`` for provenance.
        interpretation_data["label"] = entity_type
        interpretation_data["display_label"] = entity_type
    return abstract


def trace_to_abstract_graph(
    trace: DocumentTrace,
    *,
    embedding_key: str = "embedding",
    chunk_key: str = "chunk_id",
    parallel_edge_policy: ParallelEdgePolicy = "combine",
    nbits: int = 14,
    preserve_direction: bool = True,
) -> Any:
    """Convert a trace's semantic graph to an ``AbstractGraph``."""

    if not isinstance(trace, DocumentTrace):
        raise TypeError("trace must be a DocumentTrace")
    return semantic_graph_to_abstract_graph(
        trace.graph,
        embedding_key=embedding_key,
        chunk_key=chunk_key,
        parallel_edge_policy=parallel_edge_policy,
        nbits=nbits,
        document_id=trace.document_id,
        preserve_direction=preserve_direction,
    )
