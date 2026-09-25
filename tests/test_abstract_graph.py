from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from semantic_graphicalizer import (
    SemanticGraphicalizer,
    semantic_graph_to_abstract_graph,
    vectorize_abstract_graphs,
)


class FakeModel:
    def generate(self, *, stage, prompt, schema, context):  # pragma: no cover - not used
        raise AssertionError(stage)


class FakeEmbedder:
    model = "fake-v1"

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed(self, texts):
        values = list(texts)
        self.calls.append(values)
        return [[float(len(value)), 1.0] for value in values]


def graph_with_chunks() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(document_id="doc")
    graph.add_node(
        "fox", id="fox", type="Character", relation=None,
        attributes={"provenance": [{"document_id": "doc", "chunk_id": "c1"}, {"document_id": "doc", "chunk_id": "c2"}]},
        provenance=[{"document_id": "doc", "chunk_id": "c1"}, {"document_id": "doc", "chunk_id": "c2"}],
        embedding=[1.0, 2.0],
    )
    graph.add_node(
        "rabbit", id="rabbit", type="Character", relation=None,
        attributes={"provenance": [{"document_id": "doc", "chunk_id": "c1"}]},
        provenance=[{"document_id": "doc", "chunk_id": "c1"}],
    )
    graph.add_node(
        "r1", id="r1", type="Action", relation="chases",
        attributes={"confidence": 0.9, "provenance": [{"document_id": "doc", "chunk_id": "c1"}]},
        provenance=[{"document_id": "doc", "chunk_id": "c1"}],
        embedding=[3.0, 4.0],
    )
    graph.add_edge("r1", "fox", key="arg0", edge_type="argument", role="agent", attributes={"ordering": 1})
    graph.add_edge("r1", "fox", key="arg1", edge_type="argument", role="observer", attributes={"ordering": 2})
    graph.add_edge("r1", "rabbit", key="arg2", edge_type="argument", role="patient", attributes={"ordering": 3})
    return graph


def make_graphicalizer(embedder=None):
    return SemanticGraphicalizer(
        {"name": "test", "terms": [], "argument_roles": [], "relations": {}},
        {"domain": "test", "stages": {}},
        FakeModel(),
        embedder=embedder,
        verbose=False,
    )


def test_adapter_maps_semantic_labels_metadata_and_chunk_groups():
    abstract = semantic_graph_to_abstract_graph(
        graph_with_chunks(), nbits=4, interpretation_mode="by_chunk_and_type"
    )
    base = abstract.base_graph

    assert base.is_directed()
    assert base.nodes["fox"]["label"] == ""
    assert base.nodes["r1"]["label"] == "chases"
    assert base.nodes["r1"]["semantic_attributes"]["confidence"] == 0.9
    assert base.nodes["fox"]["attribute"].tolist() == [1.0, 2.0]
    assert base.nodes["rabbit"]["attribute"].tolist() == [0.0, 0.0]
    assert base.edges["r1", "fox"]["label"] == ("agent", "observer")
    assert len(base.edges["r1", "fox"]["semantic_edges"]) == 2
    assert base.edges["r1", "rabbit"]["label"] == "patient"

    by_group = {
        (data["meta"]["chunk_id"], data["meta"]["entity_type"]): set(data["mapped_subgraph"].nodes)
        for _node, data in abstract.interpretation_graph.nodes(data=True)
    }
    assert by_group == {
        ("c1", "Character"): {"fox", "rabbit"},
        ("c1", "Action"): {"r1"},
        ("c2", "Character"): {"fox"},
    }


def test_parallel_edge_error_and_embedding_dimension_validation():
    with pytest.raises(ValueError, match="parallel semantic edges"):
        semantic_graph_to_abstract_graph(graph_with_chunks(), parallel_edge_policy="error")

    graph = graph_with_chunks()
    graph.nodes["rabbit"]["embedding"] = [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="consistent dimension"):
        semantic_graph_to_abstract_graph(graph)


def test_document_scope_fallback_from_graph_metadata():
    graph = nx.MultiDiGraph(document_id="doc")
    graph.add_node("one", id="one", type="Character", relation=None, attributes={})
    graph.add_node("two", id="two", type="Character", relation=None, attributes={})
    abstract = semantic_graph_to_abstract_graph(
        graph, nbits=4, interpretation_mode="by_chunk_and_type"
    )
    assert abstract.interpretation_graph.number_of_nodes() == 1
    data = next(iter(abstract.interpretation_graph.nodes(data=True)))[1]
    assert data["meta"] == {
        "source_function": "semantic_entity_type",
        "entity_type": "Character",
        "chunk_id": None,
        "document_id": "doc",
    }


def test_graph_embedding_is_opt_in_and_matching_vectors_are_reused():
    graph = nx.MultiDiGraph(document_id="doc")
    graph.add_node("fox", id="fox", type="Character", relation=None, mentions=["fox"], attributes={})
    embedder = FakeEmbedder()
    graphicalizer = make_graphicalizer(embedder)

    no_embed = graphicalizer.to_abstract_graph(graph, nbits=4)
    assert embedder.calls == []
    assert "attribute" not in no_embed.base_graph.nodes["fox"]

    first = graphicalizer.to_abstract_graph(graph, embed_nodes=True, nbits=4)
    assert len(embedder.calls) == 1
    assert first.base_graph.nodes["fox"]["attribute"].tolist() == [15.0, 1.0]
    second = graphicalizer.to_abstract_graph(graph, embed_nodes=True, nbits=4)
    assert len(embedder.calls) == 1
    assert second.base_graph.nodes["fox"]["attribute"].tolist() == [15.0, 1.0]

    graph.nodes["fox"]["mentions"] = ["red fox"]
    graphicalizer.to_abstract_graph(graph, embed_nodes=True, nbits=4)
    assert len(embedder.calls) == 2


def test_summed_abstractgraph_vectors_have_stable_width():
    first = semantic_graph_to_abstract_graph(graph_with_chunks(), nbits=4)
    second_graph = nx.MultiDiGraph(document_id="other")
    second_graph.add_node("owl", id="owl", type="Character", relation=None, attributes={}, embedding=[0.5, 0.25])
    second = semantic_graph_to_abstract_graph(second_graph, nbits=4)

    first_vector = np.asarray(first.to_array().sum(axis=0)).ravel()
    second_vector = np.asarray(second.to_array().sum(axis=0)).ravel()
    assert first_vector.shape == second_vector.shape == (16 * 2,)


def test_graph_level_transformer_uses_simple_base_graphs_for_abstract_graphs():
    abstract = semantic_graph_to_abstract_graph(graph_with_chunks(), nbits=4)
    from abstractgraph import vectorize

    expected = np.asarray(
        vectorize(abstract.copy(), nbits=4, return_dense=False).sum(axis=0)
    ).ravel()
    interpretation_count = abstract.interpretation_graph.number_of_nodes()

    matrix = vectorize_abstract_graphs([abstract], nbits=4)

    assert matrix.shape == (1, expected.size)
    np.testing.assert_allclose(matrix.toarray()[0], expected)
    assert abstract.interpretation_graph.number_of_nodes() == interpretation_count


def test_transform_abstract_convenience_and_lazy_optional_dependency(monkeypatch):
    graph = nx.MultiDiGraph(document_id="doc")
    graph.add_node("fox", id="fox", type="Character", relation=None, attributes={})
    graphicalizer = make_graphicalizer()
    monkeypatch.setattr(graphicalizer, "transform", lambda _texts: [graph])
    assert len(graphicalizer.transform_abstract(["text"], nbits=4)) == 1

    import semantic_graphicalizer.abstract_graph as adapter

    monkeypatch.setattr(adapter, "_abstractgraph_types", lambda: (_ for _ in ()).throw(ImportError(
        "AbstractGraph support requires the optional 'abstractgraph' package. Install SemanticGraphicalizer with the 'abstractgraph' extra."
    )))
    with pytest.raises(ImportError, match="optional 'abstractgraph'"):
        semantic_graph_to_abstract_graph(graph)
