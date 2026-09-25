from hashlib import sha256

import networkx as nx
import pytest

from semantic_graphicalizer import (
    SemanticGraphicalizer,
    graph_from_dict,
    graph_to_dict,
    load_ontology,
    project_binary_relations,
)
from semantic_graphicalizer.graph import GraphValidationError, materialize_graph
from semantic_graphicalizer.pipeline import ConservativeEntityResolver, ParagraphWindowSegmenter, _format_elapsed
from semantic_graphicalizer.types import Argument, Entity, RelationInstance


ONTOLOGY = {
    "name": "recursive-test",
    "version": "1",
    "terms": [
        {"id": "Person", "description": "a person"},
        {"id": "Object", "description": "an object"},
        {"id": "Event", "description": "an event"},
        {"id": "Claim", "description": "a claim"},
        {"id": "Evidence", "description": "evidence"},
        {"id": "Experiment", "description": "an experiment"},
    ],
    "argument_roles": ["agent", "patient", "giver", "recipient", "theme", "cause", "effect", "evidence", "claim"],
    "relations": {
        "chases": {"description": "chasing", "arguments": {"agent": {"cardinality": 1, "allowed_types": ["Person"]}, "patient": {"cardinality": 1, "allowed_types": ["Person"]}}, "projection": ["agent", "patient"]},
        "gives": {"description": "giving", "arguments": {"giver": {"cardinality": 1, "allowed_types": ["Person"]}, "recipient": {"cardinality": 1, "allowed_types": ["Person"]}, "theme": {"cardinality": 1, "allowed_types": ["Object"]}}},
        "causes": {"description": "causation", "arguments": {"cause": {"cardinality": 1}, "effect": {"cardinality": 1}}},
        "supports": {"description": "support", "arguments": {"evidence": {"cardinality": "1..n"}, "claim": {"cardinality": 1}}},
    },
}


PROMPTS = {
    "domain": "recursive-test",
    "version": "1",
    "stages": {
        stage: {"system": "test", "instruction": "{text}\n{ontology}", "output": "JSON"}
        for stage in ("summarize", "normalize", "decompose", "extract", "resolve")
    },
}


class FakeModel:
    def __init__(self, extraction, resolved=None):
        self.extraction = extraction
        self.resolved = resolved or []

    def generate(self, *, stage, prompt, schema, context):
        del prompt, schema
        if stage == "summarize":
            return {"summary": "The source asserts a relation."}
        if stage == "normalize":
            return {"normalized": "The source asserts a relation."}
        if stage == "decompose":
            return {"assertions": [{"id": "a1", "text": "The source asserts a relation.", "source_text": "The source asserts a relation.", "attributes": {}}]}
        if stage == "extract":
            return self.extraction
        if stage == "resolve":
            return {"relations": self.resolved}
        raise AssertionError(stage)


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def embed(self, texts):
        texts = list(texts)
        self.calls.append(texts)
        return [[float(len(text)), float(index)] for index, text in enumerate(texts)]


def make_transformer(extraction, resolved=None):
    return SemanticGraphicalizer(ONTOLOGY, PROMPTS, FakeModel(extraction, resolved), verbose=False)


def test_compute_embeddings_stores_vectors_on_graph_nodes() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "fox",
        type="Person",
        relation=None,
        mentions=["fox"],
        attributes={"mentions": ["fox"]},
    )
    graph.add_node(
        "r1",
        type="Event",
        relation="causes",
        source_text="The fox causes the result.",
        attributes={"source_text": "The fox causes the result."},
    )
    embedder = FakeEmbedder()
    graphicalizer = SemanticGraphicalizer(
        ONTOLOGY,
        PROMPTS,
        FakeModel({"entities": [], "relations": []}),
        embedder=embedder,
        verbose=False,
    )

    result = graphicalizer.compute_embeddings(graph, batch_size=1)

    assert result == [graph]
    assert len(embedder.calls) == 2
    assert graph.nodes["fox"]["embedding"] == [len("Person | fox"), 0.0]
    assert graph.nodes["r1"]["embedding"] == [len("Event : causes | The fox causes the result."), 0.0]


def test_compute_embeddings_accepts_graphs_and_custom_text_attribute() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node("fox", type="Person", relation=None)
    embedder = FakeEmbedder()
    graphicalizer = SemanticGraphicalizer(
        ONTOLOGY,
        PROMPTS,
        FakeModel({"entities": [], "relations": []}),
        embedder=embedder,
        verbose=False,
    )

    result = graphicalizer.compute_embeddings(
        [graph],
        embedding_attribute="vector",
        node_text_fn=lambda node_id, data: f"{node_id}:{data['type']}",
    )

    assert result == [graph]
    assert graph.nodes["fox"]["vector"] == [float(len("fox:Person")), 0.0]


def test_reified_binary_relation_and_projection() -> None:
    extraction = {
        "entities": [
            {"id": "fox", "type": "Person", "mention": "fox", "key": None, "attributes": {}},
            {"id": "rabbit", "type": "Person", "mention": "rabbit", "key": None, "attributes": {}},
        ],
        "relations": [{"id": "r1", "type": "Event", "relation": "chases", "arguments": [
            {"role": "agent", "entity_id": "fox", "attributes": {}},
            {"role": "patient", "entity_id": "rabbit", "attributes": {}},
        ], "attributes": {"confidence": 0.9}}],
    }
    graph = make_transformer(extraction).transform(["source"])[0]
    assert isinstance(graph, nx.MultiDiGraph)
    relation_id = graph.graph["relations"][0]["id"]
    assert graph.nodes[relation_id]["type"] == "Event"
    assert graph.nodes[relation_id]["relation"] == "chases"
    assert {data["role"] for _, _, data in graph.out_edges(relation_id, data=True)} == {"agent", "patient"}
    projected = project_binary_relations(graph, load_ontology(ONTOLOGY))
    assert projected.has_edge("person::fox", "person::rabbit")
    assert projected.edges["person::fox", "person::rabbit", "projection:" + relation_id]["predicate"] == "chases"


def test_ternary_and_repeated_arguments_are_preserved() -> None:
    extraction = {
        "entities": [
            {"id": "john", "type": "Person", "mention": "John", "key": None, "attributes": {}},
            {"id": "mary", "type": "Person", "mention": "Mary", "key": None, "attributes": {}},
            {"id": "book", "type": "Object", "mention": "book", "key": None, "attributes": {}},
            {"id": "experiment-a", "type": "Experiment", "mention": "A", "key": None, "attributes": {}},
            {"id": "experiment-b", "type": "Experiment", "mention": "B", "key": None, "attributes": {}},
        ],
        "relations": [
            {"id": "give", "type": "Event", "relation": "gives", "arguments": [{"role": "giver", "entity_id": "john", "attributes": {}}, {"role": "recipient", "entity_id": "mary", "attributes": {}}, {"role": "theme", "entity_id": "book", "attributes": {}}], "attributes": {}},
            {"id": "support", "type": "Claim", "relation": "supports", "arguments": [{"role": "evidence", "entity_id": "experiment-a", "attributes": {}}, {"role": "evidence", "entity_id": "experiment-b", "attributes": {}}, {"role": "claim", "entity_id": "give", "attributes": {}}], "attributes": {}},
        ],
    }
    graph = make_transformer(extraction).transform(["source"])[0]
    support_id = next(relation["id"] for relation in graph.graph["relations"] if relation["relation"] == "supports")
    assert [data["role"] for _, _, data in graph.out_edges(support_id, data=True)] == ["evidence", "evidence", "claim"]
    assert graph.out_degree(support_id) == 3


def test_recursive_relation_resolution_and_provenance_are_independent() -> None:
    extraction = {
        "entities": [{"id": "experiment", "type": "Experiment", "mention": "Experiment 17", "key": None, "attributes": {}}],
        "relations": [{"id": "claim", "type": "Claim", "relation": "causes", "arguments": [{"role": "cause", "entity_id": "experiment", "attributes": {}}, {"role": "effect", "entity_id": "experiment", "attributes": {}}], "attributes": {"source_text": "claim text"}}],
    }
    resolved = [{"id": "evidence", "type": "Evidence", "relation": "supports", "arguments": [{"role": "evidence", "entity_id": "experiment::experiment_17", "attributes": {}}, {"role": "claim", "entity_id": "document-ignored:chunk-0:relation:claim", "attributes": {}}], "attributes": {"source_text": "evidence text"}}]
    # The document-specific relation ID is not known to the fixture until extraction;
    # use a model that fills it from the request context in a real integration.
    class ResolveModel(FakeModel):
        def generate(self, *, stage, prompt, schema, context):
            if stage == "resolve":
                relation_id = f"{context['document_id']}:chunk-0:relation:claim"
                return {"relations": [{**resolved[0], "arguments": [{"role": "evidence", "entity_id": "experiment::experiment_17", "attributes": {}}, {"role": "claim", "entity_id": relation_id, "attributes": {}}]}]}
            return super().generate(stage=stage, prompt=prompt, schema=schema, context=context)
    transformer = SemanticGraphicalizer(ONTOLOGY, PROMPTS, ResolveModel(extraction), verbose=False)
    graph = transformer.transform(["source"])[0]
    assert len(graph.graph["relations"]) == 2
    assert len(graph.nodes[graph.graph["relations"][0]["id"]]["attributes"]["provenance"]) == 1
    assert len(graph.nodes[graph.graph["relations"][1]["id"]]["attributes"]["provenance"]) == 1
    assert graph.has_edge(graph.graph["relations"][1]["id"], graph.graph["relations"][0]["id"])


def test_validation_detects_missing_roles_bad_types_and_dangling_references() -> None:
    ontology = load_ontology(ONTOLOGY)
    entities = [Entity("p", "Person"), Entity("o", "Object")]
    with pytest.raises(GraphValidationError, match="missing required"):
        materialize_graph(entities, [RelationInstance("r", "Event", "chases", (Argument("agent", "p"),), {})], ontology, document_id="d", document_text="x")
    with pytest.raises(GraphValidationError, match="cannot target type"):
        materialize_graph(entities, [RelationInstance("r", "Event", "chases", (Argument("agent", "p"), Argument("patient", "o")), {})], ontology, document_id="d", document_text="x")
    with pytest.raises(GraphValidationError, match="missing Entity"):
        materialize_graph(entities, [RelationInstance("r", "Event", "chases", (Argument("agent", "p"), Argument("patient", "missing")), {})], ontology, document_id="d", document_text="x")


def test_recursive_graph_round_trips_with_cycles_and_edge_attributes() -> None:
    ontology = load_ontology(ONTOLOGY)
    entities = [Entity("a", "Event"), Entity("b", "Event")]
    relations = [
        RelationInstance("r1", "Event", "causes", (Argument("cause", "a", {"ordering": 1}), Argument("effect", "r2")), {"provenance": {"source": "one"}}),
        RelationInstance("r2", "Event", "causes", (Argument("cause", "b"), Argument("effect", "r1")), {"provenance": {"source": "two"}}),
    ]
    graph = materialize_graph(entities, relations, ontology, document_id="d", document_text="x")
    restored = graph_from_dict(graph_to_dict(graph))
    assert isinstance(restored, nx.MultiDiGraph)
    assert set(restored.nodes) == set(graph.nodes)
    assert restored.number_of_edges() == graph.number_of_edges()
    assert restored["r1"]["a"]["argument:r1:0"]["ordering"] == 1


def test_graph_metadata_and_document_ids_are_stable() -> None:
    extraction = {"entities": [], "relations": []}
    text = "source"
    graph = make_transformer(extraction).transform([text])[0]
    assert graph.graph["document_id"] == f"document-{sha256(text.encode()).hexdigest()[:12]}"
    assert [stat["stage"] for stat in graph.graph["stats"]] == ["segment", "summarize", "normalize", "decompose", "extract", "resolve", "integrate", "total"]


def test_segmenter_and_resolver() -> None:
    chunks = ParagraphWindowSegmenter(max_chars=10).segment("d", "one two three four")
    assert all(len(chunk.text) <= 10 for chunk in chunks)
    assert ConservativeEntityResolver().resolve("Animal", "The Fox's") == "animal::fox"
    assert _format_elapsed(61) == "1.0 min"


def test_model_references_to_missing_entities_are_dropped() -> None:
    extraction = {"entities": [], "relations": [{"id": "r", "type": "Event", "relation": "causes", "arguments": [{"role": "cause", "entity_id": "missing", "attributes": {}}, {"role": "effect", "entity_id": "missing", "attributes": {}}], "attributes": {}}]}
    graph = make_transformer(extraction).fit_transform(["source"])[0]
    assert graph.number_of_nodes() == 0
