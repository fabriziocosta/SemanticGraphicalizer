from pathlib import Path
from hashlib import sha256

import networkx as nx
import pytest

from semantic_graphicalizer import SemanticGraphicalizer
from semantic_graphicalizer.exceptions import StageOutputError
from semantic_graphicalizer.pipeline import (
    ConservativeEntityResolver,
    ParagraphWindowSegmenter,
    _format_elapsed,
)
from semantic_graphicalizer.types import EntityMention
from semantic_graphicalizer.types import Chunk


ROOT = Path(__file__).parents[1]


class FakeModel:
    def generate(self, *, stage, prompt, schema, context):
        del prompt, schema
        if stage == "summarize":
            return {"summary": "The fox meets the crow."}
        if stage == "normalize":
            return {"normalized": "A Character fox interacts with a Character crow."}
        if stage == "decompose":
            return {"propositions": [{
                "id": "p1",
                "text": "The fox interacts with the crow.",
                "source_text": "The fox met the crow.",
                "kind": "event",
            }]}
        if stage == "triple":
            return {"triples": [{
                "id": "t1",
                "proposition_id": f"{context['chunk_id']}:p1",
                "subject": {"mention": "fox", "label": "Animal"},
                "predicate": "interacts_with",
                "object": {"mention": "crow", "label": "Animal"},
                "proposition": "The fox interacts with the crow.",
            }]}
        if stage == "link":
            return {"links": [], "state_intervals": []}
        raise AssertionError(stage)


def make_transformer(model=None):
    return SemanticGraphicalizer(
        ROOT / "configs/ontologies/aesop.yaml",
        ROOT / "configs/prompts/aesop.yaml",
        model or FakeModel(),
    )


def test_transformer_returns_one_multidigraph_per_document() -> None:
    graphs = make_transformer().fit_transform(["The fox met the crow.", "Another tale."])
    assert len(graphs) == 2
    assert all(isinstance(graph, nx.MultiDiGraph) for graph in graphs)
    graph = graphs[0]
    assert graph.nodes["animal::fox"]["label"] == "Animal"
    edges = list(graph.edges(data=True, keys=True))
    semantic_edges = [edge for edge in edges if edge[3].get("edge_type") == "semantic"]
    assert len(semantic_edges) == 1
    assert semantic_edges[0][3]["label"] == "The fox interacts with the crow."
    assert semantic_edges[0][3]["predicate"] == "interacts_with"
    assert any(data.get("node_type") == "proposition" for _, data in graph.nodes(data=True))
    expected_document_id = f"document-{sha256('The fox met the crow.'.encode()).hexdigest()[:12]}"
    assert graph.graph["document_id"] == expected_document_id
    assert graph.graph["document_text"] == "The fox met the crow."
    assert graph.nodes["animal::fox"]["document_id"] == expected_document_id
    assert graph.nodes["animal::fox"]["document_text"] == "The fox met the crow."
    assert edges[0][3]["document_id"] == expected_document_id
    assert edges[0][3]["document_text"] == "The fox met the crow."


def test_transformer_preserves_multiple_edges() -> None:
    class TwoEdgeModel(FakeModel):
        def generate(self, *, stage, prompt, schema, context):
            response = super().generate(stage=stage, prompt=prompt, schema=schema, context=context)
            if stage == "triple":
                response["triples"].append({
                    "id": "t2",
                    "proposition_id": f"{context['chunk_id']}:p1",
                    "subject": {"mention": "fox", "label": "Animal"},
                    "predicate": "causes",
                    "object": {"mention": "crow", "label": "Action"},
                    "proposition": "The fox causes an encounter.",
                })
            return response

    graph = make_transformer(TwoEdgeModel()).fit_transform(["A tale."])[0]
    assert sum(data.get("edge_type") == "semantic" for _, _, data in graph.edges(data=True)) == 2


def test_transform_with_trace_contains_all_stages_and_provenance() -> None:
    trace = make_transformer().fit(["The fox met the crow."]).transform_with_trace(["The fox met the crow."])[0]
    assert trace.summaries[0].text
    assert trace.normalized[0].text
    assert trace.propositions[0].text
    assert trace.triples[0].predicate == "interacts_with"
    assert [stat.stage for stat in trace.stats] == [
        "segment", "summarize", "normalize", "decompose", "triple", "link", "integrate", "total",
    ]
    assert all(stat.elapsed_seconds >= 0 for stat in trace.stats)
    assert trace.stats[-1].details["edges"] == trace.graph.number_of_edges()
    edge = next(iter(trace.graph.edges(data=True)))[2]
    assert edge["provenance"]["chunk_id"].endswith("chunk-0")
    assert edge["provenance"]["start_char"] == 0
    assert edge["provenance"]["end_char"] == len("The fox met the crow.")
    assert edge["provenance"]["source_text"] == "The fox met the crow."
    node_provenance = trace.graph.nodes["animal::fox"]["provenance"][0]
    assert node_provenance["mention_start_char"] == 4
    assert node_provenance["mention_end_char"] == 7


def test_reified_graph_contains_narrative_and_explicit_causal_links() -> None:
    class NarrativeModel(FakeModel):
        def generate(self, *, stage, prompt, schema, context):
            if stage == "decompose":
                return {"propositions": [
                    {
                        "id": "p1",
                        "text": "The fox runs.",
                        "source_text": "The fox runs.",
                        "kind": "event",
                        "confidence": None,
                        "qualification": {},
                    },
                    {
                        "id": "p2",
                        "text": "The fox arrives.",
                        "source_text": "The fox arrives.",
                        "kind": "event",
                        "confidence": None,
                        "qualification": {},
                    },
                ]}
            if stage == "link":
                document_id = context["document_id"]
                return {
                    "links": [{
                        "source_proposition_id": f"{document_id}:chunk-0:p1",
                        "target_proposition_id": f"{document_id}:chunk-0:p2",
                        "predicate": "causes",
                        "confidence": 0.9,
                        "qualification": {},
                    }],
                    "state_intervals": [],
                }
            return super().generate(stage=stage, prompt=prompt, schema=schema, context=context)

    trace = SemanticGraphicalizer(
        ROOT / "configs/ontologies/aesop.yaml",
        ROOT / "configs/prompts/aesop.yaml",
        NarrativeModel(),
        verbose=False,
    ).fit(["The fox runs. The fox arrives."]).transform_with_trace(["The fox runs. The fox arrives."])[0]

    proposition_nodes = [
        node for node, data in trace.graph.nodes(data=True)
        if data.get("node_type") == "proposition"
    ]
    temporal_edges = [
        data for _, _, data in trace.graph.edges(data=True)
        if data.get("predicate") == "next_in_narrative"
    ]
    causal_edges = [
        data for _, _, data in trace.graph.edges(data=True)
        if data.get("predicate") == "causes"
        and data.get("edge_type") == "causal"
    ]

    assert [trace.graph.nodes[node]["sequence"] for node in proposition_nodes] == [0, 1]
    assert len(temporal_edges) == 1
    assert len(causal_edges) == 1
    assert trace.links[0].category == "causal"
    assert trace.semantic_graph is not None


def test_sequence_does_not_create_implicit_causal_edges() -> None:
    trace = make_transformer().fit(["The fox met the crow."]).transform_with_trace(
        ["The fox met the crow."]
    )[0]

    assert [link.predicate for link in trace.links] == []
    assert [
        data for _, _, data in trace.graph.edges(data=True)
        if data.get("edge_type") == "causal"
    ] == []
    assert sum(
        data.get("predicate") == "next_in_narrative"
        for _, _, data in trace.graph.edges(data=True)
    ) == 0


def test_invalid_proposition_kind_is_rejected() -> None:
    class InvalidKindModel(FakeModel):
        def generate(self, *, stage, prompt, schema, context):
            if stage == "decompose":
                return {"propositions": [{
                    "id": "p1",
                    "text": "The fox interacts with the crow.",
                    "source_text": "The fox met the crow.",
                    "kind": "unknown",
                }]}
            return super().generate(stage=stage, prompt=prompt, schema=schema, context=context)

    with pytest.raises(StageOutputError, match="'kind' must be one"):
        make_transformer(InvalidKindModel()).fit_transform(["A tale."])


@pytest.mark.parametrize("invalid_link", [
    {
        "source_proposition_id": "missing",
        "target_proposition_id": "also-missing",
        "predicate": "causes",
        "confidence": None,
        "qualification": {},
    },
    {
        "source_proposition_id": "placeholder",
        "target_proposition_id": "placeholder",
        "predicate": "not_configured",
        "confidence": None,
        "qualification": {},
    },
])
def test_invalid_proposition_links_are_rejected(invalid_link) -> None:
    class InvalidLinkModel(FakeModel):
        def generate(self, *, stage, prompt, schema, context):
            if stage == "decompose":
                return {"propositions": [{
                    "id": "p1",
                    "text": "The fox interacts with the crow.",
                    "source_text": "The fox met the crow.",
                    "kind": "event",
                }]}
            if stage == "link":
                document_id = context["document_id"]
                link = dict(invalid_link)
                if link["source_proposition_id"] == "placeholder":
                    link["source_proposition_id"] = f"{document_id}:chunk-0:p1"
                    link["target_proposition_id"] = f"{document_id}:chunk-0:p1"
                return {"links": [link], "state_intervals": []}
            return super().generate(stage=stage, prompt=prompt, schema=schema, context=context)

    with pytest.raises(StageOutputError, match="(known propositions|unknown proposition link relation)"):
        make_transformer(InvalidLinkModel()).fit_transform(["A tale."])


def test_state_interval_edges_are_reified_and_validated() -> None:
    class StateModel(FakeModel):
        def generate(self, *, stage, prompt, schema, context):
            if stage == "decompose":
                return {"propositions": [
                    {"id": "p1", "text": "The fox runs.", "source_text": "The fox runs.", "kind": "event", "confidence": None, "qualification": {}},
                    {"id": "s1", "text": "The fox is alert.", "source_text": "The fox is alert.", "kind": "state", "confidence": None, "qualification": {}},
                    {"id": "p2", "text": "The fox arrives.", "source_text": "The fox arrives.", "kind": "event", "confidence": None, "qualification": {}},
                ]}
            if stage == "link":
                document_id = context["document_id"]
                return {
                    "links": [],
                    "state_intervals": [{
                        "state_proposition_id": f"{document_id}:chunk-0:s1",
                        "starts_at": f"{document_id}:chunk-0:p1",
                        "ends_at": f"{document_id}:chunk-0:p2",
                    }],
                }
            return super().generate(stage=stage, prompt=prompt, schema=schema, context=context)

    trace = SemanticGraphicalizer(
        ROOT / "configs/ontologies/aesop.yaml",
        ROOT / "configs/prompts/aesop.yaml",
        StateModel(),
        verbose=False,
    ).fit(["The fox runs. The fox is alert. The fox arrives."]).transform_with_trace(
        ["The fox runs. The fox is alert. The fox arrives."]
    )[0]

    interval_edges = [
        data for _, _, data in trace.graph.edges(data=True)
        if data.get("edge_type") == "state_interval"
    ]
    assert {edge["predicate"] for edge in interval_edges} == {"starts_at", "ends_at"}
    assert trace.state_intervals[0].state_proposition_id.endswith(":s1")


def test_document_ids_are_stable_across_transform_batches() -> None:
    transformer = make_transformer().fit(["A tale."])
    first = transformer.transform(["A tale."])[0]
    second = transformer.transform(["A tale."])[0]

    assert first.graph["document_id"] == second.graph["document_id"]


def test_overlapping_duplicate_triples_are_integrated_once() -> None:
    class DuplicateSegmenter:
        def segment(self, document_id, text):
            return [
                Chunk(document_id, f"{document_id}:chunk-0", text, 0, len(text)),
                Chunk(document_id, f"{document_id}:chunk-1", text, 0, len(text)),
            ]

    transformer = SemanticGraphicalizer(
        ROOT / "configs/ontologies/aesop.yaml",
        ROOT / "configs/prompts/aesop.yaml",
        FakeModel(),
        segmenter=DuplicateSegmenter(),
    )

    graph = transformer.fit_transform(["The fox met the crow."])[0]

    assert sum(data.get("edge_type") == "semantic" for _, _, data in graph.edges(data=True)) == 1


def test_transformer_display_accepts_trace_graph() -> None:
    transformer = make_transformer().fit(["A tale."])
    trace = transformer.transform_with_trace(["A tale."])[0]
    rendered = transformer.display(trace)
    assert "data:text/html;charset=utf-8," in rendered._repr_html_()
    assert rendered.src.startswith("data:text/html;charset=utf-8,")


def test_verbose_reports_pipeline_stages_and_runtimes(capsys) -> None:
    make_transformer().fit_transform(["A tale."])
    output = capsys.readouterr().out
    for stage in ("segment", "summarize", "normalize", "decompose", "triple", "link", "integrate", "total"):
        assert f"] {stage}:" in output
    assert any(unit in output for unit in ("ms", "s", "min"))


def test_elapsed_time_uses_largest_practical_unit() -> None:
    assert _format_elapsed(0.125) == "125.0 ms"
    assert _format_elapsed(5.2309) == "5.2 s"
    assert _format_elapsed(60) == "1.0 min"
    assert _format_elapsed(125) == "2.1 min"


def test_segmenter_breaks_long_paragraphs_on_word_boundaries() -> None:
    text = "one two three four five six seven eight nine ten"
    chunks = ParagraphWindowSegmenter(max_chars=18).segment("document-0", text)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 18 for chunk in chunks)
    assert all(chunk.text and not chunk.text[0].isspace() for chunk in chunks)
    assert all(text[chunk.start_char:chunk.end_char] == chunk.text for chunk in chunks)
    assert all(not chunk.text.endswith((" ", "\t")) for chunk in chunks)


def test_entity_resolver_stabilizes_determiners_and_possessives() -> None:
    resolver = ConservativeEntityResolver()

    assert resolver.resolve(EntityMention("the fox", "Animal")) == "animal::fox"
    assert resolver.resolve(EntityMention("fox's", "Animal")) == "animal::fox"
    assert resolver.resolve(EntityMention("fox", "Human")) == "human::fox"


def test_model_failures_are_retried_and_wrapped() -> None:
    class FlakyModel(FakeModel):
        def __init__(self):
            self.calls = 0

        def generate(self, *, stage, prompt, schema, context):
            self.calls += 1
            if self.calls < 3:
                raise TimeoutError("temporary provider failure")
            return super().generate(stage=stage, prompt=prompt, schema=schema, context=context)

    model = FlakyModel()
    graph = SemanticGraphicalizer(
        ROOT / "configs/ontologies/aesop.yaml",
        ROOT / "configs/prompts/aesop.yaml",
        model,
        retry_backoff=0,
    ).fit_transform(["A tale."])[0]
    assert sum(data.get("edge_type") == "semantic" for _, _, data in graph.edges(data=True)) == 1
    assert model.calls == 7

    class AlwaysFailModel(FakeModel):
        def generate(self, **kwargs):
            raise TimeoutError("provider unavailable")

    with pytest.raises(StageOutputError, match=r"after 2 attempt\(s\)"):
        SemanticGraphicalizer(
            ROOT / "configs/ontologies/aesop.yaml",
            ROOT / "configs/prompts/aesop.yaml",
            AlwaysFailModel(),
            max_retries=1,
            retry_backoff=0,
        ).fit_transform(["A tale."])

    class NonRetryableModel(FakeModel):
        def __init__(self):
            self.calls = 0

        def generate(self, **kwargs):
            self.calls += 1
            raise ValueError("invalid request")

    non_retryable = NonRetryableModel()
    with pytest.raises(StageOutputError, match="after 1 attempt"):
        SemanticGraphicalizer(
            ROOT / "configs/ontologies/aesop.yaml",
            ROOT / "configs/prompts/aesop.yaml",
            non_retryable,
            max_retries=3,
            retry_backoff=0,
        ).fit_transform(["A tale."])
    assert non_retryable.calls == 1


def test_fit_transform_accepts_single_use_iterators() -> None:
    documents = (document for document in ["A tale."])
    graphs = make_transformer().fit_transform(documents)
    assert len(graphs) == 1


def test_verbose_false_suppresses_progress_output(capsys) -> None:
    transformer = SemanticGraphicalizer(
        ROOT / "configs/ontologies/aesop.yaml",
        ROOT / "configs/prompts/aesop.yaml",
        FakeModel(),
        verbose=False,
    )
    trace = transformer.fit(["A tale."]).transform_with_trace(["A tale."])[0]
    assert capsys.readouterr().out == ""
    assert trace.stats[-1].stage == "total"


def test_input_must_be_an_iterable_of_documents() -> None:
    transformer = make_transformer()
    with pytest.raises(ValueError, match="not one string"):
        transformer.fit("one document")


def test_malformed_stage_output_is_explicit() -> None:
    class BadModel(FakeModel):
        def generate(self, *, stage, prompt, schema, context):
            if stage == "summarize":
                return {"summary": ""}
            return super().generate(stage=stage, prompt=prompt, schema=schema, context=context)

    with pytest.raises(StageOutputError, match="summarize"):
        make_transformer(BadModel()).fit_transform(["A tale."])


def test_default_model_factory_is_used_when_model_is_omitted(monkeypatch) -> None:
    import semantic_graphicalizer.transformer as transformer_module

    created = []

    class DefaultModel:
        def __init__(self):
            created.append(self)

        def generate(self, **kwargs):
            return FakeModel().generate(**kwargs)

    monkeypatch.setattr(transformer_module, "OpenAIModelClient", DefaultModel)
    transformer = SemanticGraphicalizer(
        ROOT / "configs/ontologies/aesop.yaml",
        ROOT / "configs/prompts/aesop.yaml",
    ).fit(["A tale."])
    assert transformer.pipeline_.model is created[0]


def test_triple_schema_encodes_relation_endpoint_constraints() -> None:
    class SchemaCapture(FakeModel):
        def __init__(self):
            self.schemas = {}

        def generate(self, *, stage, prompt, schema, context):
            self.schemas[stage] = schema
            return super().generate(stage=stage, prompt=prompt, schema=schema, context=context)

    model = SchemaCapture()
    make_transformer(model).fit_transform(["A tale."])

    branches = model.schemas["triple"]["properties"]["triples"]["items"]["anyOf"]
    by_relation = {
        branch["properties"]["predicate"]["enum"][0]: branch
        for branch in branches
    }
    has_trait = by_relation["has_trait"]["properties"]
    assert has_trait["subject"]["properties"]["label"]["enum"] == [
        "Animal", "Character", "Human",
    ]
    assert has_trait["object"]["properties"]["label"]["enum"] == ["Trait"]
    link_schema = model.schemas["link"]
    assert link_schema["properties"]["links"]["items"]["properties"]["predicate"]["enum"] == [
        "after", "before", "causes",
    ]
    assert link_schema["required"] == ["links", "state_intervals"]
