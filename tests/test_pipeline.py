from pathlib import Path

import networkx as nx
import pytest

from semantic_graphicalizer import SemanticGraphicalizer
from semantic_graphicalizer.exceptions import StageOutputError


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
    assert len(edges) == 1
    assert edges[0][3]["label"] == "The fox interacts with the crow."
    assert edges[0][3]["predicate"] == "interacts_with"


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
    assert graph.number_of_edges() == 2


def test_transform_with_trace_contains_all_stages_and_provenance() -> None:
    trace = make_transformer().fit(["A tale."]).transform_with_trace(["A tale."])[0]
    assert trace.summaries[0].text
    assert trace.normalized[0].text
    assert trace.propositions[0].text
    assert trace.triples[0].predicate == "interacts_with"
    edge = next(iter(trace.graph.edges(data=True)))[2]
    assert edge["provenance"]["chunk_id"].endswith("chunk-0")


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
