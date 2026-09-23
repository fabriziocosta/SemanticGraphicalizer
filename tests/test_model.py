from types import SimpleNamespace

import pytest

from semantic_graphicalizer.model import (
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    DEFAULT_OPENAI_MODEL,
    OpenAIEmbeddingClient,
    OpenAIModelClient,
)


class FakeResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text='{"summary": "A summary."}')


class FakeEmbeddings:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(data=[
            SimpleNamespace(index=1, embedding=[0.2, 0.3]),
            SimpleNamespace(index=0, embedding=[0.1, 0.2]),
        ])


def test_openai_client_uses_gpt_4_1_mini_and_structured_outputs() -> None:
    responses = FakeResponses()
    client = SimpleNamespace(responses=responses)
    model = OpenAIModelClient(client=client)

    result = model.generate(
        stage="summarize",
        prompt="Summarize this.",
        schema={
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": False,
        },
        context={},
    )

    assert model.model == DEFAULT_OPENAI_MODEL == "gpt-4.1-mini"
    assert result == {"summary": "A summary."}
    assert responses.calls[0]["model"] == "gpt-4.1-mini"
    assert responses.calls[0]["text"]["format"]["type"] == "json_schema"
    assert responses.calls[0]["store"] is False


def test_openai_client_accepts_request_options_and_nested_sdk_output() -> None:
    responses = FakeResponses()
    responses.create = lambda **kwargs: (  # type: ignore[method-assign]
        responses.calls.append(kwargs)
        or {"output": [{"content": [{"text": '{"summary": "Nested."}'}]}]}
    )
    client = SimpleNamespace(responses=responses)
    model = OpenAIModelClient(client=client, request_options={"timeout": 12})

    result = model.generate(stage="summarize", prompt="Summarize.", schema={}, context={})

    assert result == {"summary": "Nested."}
    assert responses.calls[0]["timeout"] == 12


def test_openai_client_protects_structured_request_fields() -> None:
    client = SimpleNamespace(responses=FakeResponses())

    with pytest.raises(ValueError, match="cannot override"):
        OpenAIModelClient(client=client, request_options={"text": {}})


def test_openai_embedding_client_batches_and_restores_input_order() -> None:
    embeddings = FakeEmbeddings()
    client = SimpleNamespace(embeddings=embeddings)
    embedder = OpenAIEmbeddingClient(client=client)

    result = embedder.embed(["first", "second"])

    assert embedder.model == DEFAULT_OPENAI_EMBEDDING_MODEL == "text-embedding-3-small"
    assert result == [[0.1, 0.2], [0.2, 0.3]]
    assert embeddings.calls[0]["model"] == "text-embedding-3-small"
    assert embeddings.calls[0]["input"] == ["first", "second"]


def test_openai_embedding_client_protects_request_fields() -> None:
    client = SimpleNamespace(embeddings=FakeEmbeddings())

    with pytest.raises(ValueError, match="cannot override"):
        OpenAIEmbeddingClient(client=client, request_options={"input": []})
