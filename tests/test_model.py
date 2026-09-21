from types import SimpleNamespace

import pytest

from semantic_graphicalizer.model import DEFAULT_OPENAI_MODEL, OpenAIModelClient


class FakeResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text='{"summary": "A summary."}')


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
