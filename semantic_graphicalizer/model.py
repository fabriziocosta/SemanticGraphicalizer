"""Model client interfaces and the default OpenAI implementation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from typing import Any, Protocol


DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"


class ModelClient(Protocol):
    """Minimal interface required by the staged pipeline.

    Implementations may call a hosted model, a local model, or a deterministic
    test double. The returned value must already be parsed as a mapping.
    """

    def generate(
        self,
        *,
        stage: str,
        prompt: str,
        schema: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        ...


class CallableModelClient:
    """Adapt a callable to :class:`ModelClient`."""

    def __init__(self, function: Callable[..., Mapping[str, Any]]) -> None:
        if not callable(function):
            raise TypeError("function must be callable")
        self.function = function

    def generate(
        self,
        *,
        stage: str,
        prompt: str,
        schema: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return self.function(stage=stage, prompt=prompt, schema=schema, context=context)


class OpenAIModelClient:
    """Call OpenAI Responses with structured JSON output.

    The official SDK reads ``OPENAI_API_KEY`` from the environment when no
    explicit client is provided. The default model is ``gpt-4.1-mini``.
    """

    def __init__(self, model: str = DEFAULT_OPENAI_MODEL, client: Any = None) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise ImportError(
                    "The OpenAI default requires the 'openai' package. "
                    "Install the project dependencies or provide a custom model client."
                ) from exc
            client = OpenAI()
        self.model = model
        self.client = client

    def generate(
        self,
        *,
        stage: str,
        prompt: str,
        schema: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del context
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": f"semantic_graphicalizer_{stage}",
                    "strict": True,
                    "schema": dict(schema),
                }
            },
            store=False,
        )
        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str) or not output_text.strip():
            raise ValueError(f"OpenAI returned no structured output for stage '{stage}'")
        try:
            result = json.loads(output_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"OpenAI returned invalid JSON for stage '{stage}'") from exc
        if not isinstance(result, Mapping):
            raise ValueError(f"OpenAI returned a non-object JSON result for stage '{stage}'")
        return result


def as_model_client(model: ModelClient | Callable[..., Mapping[str, Any]]) -> ModelClient:
    if hasattr(model, "generate"):
        return model  # type: ignore[return-value]
    if callable(model):
        return CallableModelClient(model)
    raise TypeError("model must expose generate(...) or be callable")
