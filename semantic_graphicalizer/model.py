"""Model client interfaces and the default OpenAI implementation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
from typing import Any, Protocol


DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"


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


class EmbeddingClient(Protocol):
    """Minimal interface for text embedding providers."""

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
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


class CallableEmbeddingClient:
    """Adapt a callable to :class:`EmbeddingClient`."""

    def __init__(self, function: Callable[[Sequence[str]], Sequence[Sequence[float]]]) -> None:
        if not callable(function):
            raise TypeError("function must be callable")
        self.function = function

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return self.function(texts)


class OpenAIEmbeddingClient:
    """Request text embeddings through the OpenAI embeddings endpoint.

    The official SDK reads ``OPENAI_API_KEY`` from the environment when no
    explicit client is provided. The default model is
    ``text-embedding-3-small``.
    """

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_EMBEDDING_MODEL,
        client: Any = None,
        *,
        request_options: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise ImportError(
                    "The OpenAI embedding default requires the 'openai' package. "
                    "Install the project dependencies or provide a custom embedder."
                ) from exc
            client = OpenAI()
        if request_options is not None and not isinstance(request_options, Mapping):
            raise TypeError("request_options must be a mapping")
        reserved_options = {"model", "input"}
        if reserved_options.intersection(request_options or {}):
            raise ValueError(
                "request_options cannot override model or input"
            )
        self.model = model
        self.client = client
        self.request_options = dict(request_options or {})

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        inputs = list(texts)
        if not inputs or any(not isinstance(text, str) or not text.strip() for text in inputs):
            raise ValueError("texts must contain non-empty strings")
        request = {
            "model": self.model,
            "input": inputs,
        }
        request.update(self.request_options)
        response = self.client.embeddings.create(**request)
        data = response.get("data") if isinstance(response, Mapping) else getattr(response, "data", None)
        if not isinstance(data, (list, tuple)):
            raise ValueError("OpenAI returned no embedding data")
        ordered = sorted(
            data,
            key=lambda item: (
                item.get("index", 0) if isinstance(item, Mapping) else getattr(item, "index", 0)
            ),
        )
        vectors: list[Sequence[float]] = []
        for item in ordered:
            vector = item.get("embedding") if isinstance(item, Mapping) else getattr(item, "embedding", None)
            if not isinstance(vector, (list, tuple)):
                raise ValueError("OpenAI returned an invalid embedding vector")
            vectors.append(vector)
        if len(vectors) != len(inputs):
            raise ValueError("OpenAI returned a different number of embeddings than inputs")
        return vectors


class OpenAIModelClient:
    """Call OpenAI Responses with structured JSON output.

    The official SDK reads ``OPENAI_API_KEY`` from the environment when no
    explicit client is provided. The default model is ``gpt-4.1-mini``.
    """

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_MODEL,
        client: Any = None,
        *,
        request_options: Mapping[str, Any] | None = None,
    ) -> None:
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
        if request_options is not None and not isinstance(request_options, Mapping):
            raise TypeError("request_options must be a mapping")
        reserved_options = {"model", "input", "text", "store"}
        if reserved_options.intersection(request_options or {}):
            raise ValueError(
                "request_options cannot override model, input, text, or store"
            )
        self.model = model
        self.client = client
        self.request_options = dict(request_options or {})

    def generate(
        self,
        *,
        stage: str,
        prompt: str,
        schema: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del context
        request = {
            "model": self.model,
            "input": prompt,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": f"semantic_graphicalizer_{stage}",
                    "strict": True,
                    "schema": dict(schema),
                }
            },
            "store": False,
        }
        request.update(self.request_options)
        response = self.client.responses.create(**request)
        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str) or not output_text.strip():
            output_text = self._nested_output_text(response)
        if not isinstance(output_text, str) or not output_text.strip():
            raise ValueError(f"OpenAI returned no structured output for stage '{stage}'")
        try:
            result = json.loads(output_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"OpenAI returned invalid JSON for stage '{stage}'") from exc
        if not isinstance(result, Mapping):
            raise ValueError(f"OpenAI returned a non-object JSON result for stage '{stage}'")
        return result

    @staticmethod
    def _nested_output_text(response: Any) -> str | None:
        """Support SDK-compatible response objects without ``output_text``."""

        output = response.get("output") if isinstance(response, Mapping) else getattr(response, "output", None)
        if not isinstance(output, (list, tuple)):
            return None
        for item in output:
            content = item.get("content") if isinstance(item, Mapping) else getattr(item, "content", None)
            if not isinstance(content, (list, tuple)):
                continue
            for block in content:
                text = block.get("text") if isinstance(block, Mapping) else getattr(block, "text", None)
                if isinstance(text, str) and text.strip():
                    return text
        return None


def as_model_client(model: ModelClient | Callable[..., Mapping[str, Any]]) -> ModelClient:
    if hasattr(model, "generate"):
        return model  # type: ignore[return-value]
    if callable(model):
        return CallableModelClient(model)
    raise TypeError("model must expose generate(...) or be callable")


def as_embedding_client(
    embedder: EmbeddingClient | Callable[[Sequence[str]], Sequence[Sequence[float]]],
) -> EmbeddingClient:
    if hasattr(embedder, "embed"):
        return embedder  # type: ignore[return-value]
    if callable(embedder):
        return CallableEmbeddingClient(embedder)
    raise TypeError("embedder must expose embed(...) or be callable")
