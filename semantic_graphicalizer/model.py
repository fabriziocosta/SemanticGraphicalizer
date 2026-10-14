"""Model client interfaces and the default OpenAI implementation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import base64
import json
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_OPENAI_IMAGE_MODEL = "gpt-5.6-terra"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"


@dataclass(frozen=True)
class ModelContent:
    """Evidence supplied to a model, such as text or a media reference."""

    modality: str
    value: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelRequest:
    """A provider-neutral structured generation request.

    ``prompt`` carries workflow instructions and rendered text context;
    ``content`` carries the source evidence without requiring it to be
    flattened into a text prompt.
    """

    stage: str
    prompt: str
    schema: Mapping[str, Any]
    context: Mapping[str, Any] = field(default_factory=dict)
    content: tuple[ModelContent, ...] = ()


class RequestModelClient(Protocol):
    """Model interface for clients that consume modality-neutral requests."""

    def generate_request(self, request: ModelRequest) -> Mapping[str, Any]: ...


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


class LegacyModelClientAdapter:
    """Adapt the existing text-only ``generate`` interface to requests."""

    def __init__(self, client: Any) -> None:
        if not callable(getattr(client, "generate", None)):
            raise TypeError("client must expose generate(...)")
        self.client = client

    def generate_request(self, request: ModelRequest) -> Mapping[str, Any]:
        unsupported = [item.modality for item in request.content if item.modality != "text"]
        if unsupported:
            raise TypeError(
                "legacy text model clients cannot consume non-text content; "
                "provide a client implementing generate_request(request)"
            )
        return self.client.generate(
            stage=request.stage,
            prompt=request.prompt,
            schema=request.schema,
            context=request.context,
        )


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

    def generate_request(self, request: ModelRequest) -> Mapping[str, Any]:
        """Generate structured output from text and/or image evidence."""

        content: list[dict[str, Any]] = [{"type": "input_text", "text": request.prompt}]
        for item in request.content:
            if item.modality == "text":
                if not isinstance(item.value, str):
                    raise TypeError("text model content must contain a string")
                content.append({"type": "input_text", "text": item.value})
            elif item.modality == "image":
                content.append({
                    "type": "input_image",
                    "image_url": self._image_data_url(item),
                    "detail": item.metadata.get("detail", "auto"),
                })
            else:
                raise TypeError(f"OpenAI Responses adapter does not support '{item.modality}' content")
        response = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": content}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": f"semantic_graphicalizer_{request.stage}",
                    "strict": True,
                    "schema": dict(request.schema),
                }
            },
            store=False,
            **self.request_options,
        )
        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str) or not output_text.strip():
            output_text = self._nested_output_text(response)
        if not isinstance(output_text, str) or not output_text.strip():
            raise ValueError(f"OpenAI returned no structured output for stage '{request.stage}'")
        try:
            result = json.loads(output_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"OpenAI returned invalid JSON for stage '{request.stage}'") from exc
        if not isinstance(result, Mapping):
            raise ValueError(f"OpenAI returned a non-object JSON result for stage '{request.stage}'")
        return result

    @staticmethod
    def _image_data_url(item: ModelContent) -> str:
        value = item.value
        if isinstance(value, Path):
            path = value
            payload = path.read_bytes()
            mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        elif isinstance(value, str):
            if value.startswith("data:image/") or value.startswith("https://") or value.startswith("http://"):
                return value
            path = Path(value)
            payload = path.read_bytes()
            mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        elif isinstance(value, (bytes, bytearray, memoryview)):
            payload = bytes(value)
            mime_type = str(item.metadata.get("mime_type", "image/png"))
        else:
            raise TypeError("image model content must be a local path, bytes, or image URL")
        encoded = base64.b64encode(payload).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

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


def as_model_client(
    model: RequestModelClient | ModelClient | Callable[..., Mapping[str, Any]],
) -> RequestModelClient:
    if callable(getattr(model, "generate_request", None)):
        return model  # type: ignore[return-value]
    if callable(getattr(model, "generate", None)):
        return LegacyModelClientAdapter(model)
    if callable(model):
        return LegacyModelClientAdapter(CallableModelClient(model))
    raise TypeError("model must expose generate_request(...), generate(...), or be callable")


def as_embedding_client(
    embedder: EmbeddingClient | Callable[[Sequence[str]], Sequence[Sequence[float]]],
) -> EmbeddingClient:
    if hasattr(embedder, "embed"):
        return embedder  # type: ignore[return-value]
    if callable(embedder):
        return CallableEmbeddingClient(embedder)
    raise TypeError("embedder must expose embed(...) or be callable")
