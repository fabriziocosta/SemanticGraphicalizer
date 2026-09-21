"""Provider-neutral model client interfaces."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol


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


def as_model_client(model: ModelClient | Callable[..., Mapping[str, Any]]) -> ModelClient:
    if hasattr(model, "generate"):
        return model  # type: ignore[return-value]
    if callable(model):
        return CallableModelClient(model)
    raise TypeError("model must expose generate(...) or be callable")
