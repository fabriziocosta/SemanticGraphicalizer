"""Local text embedding client for Google EmbeddingGemma 2."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


DEFAULT_EMBEDDINGGEMMA2_MODEL = "google/embeddinggemma-2"
EMBEDDINGGEMMA2_DIMENSIONS = 768
SUPPORTED_EMBEDDINGGEMMA2_DIMENSIONS = (128, 256, 512, 768)


def _load_embeddinggemma_model(model_id: str, *, config_kwargs: dict[str, Any]) -> Any:
    try:
        import torch
        import sentence_transformers
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "EmbeddingGemma 2 requires Sentence Transformers >=6.1; "
            "install semantic-graphicalizer[text-embeddings]."
        ) from exc
    version = tuple(
        int(component)
        for component in sentence_transformers.__version__.split(".")[:2]
        if component.isdigit()
    )
    if version < (6, 1):
        raise ImportError(
            "EmbeddingGemma 2 requires sentence-transformers>=6.1.0; "
            "install semantic-graphicalizer[text-embeddings]."
        )

    dtype = torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    return SentenceTransformer(
        model_id,
        config_kwargs=config_kwargs,
        model_kwargs={"torch_dtype": dtype},
    )


class EmbeddingGemma2TextClient:
    """Lazily embed text locally with EmbeddingGemma 2.

    The default ``Document`` prompt applies the model's recommended
    ``title: none | text: `` prefix. Model weights load on first use only.
    """

    precision_policy = "bfloat16-when-cuda-supported-else-float32"

    def __init__(
        self,
        model_id: str = DEFAULT_EMBEDDINGGEMMA2_MODEL,
        *,
        embedding_dim: int = EMBEDDINGGEMMA2_DIMENSIONS,
        prompt_name: str = "Document",
        batch_size: int = 32,
        model: Any = None,
    ) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if (
            isinstance(embedding_dim, bool)
            or not isinstance(embedding_dim, int)
            or embedding_dim not in SUPPORTED_EMBEDDINGGEMMA2_DIMENSIONS
        ):
            raise ValueError(
                f"embedding_dim must be one of {SUPPORTED_EMBEDDINGGEMMA2_DIMENSIONS}"
            )
        if not isinstance(prompt_name, str) or not prompt_name.strip():
            raise ValueError("prompt_name must be a non-empty string")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.model_id = model_id
        self.model = model_id
        self.embedding_dim = embedding_dim
        self.prompt_name = prompt_name
        self.batch_size = batch_size
        self._model = model

    def _load_model(self) -> Any:
        if self._model is None:
            self._model = _load_embeddinggemma_model(
                self.model_id,
                config_kwargs={"vision_config": None, "audio_config": None},
            )
        return self._model

    def embed(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        embedding_dim: int | None = None,
    ) -> Sequence[Sequence[float]]:
        inputs = list(texts)
        if not inputs or any(not isinstance(text, str) or not text.strip() for text in inputs):
            raise ValueError("texts must contain non-empty strings")
        effective_batch_size = self.batch_size if batch_size is None else batch_size
        effective_dim = self.embedding_dim if embedding_dim is None else embedding_dim
        if isinstance(effective_batch_size, bool) or not isinstance(effective_batch_size, int) or effective_batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if (
            isinstance(effective_dim, bool)
            or not isinstance(effective_dim, int)
            or effective_dim not in SUPPORTED_EMBEDDINGGEMMA2_DIMENSIONS
        ):
            raise ValueError(
                f"embedding_dim must be one of {SUPPORTED_EMBEDDINGGEMMA2_DIMENSIONS}"
            )
        vectors = self._load_model().encode(
            inputs,
            batch_size=effective_batch_size,
            prompt_name=self.prompt_name,
            truncate_dim=effective_dim,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if hasattr(vectors, "tolist"):
            vectors = vectors.tolist()
        return vectors
