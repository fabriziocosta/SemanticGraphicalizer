"""Image embedding clients for the image domain."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol


DEFAULT_IMAGE_EMBEDDING_MODEL = "google/embeddinggemma-2"
IMAGE_EMBEDDING_DIMENSIONS = 768
SUPPORTED_IMAGE_EMBEDDING_DIMENSIONS = (128, 256, 512, 768)


class ImageEmbeddingClient(Protocol):
    """Minimal interface for embedding saved image parts."""

    model_id: str

    def embed_images(
        self,
        image_paths: Sequence[str | Path],
        *,
        batch_size: int = 32,
        embedding_dim: int = IMAGE_EMBEDDING_DIMENSIONS,
    ) -> Sequence[Sequence[float]]: ...


class CallableImageEmbeddingClient:
    """Adapt an image-path embedding callable to the client interface."""

    def __init__(
        self,
        function: Callable[[Sequence[str | Path], int], Sequence[Sequence[float]]],
        *,
        model_id: str | None = None,
    ) -> None:
        if not callable(function):
            raise TypeError("function must be callable")
        self.function = function
        self.model_id = model_id or (
            f"{getattr(function, '__module__', '')}."
            f"{getattr(function, '__qualname__', type(function).__qualname__)}"
        )

    def embed_images(
        self,
        image_paths: Sequence[str | Path],
        *,
        batch_size: int = 32,
        embedding_dim: int = IMAGE_EMBEDDING_DIMENSIONS,
    ) -> Sequence[Sequence[float]]:
        del batch_size
        return self.function(image_paths, embedding_dim)


class EmbeddingGemma2ImageClient:
    """Locally embed image paths with Google's multimodal EmbeddingGemma 2.

    Sentence Transformers and the model weights are loaded only on the first
    call to :meth:`embed_images`, keeping ordinary image graph extraction
    independent from the embedding stack.
    """

    precision_policy = "bfloat16-when-cuda-supported-else-float32"

    def __init__(
        self,
        model_id: str = DEFAULT_IMAGE_EMBEDDING_MODEL,
        *,
        model: Any = None,
    ) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        self.model_id = model_id
        self._model = model

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                import torch
                import sentence_transformers
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - optional dependencies
                raise ImportError(
                    "Image embeddings require Sentence Transformers with image support; "
                    "install semantic-graphicalizer[image-embeddings]."
                ) from exc
            version = tuple(
                int(component)
                for component in sentence_transformers.__version__.split(".")[:2]
                if component.isdigit()
            )
            if version < (6, 1):
                raise ImportError(
                    "EmbeddingGemma 2 requires sentence-transformers>=6.1.0; "
                    "install semantic-graphicalizer[image-embeddings]."
                )

            dtype = torch.float32
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            self._model = SentenceTransformer(
                self.model_id,
                config_kwargs={"audio_config": None},
                model_kwargs={"torch_dtype": dtype},
            )
        return self._model

    def embed_images(
        self,
        image_paths: Sequence[str | Path],
        *,
        batch_size: int = 32,
        embedding_dim: int = IMAGE_EMBEDDING_DIMENSIONS,
    ) -> Sequence[Sequence[float]]:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if (
            isinstance(embedding_dim, bool)
            or not isinstance(embedding_dim, int)
            or embedding_dim not in SUPPORTED_IMAGE_EMBEDDING_DIMENSIONS
        ):
            raise ValueError(
                f"embedding_dim must be one of {SUPPORTED_IMAGE_EMBEDDING_DIMENSIONS}"
            )
        paths = [str(path) for path in image_paths]
        if any(not path for path in paths):
            raise ValueError("image_paths must contain non-empty paths")
        model = self._load_model()
        vectors = model.encode(
            paths,
            batch_size=batch_size,
            truncate_dim=embedding_dim,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if hasattr(vectors, "tolist"):
            vectors = vectors.tolist()
        return vectors


def as_image_embedding_client(
    embedder: ImageEmbeddingClient | Callable[[Sequence[str | Path], int], Sequence[Sequence[float]]],
) -> ImageEmbeddingClient:
    if callable(getattr(embedder, "embed_images", None)):
        return embedder  # type: ignore[return-value]
    if callable(embedder):
        return CallableImageEmbeddingClient(embedder)
    raise TypeError("image_embedder must expose embed_images(...) or be callable")
