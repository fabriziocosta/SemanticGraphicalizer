"""Graph attachment workflow for saved image-part embeddings."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

import networkx as nx

from .embedding_utils import project_embedding_vectors
from .image_embeddings import (
    IMAGE_EMBEDDING_DIMENSIONS,
    SUPPORTED_IMAGE_EMBEDDING_DIMENSIONS,
    EmbeddingGemma2ImageClient,
    as_image_embedding_client,
)


class ImageEmbeddingWorkflow:
    """Compute and cache embeddings on an image graph's saved part records."""

    def compute_part_embeddings(
        self,
        values: nx.MultiDiGraph | Iterable[nx.MultiDiGraph],
        *,
        batch_size: int = 32,
        embedding_dim: int | None = None,
        source_embedding_dim: int | None = None,
        skip_matching: bool = True,
    ) -> list[nx.MultiDiGraph]:
        """Embed saved image parts and attach vectors to graph part records.

        Graphs are updated in place and returned in input order. The configured
        image embedder and output dimension from ``__init__`` are used unless
        overridden for this call. Output dimensions supported natively by
        EmbeddingGemma 2 use its Matryoshka truncation. Other smaller dimensions
        are produced with a seeded Gaussian random projection from
        ``source_embedding_dim`` (or 768 by default). The default client and
        model weights load only if a part needs an embedding. Entity provenance
        references vectors through each part's existing ``part_id``.
        """

        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(skip_matching, bool):
            raise TypeError("skip_matching must be a bool")
        if embedding_dim is None:
            embedding_dim = self.image_embedding_dim
        if (
            isinstance(embedding_dim, bool)
            or not isinstance(embedding_dim, int)
            or embedding_dim < 1
        ):
            raise ValueError("embedding_dim must be a positive integer")
        if source_embedding_dim is None:
            source_embedding_dim = self.image_embedding_source_dim
        if source_embedding_dim is None:
            source_embedding_dim = (
                embedding_dim
                if embedding_dim in SUPPORTED_IMAGE_EMBEDDING_DIMENSIONS
                else IMAGE_EMBEDDING_DIMENSIONS
            )
        if (
            isinstance(source_embedding_dim, bool)
            or not isinstance(source_embedding_dim, int)
            or source_embedding_dim not in SUPPORTED_IMAGE_EMBEDDING_DIMENSIONS
        ):
            raise ValueError(
                f"source_embedding_dim must be one of {SUPPORTED_IMAGE_EMBEDDING_DIMENSIONS}"
            )
        if embedding_dim > source_embedding_dim:
            raise ValueError("embedding_dim cannot exceed source_embedding_dim")
        if isinstance(values, nx.MultiDiGraph):
            graphs = [values]
        elif isinstance(values, (str, bytes)):
            raise TypeError("values must be image graphs or an iterable of image graphs")
        else:
            try:
                graphs = list(values)
            except TypeError as exc:
                raise TypeError("values must be image graphs or an iterable of image graphs") from exc
        if not all(isinstance(graph, nx.MultiDiGraph) for graph in graphs):
            raise TypeError("values must contain only NetworkX MultiDiGraph objects")

        part_records: list[tuple[nx.MultiDiGraph, dict[str, Any], str, str]] = []
        for graph in graphs:
            if graph.graph.get("modality") != "image":
                raise ValueError("compute_part_embeddings accepts only image graphs")
            parts = graph.graph.get("parts", [])
            if not isinstance(parts, list):
                raise ValueError("image graph metadata 'parts' must be a list")
            for index, raw_part in enumerate(parts):
                if not isinstance(raw_part, Mapping):
                    raise ValueError(f"image graph part {index} must be an object")
                part = raw_part if isinstance(raw_part, dict) else dict(raw_part)
                if part is not raw_part:
                    parts[index] = part
                part_id = part.get("part_id")
                if not isinstance(part_id, str) or not part_id:
                    raise ValueError(f"image graph part {index} is missing a valid part_id")
                raw_path = part.get("content")
                if raw_path is None:
                    metadata = part.get("metadata", {})
                    raw_path = metadata.get("crop_path") if isinstance(metadata, Mapping) else None
                if not isinstance(raw_path, (str, Path)) or not str(raw_path):
                    raise ValueError(f"image graph part {part_id!r} has no saved crop path")
                crop_path = Path(raw_path)
                if not crop_path.is_file():
                    raise ValueError(f"image crop for part {part_id!r} does not exist: {crop_path}")
                digest = sha256()
                with crop_path.open("rb") as crop_file:
                    for chunk in iter(lambda: crop_file.read(1024 * 1024), b""):
                        digest.update(chunk)
                part_records.append((graph, part, str(crop_path), digest.hexdigest()))

        if not part_records:
            return graphs

        raw_client = self.image_embedder
        if raw_client is None:
            raw_client = EmbeddingGemma2ImageClient(self.image_embedding_model)
            self.image_embedder = raw_client
        client = as_image_embedding_client(raw_client)
        configured_model_id = getattr(client, "model_id", None)
        model_id = (
            configured_model_id
            if isinstance(configured_model_id, str) and configured_model_id.strip()
            else f"{type(client).__module__}.{type(client).__qualname__}"
        )
        config = {
            "model": model_id,
            "dimension": embedding_dim,
            "source_dimension": source_embedding_dim,
            "normalized": True,
            "precision_policy": getattr(client, "precision_policy", "client-managed"),
            "input": "image-part-v1",
            "projection_method": (
                "GaussianRandomProjection"
                if embedding_dim != source_embedding_dim
                else None
            ),
            "projection_random_seed": (
                self.image_embedding_projection_seed
                if embedding_dim != source_embedding_dim
                else None
            ),
        }
        config_hash = sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()

        pending: list[tuple[nx.MultiDiGraph, dict[str, Any], str, str]] = []
        for graph, part, crop_path, crop_sha256 in part_records:
            if skip_matching:
                vector = part.get("embedding")
                metadata = part.get("embedding_metadata")
                if (
                    isinstance(vector, list)
                    and len(vector) == embedding_dim
                    and isinstance(metadata, Mapping)
                    and metadata.get("crop_sha256") == crop_sha256
                    and metadata.get("config_hash") == config_hash
                    and all(isinstance(value, (int, float)) and math.isfinite(value) for value in vector)
                ):
                    continue
            pending.append((graph, part, crop_path, crop_sha256))

        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            vectors = client.embed_images(
                [crop_path for _graph, _part, crop_path, _digest in batch],
                batch_size=batch_size,
                embedding_dim=source_embedding_dim,
            )
            if hasattr(vectors, "tolist"):
                vectors = vectors.tolist()
            if not isinstance(vectors, (list, tuple)) or len(vectors) != len(batch):
                raise ValueError("image_embedder returned a different number of vectors than image parts")
            validated: list[list[float]] = []
            for index, raw_vector in enumerate(vectors):
                if hasattr(raw_vector, "tolist"):
                    raw_vector = raw_vector.tolist()
                if not isinstance(raw_vector, (list, tuple)):
                    raise ValueError(f"image_embedder returned an invalid vector for part '{batch[index][1]['part_id']}'")
                try:
                    vector = [float(value) for value in raw_vector]
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"image_embedder returned an invalid vector for part '{batch[index][1]['part_id']}'"
                    ) from exc
                if len(vector) != source_embedding_dim or not all(math.isfinite(value) for value in vector):
                    raise ValueError(
                        f"image_embedder vectors must contain {source_embedding_dim} finite values; "
                        f"part '{batch[index][1]['part_id']}' returned {len(vector)}"
                    )
                norm = math.hypot(*vector)
                if norm == 0:
                    raise ValueError(f"image_embedder returned a zero vector for part '{batch[index][1]['part_id']}'")
                validated.append([value / norm for value in vector])

            if embedding_dim != source_embedding_dim:
                validated = project_embedding_vectors(
                    validated,
                    output_dim=embedding_dim,
                    random_seed=self.image_embedding_projection_seed,
                )

            for (graph, part, _crop_path, crop_sha256), vector in zip(batch, validated):
                part["embedding"] = vector
                part["embedding_metadata"] = {
                    **config,
                    "config_hash": config_hash,
                    "crop_sha256": crop_sha256,
                }
        return graphs
