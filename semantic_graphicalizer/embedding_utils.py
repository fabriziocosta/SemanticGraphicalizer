"""Shared utilities for working with embedding vectors."""

from __future__ import annotations

from collections.abc import Sequence


def project_embedding_vectors(
    vectors: Sequence[Sequence[float]],
    *,
    output_dim: int,
    random_seed: int = 17,
) -> list[list[float]]:
    """Project a batch through a reproducible Gaussian random map.

    The same seed and source dimension produce the same projection matrix, so
    independently embedded batches remain in a comparable vector space.
    """

    import numpy as np
    from sklearn.random_projection import GaussianRandomProjection

    matrix = np.asarray(vectors, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("vectors must be a non-empty two-dimensional array")
    if not np.isfinite(matrix).all():
        raise ValueError("vectors must contain only finite values")
    if isinstance(output_dim, bool) or not isinstance(output_dim, int) or output_dim < 1:
        raise ValueError("output_dim must be a positive integer")
    if output_dim >= matrix.shape[1]:
        raise ValueError("output_dim must be smaller than the source dimension")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise ValueError("random_seed must be a non-negative integer")

    projected = GaussianRandomProjection(
        n_components=output_dim,
        random_state=random_seed,
    ).fit_transform(matrix)
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.isfinite(norms).all():
        raise ValueError("random projection produced an invalid zero or non-finite vector")
    return (projected / norms).tolist()
