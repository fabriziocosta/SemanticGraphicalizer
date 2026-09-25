"""Reusable workflow for the Aesop AbstractGraph clustering notebook."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from semantic_graphicalizer import (
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    DEFAULT_OPENAI_MODEL,
    OpenAIEmbeddingClient,
    SemanticGraphicalizer,
    load_aesop_fables,
    vectorize_abstract_graphs,
)
from semantic_graphicalizer.model import as_embedding_client


def text_chunks(text: str, max_chars: int = 6000) -> list[str]:
    """Split text near whitespace, keeping chunks within ``max_chars``."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 1:
        raise ValueError("max_chars must be a positive integer")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + 1, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


def _corpus_metadata(stories: Sequence[str]) -> tuple[list[dict[str, Any]], list[str], str]:
    metadata = []
    source_hashes = []
    for index, story in enumerate(stories):
        story_hash = hashlib.sha256(story.encode("utf-8")).hexdigest()
        source_hashes.append(story_hash)
        metadata.append({
            "tale_id": f"aesop-{index:04d}-{story_hash[:12]}",
            "source_order": index,
            "title": story.splitlines()[0].strip(),
            "characters": len(story),
            "words": len(story.split()),
            "source_sha256": story_hash,
        })
    corpus_hash = hashlib.sha256("".join(source_hashes).encode("utf-8")).hexdigest()
    return metadata, source_hashes, corpus_hash


def _load_or_build_graphs(
    stories: Sequence[str],
    metadata: Sequence[dict[str, Any]],
    graphicalizer: SemanticGraphicalizer,
    cache_dir: Path,
    graph_config_hash: str,
) -> list[Any]:
    graphs = []
    total = len(stories)
    for index, (story, row) in enumerate(zip(stories, metadata), start=1):
        print(f"\nStory {index}/{total}: {row['title']}")
        print("-" * 80)
        checkpoint = cache_dir / f"{row['tale_id']}-{graph_config_hash[:8]}.graph.pkl"
        if checkpoint.exists():
            with checkpoint.open("rb") as handle:
                graph = pickle.load(handle)
        else:
            graph = graphicalizer.transform([story])[0]
            with checkpoint.open("wb") as handle:
                pickle.dump(graph, handle)
        graphs.append(graph)
    return graphs


def _load_or_build_text_vectors(
    stories: Sequence[str],
    source_hashes: Sequence[str],
    embedding_client: Any,
    cache_path: Path,
    embedding_model: str,
    *,
    max_chars: int = 6000,
    batch_size: int = 128,
) -> np.ndarray:
    pooling = "mean"
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            cached = pickle.load(handle)
    else:
        cached = None
    if (
        cached
        and cached.get("source_hashes") == list(source_hashes)
        and cached.get("embedding_model") == embedding_model
        and cached.get("chunk_chars") == max_chars
        and cached.get("pooling") == pooling
    ):
        return np.vstack(cached["vectors"])

    chunks_by_story = [text_chunks(story, max_chars=max_chars) for story in stories]
    flat_chunks = [chunk for chunks in chunks_by_story for chunk in chunks]
    chunk_vectors = []
    for start in range(0, len(flat_chunks), batch_size):
        response = embedding_client.embed(flat_chunks[start:start + batch_size])
        chunk_vectors.extend(np.asarray(vector, dtype=float) for vector in response)

    story_vectors = []
    offset = 0
    for chunks in chunks_by_story:
        current = chunk_vectors[offset:offset + len(chunks)]
        if not current:
            raise ValueError("cannot build a text vector for an empty story")
        story_vectors.append(np.mean(current, axis=0))
        offset += len(chunks)

    with cache_path.open("wb") as handle:
        pickle.dump({
            "source_hashes": list(source_hashes),
            "embedding_model": embedding_model,
            "chunk_chars": max_chars,
            "pooling": pooling,
            "vectors": story_vectors,
        }, handle)
    return np.vstack(story_vectors)


def _cluster_representations(
    graph_matrix: Any,
    text_matrix: np.ndarray,
    random_seed: int,
    max_cluster_size: int = 5,
) -> dict[str, Any]:
    n_tales = graph_matrix.shape[0]
    if n_tales < 3:
        raise ValueError("Clustering inspection needs at least three tales")
    if isinstance(max_cluster_size, bool) or not isinstance(max_cluster_size, int) or max_cluster_size < 1:
        raise ValueError("max_cluster_size must be a positive integer")

    def hierarchical_labels(features: np.ndarray) -> np.ndarray:
        tree = AgglomerativeClustering(n_clusters=1, linkage="ward").fit(features)
        n_samples = features.shape[0]
        children = tree.children_
        subtree_sizes = {node: 1 for node in range(n_samples)}
        for merge_index, (left, right) in enumerate(children):
            parent = n_samples + merge_index
            subtree_sizes[parent] = subtree_sizes[left] + subtree_sizes[right]

        def leaves(node: int) -> list[int]:
            if node < n_samples:
                return [node]
            left, right = children[node - n_samples]
            return leaves(int(left)) + leaves(int(right))

        groups: list[list[int]] = []

        def split_oversized(node: int) -> None:
            if subtree_sizes[node] <= max_cluster_size:
                groups.append(leaves(node))
                return
            left, right = children[node - n_samples]
            split_oversized(int(left))
            split_oversized(int(right))

        split_oversized(2 * n_samples - 2)
        labels = np.empty(n_samples, dtype=int)
        for label, members in enumerate(groups):
            labels[members] = label
        return labels

    scaled_graph = StandardScaler(with_mean=False).fit_transform(graph_matrix)
    svd_components = max(2, min(50, n_tales - 1, scaled_graph.shape[1] - 1))
    graph_features = TruncatedSVD(
        n_components=svd_components, random_state=random_seed
    ).fit_transform(scaled_graph)
    text_features = StandardScaler().fit_transform(text_matrix)

    cluster_results: dict[tuple[str, str, int], np.ndarray] = {}
    metric_rows = []
    for representation, features in (("AbstractGraph", graph_features), ("Direct text", text_features)):
        labels = hierarchical_labels(features)
        cluster_results[(representation, "hierarchical_ward", max_cluster_size)] = labels
        cluster_count = len(np.unique(labels))
        if 1 < cluster_count < n_tales:
            silhouette = float(silhouette_score(features, labels))
            calinski_harabasz = float(calinski_harabasz_score(features, labels))
            davies_bouldin = float(davies_bouldin_score(features, labels))
        else:
            silhouette = calinski_harabasz = davies_bouldin = None
        metric_rows.append({
            "representation": representation,
            "algorithm": "hierarchical_ward",
            "max_cluster_size": max_cluster_size,
            "cluster_count": cluster_count,
            "largest_cluster": max(np.bincount(labels)),
            "silhouette": silhouette,
            "calinski_harabasz": calinski_harabasz,
            "davies_bouldin": davies_bouldin,
        })

    graph_labels = cluster_results[("AbstractGraph", "hierarchical_ward", max_cluster_size)]
    text_labels = cluster_results[("Direct text", "hierarchical_ward", max_cluster_size)]
    agreement_rows = [{
        "algorithm": "hierarchical_ward",
        "max_cluster_size": max_cluster_size,
        "ari": float(adjusted_rand_score(graph_labels, text_labels)),
        "nmi": float(normalized_mutual_info_score(graph_labels, text_labels)),
    }]

    return {
        "graph_features": graph_features,
        "text_features": text_features,
        "cluster_results": cluster_results,
        "metric_rows": metric_rows,
        "agreement_rows": agreement_rows,
        "max_cluster_size": max_cluster_size,
        "svd_components": svd_components,
    }


def run_experiment(
    root: str | Path,
    *,
    smoke_test: bool = True,
    smoke_limit: int = 5,
    embed_nodes: bool = True,
    embedding_model: str = DEFAULT_OPENAI_EMBEDDING_MODEL,
    random_seed: int = 17,
    nbits: int = 14,
    max_cluster_size: int = 5,
) -> dict[str, Any]:
    """Load/cache Aesop graphs and vectors, cluster them, and save a manifest."""

    root = Path(root)
    cache_dir = root / "data" / "processed" / "aesop_abstractgraph"
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = root / "data" / "raw"
    stories = load_aesop_fables(
        limit=smoke_limit if smoke_test else None,
        cache_dir=raw_dir,
    )
    metadata, source_hashes, corpus_hash = _corpus_metadata(stories)

    config_paths = {
        "ontology": root / "configs" / "ontologies" / "aesop.yaml",
        "prompts": root / "configs" / "prompts" / "aesop.yaml",
    }
    config_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in config_paths.items()
    }
    graph_config_hash = hashlib.sha256(json.dumps({
        "extraction_model": DEFAULT_OPENAI_MODEL,
        "config_sha256": config_hashes,
    }, sort_keys=True).encode("utf-8")).hexdigest()

    graphicalizer = SemanticGraphicalizer(
        ontology=config_paths["ontology"],
        prompts=config_paths["prompts"],
        embedding_model=embedding_model,
    )
    graphs = _load_or_build_graphs(
        stories, metadata, graphicalizer, cache_dir, graph_config_hash
    )
    abstract_settings = {
        "embedding_key": "embedding",
        "interpretation_mode": "per_entity",
        "chunk_key": "chunk_id",
        "parallel_edge_policy": "combine",
        "preserve_direction": True,
        "nbits": nbits,
    }
    abstract_graphs = graphicalizer.to_abstract_graphs(
        graphs, embed_nodes=embed_nodes, **abstract_settings
    )
    graph_matrix = vectorize_abstract_graphs(abstract_graphs, nbits=nbits)

    # Persist any node embeddings computed during AbstractGraph conversion.
    for graph, row in zip(graphs, metadata):
        checkpoint = cache_dir / f"{row['tale_id']}-{graph_config_hash[:8]}.graph.pkl"
        with checkpoint.open("wb") as handle:
            pickle.dump(graph, handle)

    raw_embedder = graphicalizer.embedder or OpenAIEmbeddingClient(model=embedding_model)
    embedding_client = as_embedding_client(raw_embedder)
    text_matrix = _load_or_build_text_vectors(
        stories,
        source_hashes,
        embedding_client,
        cache_dir / "direct_text_vectors.pkl",
        embedding_model,
    )
    clustering = _cluster_representations(
        graph_matrix,
        text_matrix,
        random_seed,
        max_cluster_size=max_cluster_size,
    )

    settings = {
        "smoke_test": smoke_test,
        "smoke_limit": smoke_limit if smoke_test else None,
        "extraction_model": DEFAULT_OPENAI_MODEL,
        "embedding_model": embedding_model,
        "embed_nodes": embed_nodes,
        "abstractgraph": abstract_settings,
        "graph_vectorizer": {
            "transformer": "AbstractGraphTransformer",
            "interpretation": "per_entity",
            "return_dense": False,
        },
        "text_baseline": {"chunk_chars": 6000, "pooling": "mean"},
        "random_seed": random_seed,
        "max_cluster_size": max_cluster_size,
    }
    run_hash = hashlib.sha256(json.dumps({
        "settings": settings,
        "config_sha256": config_hashes,
    }, sort_keys=True).encode("utf-8")).hexdigest()
    node_embedding_dimensions = sorted({
        len(data["embedding"])
        for graph in graphs
        for _node, data in graph.nodes(data=True)
        if "embedding" in data
    })
    manifest = {
        "schema_version": 1,
        "corpus": {
            "sha256": corpus_hash,
            "tale_count": len(stories),
            "tales": [{
                key: row[key]
                for key in ("tale_id", "source_order", "title", "characters", "words", "source_sha256")
            } for row in metadata],
            "config_sha256": config_hashes,
        },
        "settings": settings,
        "run_sha256": run_hash,
        "dimensions": {
            "node_embeddings": node_embedding_dimensions,
            "graph_vectors": graph_matrix.shape[1],
            "text_vectors": int(text_matrix.shape[1]),
            "graph_svd_components": clustering["svd_components"],
        },
        "metrics": [{
            key: (float(value) if isinstance(value, np.generic) else value)
            for key, value in row.items()
        } for row in clustering["metric_rows"]],
        "graph_text_agreement": clustering["agreement_rows"],
        "cluster_assignments": {
            f"{representation}|{algorithm}|max_size={cluster_size}": [int(label) for label in labels]
            for (representation, algorithm, cluster_size), labels in clustering["cluster_results"].items()
        },
    }
    run_mode = "smoke" if smoke_test else "full"
    manifest_path = cache_dir / f"{run_mode}-{corpus_hash[:12]}-{run_hash[:8]}-results.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "stories": stories,
        "metadata": metadata,
        "graphs": graphs,
        "abstract_graphs": abstract_graphs,
        "graph_matrix": graph_matrix,
        "text_matrix": text_matrix,
        "manifest_path": manifest_path,
        **clustering,
    }
