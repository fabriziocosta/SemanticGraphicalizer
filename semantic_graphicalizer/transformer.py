"""Scikit-learn compatible public transformer."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import networkx as nx
from sklearn.base import BaseEstimator, TransformerMixin

from .config import OntologyConfig, PromptConfig, load_ontology, load_prompts
from .abstract_graph import InterpretationMode, ParallelEdgePolicy, semantic_graph_to_abstract_graph
from .model import (
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    EmbeddingClient,
    ModelClient,
    OpenAIEmbeddingClient,
    OpenAIModelClient,
    as_embedding_client,
)
from .pipeline import (
    ConservativeEntityResolver,
    EntityResolver,
    ParagraphWindowSegmenter,
    Segmenter,
    SemanticPipeline,
)
from .visualization import display_graph


class SemanticGraphicalizer(BaseEstimator, TransformerMixin):
    """Compile complete documents into ontology-grounded NetworkX graphs."""

    def __init__(
        self,
        ontology: str | Path | Mapping[str, Any] | OntologyConfig,
        prompts: str | Path | Mapping[str, Any] | PromptConfig,
        model: ModelClient | None = None,
        *,
        embedder: EmbeddingClient | Callable[[Sequence[str]], Sequence[Sequence[float]]] | None = None,
        embedding_model: str = DEFAULT_OPENAI_EMBEDDING_MODEL,
        max_chunk_chars: int = 4000,
        chunk_overlap: int = 0,
        max_retries: int = 2,
        retry_backoff: float = 0.25,
        document_id_fn: Callable[[str, int], str] | None = None,
        segmenter: Segmenter | None = None,
        entity_resolver: EntityResolver | None = None,
        verbose: bool = True,
    ) -> None:
        if document_id_fn is not None and not callable(document_id_fn):
            raise TypeError("document_id_fn must be callable")
        self.ontology = ontology
        self.prompts = prompts
        self.model = model
        self.embedder = embedder
        self.embedding_model = embedding_model
        self.max_chunk_chars = max_chunk_chars
        self.chunk_overlap = chunk_overlap
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.document_id_fn = document_id_fn
        self.segmenter = segmenter
        self.entity_resolver = entity_resolver
        self.verbose = verbose

    def fit(self, X: Iterable[str], y: Any = None) -> "SemanticGraphicalizer":
        # This transformer has no corpus-dependent learned state. ``fit`` is
        # retained for sklearn compatibility and prepares the configured
        # pipeline; document validation happens when documents are transformed.
        del X, y
        self._initialize_pipeline()
        return self

    def _initialize_pipeline(self) -> None:
        self.ontology_ = load_ontology(self.ontology)
        self.prompts_ = load_prompts(self.prompts)
        segmenter = self.segmenter or ParagraphWindowSegmenter(self.max_chunk_chars, self.chunk_overlap)
        resolver = self.entity_resolver or ConservativeEntityResolver()
        model = self.model if self.model is not None else OpenAIModelClient()
        self.pipeline_ = SemanticPipeline(
            model,
            self.ontology_,
            self.prompts_,
            segmenter,
            resolver,
            self.verbose,
            self.max_retries,
            self.retry_backoff,
        )
        if self.verbose:
            print(
                f"[SemanticGraphicalizer] ready: model={type(model).__name__}, "
                f"ontology={self.ontology_.name}, domain={self.prompts_.domain}"
            )

    def _ensure_pipeline(self) -> None:
        if not all(hasattr(self, name) for name in ("ontology_", "prompts_", "pipeline_")):
            self._initialize_pipeline()

    @staticmethod
    def _validate_input(X: Iterable[str]) -> list[str]:
        if isinstance(X, (str, bytes)):
            raise ValueError("X must be an iterable of complete document strings, not one string")
        try:
            documents = list(X)
        except TypeError as exc:
            raise ValueError("X must be an iterable of document strings") from exc
        if not all(isinstance(document, str) and document.strip() for document in documents):
            raise ValueError("every document must be a non-empty string")
        return documents

    def fit_transform(self, X: Iterable[str], y: Any = None, **fit_params: Any) -> list[nx.MultiDiGraph]:
        if fit_params:
            raise TypeError("fit_params are not supported by SemanticGraphicalizer")
        documents = self._validate_input(X)
        self.fit(documents, y)
        return self.transform(documents)

    def _document_ids(self, documents: list[str]) -> list[str]:
        occurrences: dict[str, int] = {}
        document_ids: list[str] = []
        for index, document in enumerate(documents):
            if self.document_id_fn is None:
                digest = sha256(document.encode("utf-8")).hexdigest()[:12]
                base_id = f"document-{digest}"
            else:
                base_id = self.document_id_fn(document, index)
                if not isinstance(base_id, str) or not base_id.strip():
                    raise ValueError("document_id_fn must return a non-empty string")
            occurrence = occurrences.get(base_id, 0)
            occurrences[base_id] = occurrence + 1
            document_ids.append(base_id if occurrence == 0 else f"{base_id}-{occurrence}")
        return document_ids

    def transform(self, X: Iterable[str]) -> list[nx.MultiDiGraph]:
        documents = self._validate_input(X)
        self._ensure_pipeline()
        document_ids = self._document_ids(documents)
        return [
            self.pipeline_.process(document_id, document)
            for document_id, document in zip(document_ids, documents)
        ]

    @staticmethod
    def _default_node_embedding_text(node_id: Any, data: Mapping[str, Any]) -> str:
        """Build embedding input from ontology labels and available evidence."""

        entity_type = data.get("type") or data.get("label") or node_id
        relation = data.get("relation")
        primary = f"{entity_type} : {relation}" if relation is not None else str(entity_type)
        pieces = [primary]
        attributes = data.get("attributes")
        attributes = attributes if isinstance(attributes, Mapping) else {}

        def add(value: Any) -> None:
            if isinstance(value, str) and value.strip() and value not in pieces:
                pieces.append(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    add(item)

        add(data.get("mentions", attributes.get("mentions")))
        add(data.get("source_text", attributes.get("source_text")))
        if len(pieces) == 1:
            add(str(node_id))
        return " | ".join(pieces)

    @staticmethod
    def _embedding_targets(
        values: nx.Graph | Iterable[nx.Graph],
    ) -> list[nx.Graph]:
        if isinstance(values, nx.Graph):
            return [values]
        if isinstance(values, (str, bytes)):
            raise TypeError("values must contain NetworkX graph objects")
        try:
            targets = list(values)
        except TypeError as exc:
            raise TypeError("values must be a NetworkX graph or iterable of graphs") from exc
        if not all(isinstance(value, nx.Graph) for value in targets):
            raise TypeError("values must contain only NetworkX graph objects")
        return targets

    def compute_embeddings(
        self,
        values: nx.Graph | Iterable[nx.Graph],
        *,
        embedding_attribute: str = "embedding",
        node_text_fn: Callable[[Any, Mapping[str, Any]], str] | None = None,
        batch_size: int = 128,
        skip_matching: bool = False,
    ) -> list[nx.Graph]:
        """Compute and attach one text embedding to every graph node.

        Inputs are mutated in place and returned as a list. Vectors are stored
        on each node under
        ``embedding_attribute``; the default is ``graph.nodes[node_id]["embedding"]``.
        """

        if not isinstance(embedding_attribute, str) or not embedding_attribute.strip():
            raise ValueError("embedding_attribute must be a non-empty string")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(skip_matching, bool):
            raise TypeError("skip_matching must be a bool")
        targets = self._embedding_targets(values)
        text_builder = node_text_fn or self._default_node_embedding_text
        provider = self.embedder
        model_id = (
            self.embedding_model
            if provider is None
            else getattr(provider, "model", f"{type(provider).__module__}.{type(provider).__qualname__}")
        )
        provider_function = getattr(provider, "function", None) if provider is not None else None
        provider_request_options = getattr(provider, "request_options", {}) if provider is not None else {}
        provider_settings = {
            "provider": f"{type(provider).__module__}.{type(provider).__qualname__}" if provider is not None else "openai",
            "model": str(model_id),
            "request_options": dict(provider_request_options) if isinstance(provider_request_options, Mapping) else {},
            "cache_key": getattr(provider, "cache_key", None) if provider is not None else None,
        }
        if callable(provider_function):
            provider_settings["function"] = (
                f"{getattr(provider_function, '__module__', '')}."
                f"{getattr(provider_function, '__qualname__', type(provider_function).__qualname__)}"
            )
        builder_id = "default-node-text-v1"
        if node_text_fn is not None:
            builder_id = f"{getattr(node_text_fn, '__module__', '')}.{getattr(node_text_fn, '__qualname__', type(node_text_fn).__qualname__)}"
        config_digest = sha256(
            json.dumps(
                {
                    "provider": provider_settings,
                    "text_builder": builder_id,
                    "embedding_attribute": embedding_attribute,
                },
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        records: list[tuple[nx.Graph, Any, str]] = []
        for value in targets:
            graph = value
            for node_id, data in graph.nodes(data=True):
                text = text_builder(node_id, data)
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"node_text_fn returned empty text for node '{node_id}'")
                if skip_matching and embedding_attribute in data:
                    metadata = data.get("embedding_metadata")
                    text_digest = sha256(text.encode("utf-8")).hexdigest()
                    try:
                        existing = [float(component) for component in data[embedding_attribute]]
                    except (TypeError, ValueError):
                        existing = []
                    if (
                        isinstance(metadata, Mapping)
                        and metadata.get("config_hash") == config_digest
                        and metadata.get("text_sha256") == text_digest
                        and metadata.get("dimension") == len(existing)
                        and bool(existing)
                    ):
                        continue
                records.append((graph, node_id, text))
        if not records:
            return targets

        client = getattr(self, "embedding_client_", None)
        if client is None:
            raw_client = (
                self.embedder
                if self.embedder is not None
                else OpenAIEmbeddingClient(model=self.embedding_model)
            )
            client = as_embedding_client(raw_client)
            self.embedding_client_ = client

        pending: list[tuple[nx.Graph, Any, list[float], str]] = []
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            vectors = list(client.embed([text for _graph, _node_id, text in batch]))
            if len(vectors) != len(batch):
                raise ValueError("embedder returned a different number of vectors than node texts")
            for (graph, node_id, text), vector in zip(batch, vectors):
                if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence):
                    raise ValueError(f"embedder returned an invalid vector for node '{node_id}'")
                try:
                    values_as_float = [float(component) for component in vector]
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"embedder returned an invalid vector for node '{node_id}'") from exc
                if not values_as_float:
                    raise ValueError(f"embedder returned an empty vector for node '{node_id}'")
                pending.append((graph, node_id, values_as_float, text))
        for graph, node_id, vector, text in pending:
            graph.nodes[node_id][embedding_attribute] = vector
            graph.nodes[node_id]["embedding_metadata"] = {
                "model": str(model_id),
                "config_hash": config_digest,
                "text_sha256": sha256(text.encode("utf-8")).hexdigest(),
                "dimension": len(vector),
            }
        return targets

    def to_abstract_graph(
        self,
        graph: nx.MultiDiGraph,
        *,
        embed_nodes: bool = False,
        embedding_key: str = "embedding",
        chunk_key: str = "chunk_id",
        parallel_edge_policy: ParallelEdgePolicy = "combine",
        nbits: int = 14,
        preserve_direction: bool = True,
        interpretation_mode: InterpretationMode = "per_entity",
        node_text_fn: Callable[[Any, Mapping[str, Any]], str] | None = None,
        batch_size: int = 128,
    ) -> Any:
        """Convert a semantic graph to an optional AbstractGraph.

        Set ``embed_nodes=True`` to compute missing or stale node text embeddings
        before conversion. Matching embeddings are reused. By default, each
        semantic node maps to a distinct interpretation node; set
        ``interpretation_mode="by_chunk_and_type"`` to group them as before.
        ``parallel_edge_policy`` can combine parallel edges, keep only the
        first, or raise an error.
        """

        if not isinstance(graph, nx.MultiDiGraph):
            raise TypeError("graph must be a NetworkX MultiDiGraph")
        if not isinstance(embed_nodes, bool):
            raise TypeError("embed_nodes must be a bool")
        if parallel_edge_policy not in {"combine", "first", "error"}:
            raise ValueError("parallel_edge_policy must be 'combine', 'first', or 'error'")
        if not isinstance(interpretation_mode, str) or interpretation_mode not in {
            "per_entity",
            "by_chunk_and_type",
        }:
            raise ValueError("interpretation_mode must be 'per_entity' or 'by_chunk_and_type'")
        if embed_nodes:
            self.compute_embeddings(
                graph,
                embedding_attribute=embedding_key,
                node_text_fn=node_text_fn,
                batch_size=batch_size,
                skip_matching=True,
            )
        return semantic_graph_to_abstract_graph(
            graph,
            embedding_key=embedding_key,
            chunk_key=chunk_key,
            parallel_edge_policy=parallel_edge_policy,
            nbits=nbits,
            preserve_direction=preserve_direction,
            interpretation_mode=interpretation_mode,
        )

    def to_abstract_graphs(
        self,
        graphs: Iterable[nx.MultiDiGraph],
        *,
        embed_nodes: bool = False,
        embedding_key: str = "embedding",
        chunk_key: str = "chunk_id",
        parallel_edge_policy: ParallelEdgePolicy = "combine",
        nbits: int = 14,
        preserve_direction: bool = True,
        interpretation_mode: InterpretationMode = "per_entity",
        node_text_fn: Callable[[Any, Mapping[str, Any]], str] | None = None,
        batch_size: int = 128,
    ) -> list[Any]:
        """Convert multiple semantic graphs to AbstractGraphs.

        When ``embed_nodes=True``, node embeddings are computed across all
        inputs in batches before conversion. ``interpretation_mode`` is applied
        to every result. Results preserve input order.
        """

        if isinstance(graphs, (str, bytes)):
            raise TypeError("graphs must contain NetworkX MultiDiGraph objects")
        try:
            values = list(graphs)
        except TypeError as exc:
            raise TypeError("graphs must be an iterable of NetworkX MultiDiGraph objects") from exc

        if not all(isinstance(graph, nx.MultiDiGraph) for graph in values):
            raise TypeError("graphs must contain only NetworkX MultiDiGraph objects")

        if not isinstance(embed_nodes, bool):
            raise TypeError("embed_nodes must be a bool")
        if not isinstance(embedding_key, str) or not embedding_key:
            raise ValueError("embedding_key must be a non-empty string")
        if not isinstance(chunk_key, str) or not chunk_key:
            raise ValueError("chunk_key must be a non-empty string")
        if parallel_edge_policy not in {"combine", "first", "error"}:
            raise ValueError("parallel_edge_policy must be 'combine', 'first', or 'error'")
        if isinstance(nbits, bool) or not isinstance(nbits, int) or nbits < 1:
            raise ValueError("nbits must be a positive integer")
        if not isinstance(preserve_direction, bool):
            raise TypeError("preserve_direction must be a bool")
        if not isinstance(interpretation_mode, str) or interpretation_mode not in {
            "per_entity",
            "by_chunk_and_type",
        }:
            raise ValueError("interpretation_mode must be 'per_entity' or 'by_chunk_and_type'")
        if embed_nodes and values:
            self.compute_embeddings(
                values,
                embedding_attribute=embedding_key,
                node_text_fn=node_text_fn,
                batch_size=batch_size,
                skip_matching=True,
            )

        return [
            self.to_abstract_graph(
                value,
                embed_nodes=False,
                embedding_key=embedding_key,
                chunk_key=chunk_key,
                parallel_edge_policy=parallel_edge_policy,
                nbits=nbits,
                preserve_direction=preserve_direction,
                interpretation_mode=interpretation_mode,
            )
            for value in values
        ]

    def transform_abstract(
        self,
        X: Iterable[str],
        *,
        embed_nodes: bool = False,
        embedding_key: str = "embedding",
        chunk_key: str = "chunk_id",
        parallel_edge_policy: ParallelEdgePolicy = "combine",
        nbits: int = 14,
        preserve_direction: bool = True,
        interpretation_mode: InterpretationMode = "per_entity",
        node_text_fn: Callable[[Any, Mapping[str, Any]], str] | None = None,
        batch_size: int = 128,
    ) -> list[Any]:
        """Transform documents directly into AbstractGraph objects.

        ``interpretation_mode`` selects per-entity mappings or the legacy
        chunk-and-type grouping for each returned graph.
        """

        graphs = self.transform(X)
        return [
            self.to_abstract_graph(
                graph,
                embed_nodes=embed_nodes,
                embedding_key=embedding_key,
                chunk_key=chunk_key,
                parallel_edge_policy=parallel_edge_policy,
                nbits=nbits,
                preserve_direction=preserve_direction,
                interpretation_mode=interpretation_mode,
                node_text_fn=node_text_fn,
                batch_size=batch_size,
            )
            for graph in graphs
        ]

    def display(
        self,
        graph: nx.Graph,
        *,
        mode: str = "dynamic",
        **kwargs: Any,
    ) -> Any:
        """Return a dynamic D3, static SVG, or indented text visualization."""

        return display_graph(graph, mode=mode, **kwargs)
