"""Scikit-learn compatible public transformer."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

import networkx as nx
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .config import OntologyConfig, PromptConfig, load_ontology, load_prompts
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
from .types import DocumentTrace
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
        del y
        documents = self._validate_input(X)
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
        self.n_documents_in_fit_ = len(documents)
        return self

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
        traces = self.transform_with_trace(X)
        return [trace.graph for trace in traces]

    def transform_with_trace(self, X: Iterable[str]) -> list[DocumentTrace]:
        check_is_fitted(self, ["ontology_", "prompts_", "pipeline_"])
        documents = self._validate_input(X)
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
        values: DocumentTrace | nx.Graph | Iterable[DocumentTrace | nx.Graph],
    ) -> list[DocumentTrace | nx.Graph]:
        if isinstance(values, (DocumentTrace, nx.Graph)):
            return [values]
        if isinstance(values, (str, bytes)):
            raise TypeError("values must contain DocumentTrace or NetworkX graph objects")
        try:
            targets = list(values)
        except TypeError as exc:
            raise TypeError("values must be a DocumentTrace, NetworkX graph, or iterable of either") from exc
        if not all(isinstance(value, (DocumentTrace, nx.Graph)) for value in targets):
            raise TypeError("values must contain only DocumentTrace or NetworkX graph objects")
        return targets

    def compute_embeddings(
        self,
        values: DocumentTrace | nx.Graph | Iterable[DocumentTrace | nx.Graph],
        *,
        embedding_attribute: str = "embedding",
        node_text_fn: Callable[[Any, Mapping[str, Any]], str] | None = None,
        batch_size: int = 128,
    ) -> list[DocumentTrace | nx.Graph]:
        """Compute and attach one text embedding to every graph node.

        ``values`` may contain traces or NetworkX graphs. Inputs are mutated
        in place and returned as a list. Vectors are stored on each node under
        ``embedding_attribute``; the default is ``graph.nodes[node_id]["embedding"]``.
        """

        if not isinstance(embedding_attribute, str) or not embedding_attribute.strip():
            raise ValueError("embedding_attribute must be a non-empty string")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        targets = self._embedding_targets(values)
        text_builder = node_text_fn or self._default_node_embedding_text
        records: list[tuple[nx.Graph, Any, str]] = []
        for value in targets:
            graph = value.graph if isinstance(value, DocumentTrace) else value
            for node_id, data in graph.nodes(data=True):
                text = text_builder(node_id, data)
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"node_text_fn returned empty text for node '{node_id}'")
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

        pending: list[tuple[nx.Graph, Any, list[float]]] = []
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            vectors = list(client.embed([text for _graph, _node_id, text in batch]))
            if len(vectors) != len(batch):
                raise ValueError("embedder returned a different number of vectors than node texts")
            for (graph, node_id, _text), vector in zip(batch, vectors):
                if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence):
                    raise ValueError(f"embedder returned an invalid vector for node '{node_id}'")
                try:
                    values_as_float = [float(component) for component in vector]
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"embedder returned an invalid vector for node '{node_id}'") from exc
                if not values_as_float:
                    raise ValueError(f"embedder returned an empty vector for node '{node_id}'")
                pending.append((graph, node_id, values_as_float))
        for graph, node_id, vector in pending:
            graph.nodes[node_id][embedding_attribute] = vector
        return targets

    def display(
        self,
        graph_or_trace: nx.Graph | DocumentTrace,
        *,
        mode: str = "dynamic",
        **kwargs: Any,
    ) -> Any:
        """Return a dynamic D3, static SVG, or indented text visualization."""

        return display_graph(graph_or_trace, mode=mode, **kwargs)
