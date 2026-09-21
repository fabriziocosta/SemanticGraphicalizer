"""Scikit-learn compatible public transformer."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

import networkx as nx
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .config import OntologyConfig, PromptConfig, load_ontology, load_prompts
from .model import ModelClient, OpenAIModelClient
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

    def display(
        self,
        graph_or_trace: nx.Graph | DocumentTrace,
        *,
        mode: str = "dynamic",
        **kwargs: Any,
    ) -> Any:
        """Return a dynamic D3, static SVG, or indented text visualization."""

        return display_graph(graph_or_trace, mode=mode, **kwargs)
