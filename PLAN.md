# Plan: Aesop AbstractGraph Embedding and Clustering Experiment

## Objective

Process the complete cached Aesop corpus, construct a recursive semantic graph
for each tale, optionally embed the text associated with each semantic node
during AbstractGraph conversion, and cluster one AbstractGraph-derived vector
per tale. Compare those clusters with a direct whole-tale text embedding
baseline.

The experiment keeps titles and moral headings as evaluation metadata. They are
not included in either the node embedding inputs or the direct-text baseline.

## Deliverables

- `notebooks/aesop_graph_embeddings_clustering.ipynb`, runnable in full and
  small-corpus modes with checkpointed traces and embeddings.
- An optional SemanticGraphicalizer adapter for `abstractgraph.AbstractGraph`
  with on-demand node embedding.
- Focused offline tests using deterministic model and embedding clients.
- Cached experiment artifacts under `data/processed/`, excluded from version
  control.

## Experiment design

### Corpus and semantic graphs

- Load the complete parsed cache with `load_aesop_fables(limit=None)`; use a
  configurable small limit in smoke mode.
- Assign stable tale IDs from source order and a content hash. Keep title,
  character count, word count, and source hash as evaluation metadata.
- Run `transform_with_trace` for each tale and preserve the canonical recursive
  `MultiDiGraph` as the input to the adapter. Checkpoint each completed trace so
  a failed model request can resume without reprocessing earlier tales.

### AbstractGraph conversion and node embeddings

- Provide `semantic_graph_to_abstract_graph(...)` and
  `trace_to_abstract_graph(...)`, plus `SemanticGraphicalizer.to_abstract_graph`
  and `transform_abstract` convenience methods.
- Conversion uses entity `relation` as the base-node label, argument `role` as
  the base-edge label, and provenance plus `Entity.type` to create local
  interpretation nodes. If chunk provenance is absent, group at document scope.
- Embedding is opt-in with `embed_nodes=False` by default. When enabled,
  generate vectors from the existing node-text policy, reuse vectors only when
  their text and embedding configuration match, and recompute missing or stale
  vectors. Store the configuration and text hashes with each vector.
- Validate one vector dimension per document conversion and zero-fill missing
  node vectors when other vectors are present. If no vectors are present, omit
  continuous base-node attributes.
- Preserve semantic metadata when converting the `MultiDiGraph` to a directed
  simple graph. Combine parallel argument roles deterministically by default;
  offer an error policy for callers that disallow them.
- Install AbstractGraph through the optional `abstractgraph` extra. Ordinary
  SemanticGraphicalizer imports and graph processing must work without it.

### Graph and text representations

- Create one AbstractGraph per tale with the existing sum attribute aggregation
  and hash-based interpretation labels.
- Convert `AbstractGraph.to_array()` to one sparse tale vector by summing its
  base-node rows. Keep the sparse representation through scaling and clustering.
- Embed the complete tale text separately for a direct-text baseline. Split
  long tales into deterministic 6,000-character, word-boundary chunks, embed
  those chunks, then mean-pool their vectors. This baseline does not use graph
  content.
- Record embedding model, dimensions, source/configuration hashes, AbstractGraph
  settings, and corpus hash with the run outputs.

### Clustering and interpretation

- Standardize the AbstractGraph sparse vectors and direct-text vectors
  independently, then compare KMeans and agglomerative clustering over a small
  range of cluster counts.
- Report silhouette, Calinski-Harabasz, and Davies-Bouldin scores, cluster
  sizes, PCA/SVD views, and representative tales. Treat internal scores as
  diagnostics, not evidence of semantic validity by themselves.
- Compare assignments with adjusted Rand index and normalized mutual
  information. Inspect common entities, relation names, argument roles,
  excerpts, and boundary cases. Use titles and morals only for this post-hoc
  interpretation.

## Tests and acceptance

- Test semantic label and metadata preservation, chunk grouping and fallback,
  overlapping memberships, parallel-edge policies, embedding opt-in, matching
  vector reuse, stale vector refresh, missing-vector zero fill, and stable
  summed-vector width.
- Verify ordinary package import and graph transformation without the optional
  dependency; run adapter tests with the dependency installed.
- The notebook must run top-to-bottom in small-corpus mode, produce one trace,
  one AbstractGraph, and one vector per selected tale, and compare graph and
  text clustering. API-backed cells must clearly identify embedding costs.
- Keep generated vectors, traces, and credentials out of version control.
