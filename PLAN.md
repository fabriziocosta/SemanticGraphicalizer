# Plan: Aesop Attributed-Graph Embedding and Clustering Experiment

## Objective

Create a reproducible notebook that processes the complete cached Aesop corpus,
builds one recursive semantic graph per tale, computes text embeddings for graph
nodes, converts each attributed graph into one fixed-length graph vector, and
clusters the tales to test whether the resulting groups are semantically
coherent.

The experiment should distinguish semantic signal from implementation artifacts:
story titles, moral headings, and cluster labels may be used for interpretation
and evaluation, but must not be included in the graph embeddings used for
clustering.

## Deliverables

- `notebooks/aesop_graph_embeddings_clustering.ipynb`
  - executable from top to bottom;
  - explains each experiment decision and displays intermediate artifacts;
  - uses cached inputs and checkpointed outputs where possible.
- A reusable attributed-graph vectorizer in the package, preferably exposed as
  `AttributedGraphVectorizer`.
- Tests for graph vectorization, batching, empty/sparse graphs, and stable
  feature dimensions.
- Optional cached experiment artifacts outside version control:
  - corpus metadata;
  - serialized traces/graphs with node embeddings;
  - graph-level vectors and cluster assignments;
  - experiment configuration and corpus/model hashes.

## Experiment design

### 1. Corpus and metadata

- Load the complete cached corpus with `load_aesop_fables`, rather than a random
  sample.
- Assign stable tale IDs derived from source order and a content hash.
- Keep a metadata table with tale ID, title, character count, word count, and
  source text hash.
- Treat title/moral text as evaluation metadata only. Do not feed it to the
  node-text embedder unless it is genuinely part of the graph node evidence.
- Record the exact corpus cache path, source URL, extraction count, and date of
  the run.

### 2. Graph construction

- Instantiate `SemanticGraphicalizer` with the Aesop ontology and prompts.
- Run `transform_with_trace` for every tale so both the trace and canonical
  recursive `MultiDiGraph` are available.
- Preserve reified relation nodes and argument-role edges as the primary graph
  representation. This is important because relations can point to other
  relations.
- Keep a projected binary graph as an explicit ablation using
  `project_binary_relations`; do not silently replace the recursive graph.
- Add checkpointing after each completed tale or batch so a failed API request
  does not require restarting the full corpus.

### 3. Node text embeddings

- Call `compute_embeddings(traces)` using the default OpenAI text embedder.
- Use the existing default node text policy: ontology type/relation plus node
  mentions and available source text.
- Store each vector at `graph.nodes[node_id]["embedding"]`.
- Record the embedding model, dimensions, batch size, and number of embedded
  nodes.
- Support resume behavior by skipping nodes that already have an embedding with
  the same model/configuration hash.
- Keep an injectable deterministic fake embedder in tests; notebook execution
  must not be required for the test suite.

### 3b. Direct whole-tale text baseline

- Compute one embedding directly from the complete text of each tale using the
  same embedding provider and model family.
- Exclude graph construction, node pooling, relation attributes, and structural
  features from this baseline.
- Use the direct tale vectors as a separate clustering input, with the same
  standardization policy, random seeds, cluster-count sweep, and evaluation
  metrics used for graph vectors.
- Store the direct text vector and its input-text hash alongside the tale
  metadata so the comparison is auditable.
- If the complete tale exceeds the embedding model's input limit, use a defined
  chunking policy and deterministic pooling, and report that policy explicitly.

### 4. Attributed-graph vectorization

Implement a reusable `AttributedGraphVectorizer` that maps one graph to one
fixed-length vector. The vectorizer must preserve both node attributes and graph
structure and must fit its vocabulary/statistics on the experiment collection.

Use the following representations as explicit comparisons:

1. **Node-attribute baseline**
   - mean, standard deviation, and max pooling over node text embeddings;
   - normalized graph size and relation-node proportion.

2. **Symbolic attributed-graph representation**
   - normalized counts for ontology node types;
   - normalized counts for relation names, argument roles, and edge categories;
   - structural features such as node/edge counts, density, in/out-degree
     summaries, connected components, recursive relation depth, and cycle count;
   - optional Weisfeiler-Lehman graph tokens with TF-IDF or a fitted hashing
     representation.

3. **Combined representation**
   - concatenate the pooled text-embedding features and symbolic/structural
     features;
   - standardize numeric features before clustering;
   - optionally apply PCA only after reporting the unreduced baseline.

The notebook should compare canonical recursive graphs with projected binary
graphs as an ablation. The vectorizer should make the graph view explicit in its
configuration so the two experiments cannot be confused.

### 5. Clustering

- Run the same clustering workflow independently on:
  - direct whole-tale text embeddings;
  - node-attribute pooled graph vectors;
  - symbolic/structural graph vectors;
  - combined attributed-graph vectors;
  - projected-binary graph vectors as an ablation.
- Start with standardized vectors for each representation.
- Evaluate a small range of `k` values with KMeans and agglomerative clustering.
- Select candidate clusterings using silhouette, Calinski-Harabasz, and
  Davies-Bouldin scores, but do not treat the best internal score as proof of
  semantic validity.
- Include PCA plots for visual inspection; use a fixed random seed.
- Report cluster sizes and representative tales nearest to each centroid or
  medoid.
- If the dataset contains too many tiny or noisy clusters, compare a density or
  hierarchical alternative rather than forcing an arbitrary `k`.

### 6. Semantic-sense validation

For each candidate clustering:

- display representative tale titles and short source excerpts per cluster;
- compare shared entities, relation types, argument roles, and graph motifs;
- inspect nearest-neighbor tale pairs in graph-vector space;
- compare against a text-only embedding baseline and a structure-only baseline;
- compare cluster assignments directly across the whole-tale text baseline and
  each graph representation using adjusted Rand index, normalized mutual
  information, and a contingency/alignment table;
- measure cluster stability under resampling, feature ablations, and random
  seeds;
- use titles/morals only as post-hoc human-readable validation signals;
- manually inspect a few boundary cases and likely false positives.

Success means that several clusters are interpretable through recurring themes,
roles, mechanisms, or narrative structures, and that the combined graph view
either improves semantic coherence over direct text clustering or reveals a
useful complementary organization of the tales. If graph clustering merely
duplicates the text baseline, that is still a meaningful result. A high
silhouette score alone is not sufficient.

## Notebook outline

1. **Experiment configuration**
   - paths, seeds, model names, batch sizes, clustering range, and cache policy;
   - a visible warning that embeddings require `OPENAI_API_KEY` and incur API
     usage.
2. **Load and inspect the corpus**
   - corpus count, length distribution, title examples, and hashes.
3. **Build traces and canonical graphs**
   - progress table, graph size distributions, and one recursive graph example.
4. **Compute node embeddings**
   - checkpoint/resume logic, embedding dimensions, and node-text examples.
5. **Vectorize attributed graphs**
   - feature counts, ablation configuration, and sanity checks on vector shapes.
6. **Cluster graph vectors**
   - metric sweep, selected configurations, cluster sizes, and PCA views for
     every representation.
7. **Compare direct text and graph clustering**
   - side-by-side metrics, cluster alignment, nearest-neighbor overlap, and
     examples where graph structure changes the assignment.
8. **Interpret clusters**
   - representative tales, common graph features, nearest neighbors, and
     boundary cases.
9. **Compare baselines and ablations**
   - direct text, node-pooled, structure-only, combined, recursive, and
     projected views.
10. **Conclusions and limitations**
   - whether semantic grouping is supported, what failed, and what to try next.

## Reproducibility and cost controls

- Never commit API keys, raw credentials, or large generated embeddings.
- Use deterministic seeds for sampling, PCA, and clustering.
- Persist configuration, package version, ontology/prompt hashes, corpus hash,
  embedding model, and vectorizer settings with every result.
- Make the notebook runnable in a small smoke-test mode, for example 3 tales,
  before the full-corpus mode.
- Cache model outputs and embeddings by content/configuration hash.
- Make failures retryable and report which tales remain incomplete.
- Separate data acquisition, graph extraction, embedding, vectorization, and
  clustering outputs so each stage can be rerun independently.

## Acceptance criteria

- The notebook runs end-to-end in smoke-test mode without manual edits.
- Full-corpus mode produces one trace/graph and one graph vector per tale.
- Every embedded node has a vector of consistent dimension and provenance for
  the embedding configuration.
- Recursive and projected graph representations are compared explicitly.
- Direct whole-tale text clustering is evaluated alongside at least three graph
  representations and two clustering algorithms.
- The notebook shows cluster metrics, representative tales, PCA plots, and
  qualitative semantic inspection.
- Results can be regenerated from the recorded configuration and cached inputs.
- Tests pass without network access by using fake model and embedding clients.

## Recommended implementation order

- [ ] Add `AttributedGraphVectorizer` and its tests.
- [ ] Add serialization/checkpoint helpers for traces, embeddings, and vectors.
- [ ] Add a smoke-test notebook skeleton with synthetic/fake clients.
- [ ] Wire the notebook to the cached Aesop corpus and real embeddings.
- [ ] Add clustering, baseline, and recursive-vs-projected ablation sections.
- [ ] Run the full experiment, inspect clusters, and record conclusions in the
      notebook.
- [ ] Update the README with the notebook workflow and artifact instructions.
