# Semantic Graphicalizer

Ontology-guided graphicalization for text and image inputs, exposed through
scikit-learn-compatible transformers.

Project design notes, the Aesop experiment plan, and the implementation
scratchpad are collected in the [docs folder](docs/README.md).

The transformer accepts complete documents and returns one `networkx.MultiDiGraph` per document. Call `transform` directly; the configured pipeline is initialized on its first use. Model execution is intentionally provider-neutral: pass an object with `generate(...)` or a compatible callable.

```python
from semantic_graphicalizer import SemanticGraphicalizer

graphicalizer = SemanticGraphicalizer(
    ontology="domains/text/configs/ontologies/aesop.yaml",
    prompts="domains/text/configs/prompts/aesop.yaml",
)
graphs = graphicalizer.transform(["The fox met the crow."])
```

## Modality extension contracts

`DocumentInput` and `Part` provide modality-neutral contracts for input
adapters. Parts carry a modality, content or a media reference, a source
locator, and metadata. The paragraph segmenter exposes
`partition(DocumentInput)` and still supports its original `segment(...)`
method. `SemanticGraphicalizer` remains the text workflow. `ImageGraphicalizer`
accepts local image paths, identifies foreground regions, persists crops,
analyzes them against an image ontology, and extracts document-level relations.
See the [image domain guide](domains/image/README.md) and its
[scene-to-graph notebook](domains/image/notebooks/scene_to_graph.ipynb).

Model providers can implement `generate_request(ModelRequest)` to receive
structured requests with separate text or image evidence. Existing
`generate(stage=..., prompt=..., schema=..., context=...)` clients and
callables continue to work through a text-only adapter for the text workflow.
The image workflow requires a request client that accepts image evidence.

When `model` is omitted, the package uses OpenAI `gpt-4.1-mini` through the
Responses API. Set `OPENAI_API_KEY` in the environment before running it:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Pass `model=...` to inject a fake client for tests or another provider.
`ImageGraphicalizer` defaults to `gpt-5.6-terra` while preserving the text
default above. Install Pillow for image loading and crop persistence with
`pip install -e '.[image]'`.

Every semantic object is an Entity node. Atomic nodes have `relation=None`;
reified relation nodes have a configured `relation` and outgoing
`edge_type="argument"` edges labelled by `role`. Node `type`, `relation`, and
the complete `attributes` mapping are preserved alongside document metadata
and provenance. Relation nodes can point to other relation nodes, so causal,
temporal, evidential, logical, state, and measurement assertions use the same
graph mechanism. Pipeline details are stored on each graph's global metadata
dictionary (`graph.graph`): `chunks`, `summaries`, `normalized`, `entities`,
`relations`, and per-stage `stats`. The graph itself remains the single result
object.

Use `project_binary_relations(graph, ontology)` for an explicit, schema-driven
direct-edge view. Use `graph_to_dict` and `graph_from_dict` for JSON-safe
round-trip serialization. Visualization projects standalone reified relations
to direct edges by default, without requiring ontology projection settings.
N-ary relations render as one edge per pair of participants, following their
argument order. A relation node used as an argument by another relation stays
reified. Pass `show_relations_as_edges=False` to keep the reified relation
nodes and their argument edges. Pass node IDs with
`display_graph(graph, exclude_nodes=...)` to omit nodes from a rendered copy
without changing the graph data.
Document IDs are stable content-derived IDs by default; pass
`document_id_fn=(text, index) -> str` when an external identifier is available.
Node and edge provenance includes source spans when the model's source text can
be aligned to the original document.

Install the package and development tools with Python 3.12:

```bash
~/.venvs/py312/bin/python -m pip install -e '.[dev]'
~/.venvs/py312/bin/python -m pytest
```

See `domains/text/notebooks/aesop_tales.ipynb` for the Gutenberg download, fable parsing, graph construction, and visualization workflow. Text-specific configs, notebooks, experiment code, and data live under `domains/text/`.

The download and cache logic is reusable outside the notebook:

```python
from semantic_graphicalizer import load_aesop_fables

stories = load_aesop_fables(limit=2)
# Complete parsed corpus:
all_stories = load_aesop_fables(limit=None)
# Reproducible random selection from the full cached collection:
stories = load_aesop_fables(limit=2, select_at_random=True, rand_seed=7)
```

This returns a `list[str]`, one complete story per item, from Project
Gutenberg's *Three Hundred Aesop's Fables*. The edition contains 313 indexed
story headings. The parsed stories are cached under
`domains/text/data/raw/aesop_300_fables.json`, and the source under
`domains/text/data/raw/pg21.txt`;
after the first call, subsequent calls do not download anything.

Graphs can be rendered inline in a notebook with either the interactive D3
layout or a deterministic static SVG layout:

```python
graphicalizer.display(graphs[0], mode="dynamic")
graphicalizer.display(graphs[0], mode="static")
graphicalizer.display(graphs[0], mode="text")
```

The default OpenAI text embedder can vectorize the available text for every
node in graphs. Vectors are attached in place under the `embedding`
node attribute; pass `embedder=...` to inject another provider or
`node_text_fn=...` to control the text sent for each node:

```python
graphs = graphicalizer.transform(["The fox met the crow."])
embedded_graphs = graphicalizer.compute_embeddings(graphs)
vector = embedded_graphs[0].nodes["some-node"]["embedding"]
```

The default embedding model is `text-embedding-3-small` and uses the
OpenAI embeddings endpoint. Set `OPENAI_API_KEY` before requesting embeddings.

For a compact dynamic layout, reduce the link distance and component spacing;
set `charge_strength=0` to disable charge repulsion entirely:

```python
graphicalizer.display(
    graphs[0],
    mode="dynamic",
    timeline_stiffness=1.0,
    charge_strength=0,
    link_distance=80,
    component_spacing=120,
    component_strength=0.15,
)
```

`timeline_stiffness` remains available for compatibility with manually
annotated timeline graphs. Reified graphs default to the force layout; use
`layout="force"` or `layout="timeline"` explicitly when desired.

Text mode prints relation nodes with their type and relation, followed by
indented argument-role and target lines. Dynamic and static renderers show
Entity types, relation names, argument roles, and source mentions.

The default view shows ontology values plus source mentions. Use
`show_source=False` to display only ontology IDs, relation names, and argument
roles.

Progress reporting is enabled by default. Set `verbose=False` to suppress it;
stage timings and counts remain available under `graph.graph["stats"]`.

Long paragraphs are split at word boundaries, and transient model-provider
failures are retried with exponential backoff. Configure `max_retries` and
`retry_backoff` on `SemanticGraphicalizer` when a provider needs different
limits; set `max_retries=0` to disable retries.

The `extract` and `resolve` stages validate relation argument IDs through the
same retry-and-recovery mechanism. If an ID is missing from the response or
current graph, the model gets corrective feedback and another attempt, up to
`max_retries`. Relations that still contain invalid references, along with
relations that depend on them, are omitted. If a model fails or keeps
returning structurally invalid output, the affected extraction chunk or
optional resolve stage is skipped so the rest of the document can continue.
Other pipeline stages retain their normal validation errors.

## AbstractGraph adapter

Install the optional integration with `pip install -e '.[abstractgraph]'`.
Convert a graph to an AbstractGraph, optionally computing node text embeddings
during conversion. Node embeddings can also be reduced with a seeded Gaussian
random projection by setting `embedding_dim` to the desired positive dimension.
The default `embedding_dim=None` leaves embeddings unchanged, and the random
seed defaults to 17. The projection is applied to a graph copy, so cached
source embeddings keep their original dimension:

```python
from abstractgraph import AbstractGraphTransformer

abstract = graphicalizer.to_abstract_graph(
    graphs[0],
    embed_nodes=True,
    embedding_dim=128,
    random_seed=17,
)
graph_vectorizer = AbstractGraphTransformer(
    nbits=14,
    decomposition_function=None,
    return_dense=False,
    n_jobs=1,
)
story_vector = graph_vectorizer.fit_transform([abstract])
```

Directedness is preserved by default. Pass `preserve_direction=False` to build
an undirected base graph; reciprocal semantic edges are combined and their
original endpoints remain recorded in `semantic_edges`:

```python
abstract = graphicalizer.to_abstract_graph(graphs[0], preserve_direction=False)
```

Convert multiple existing graphs in input order with
`to_abstract_graphs`. When `embed_nodes=True`, node texts from the full input
collection are embedded in batches before conversion:

```python
abstract_graphs = graphicalizer.to_abstract_graphs(graphs, embed_nodes=True)
```

`embed_nodes` defaults to `False`, so conversion does not make embedding API
requests unless requested. Matching vectors are reused when their text and
embedding configuration are unchanged. Direct conversion helpers are also
available as `semantic_graph_to_abstract_graph(graph)`; this uses embeddings
already attached to the semantic graph.
