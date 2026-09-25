# Semantic Graphicalizer

Ontology-guided text graphicalization exposed as a scikit-learn-compatible transformer.

The transformer accepts complete documents and returns one `networkx.MultiDiGraph` per document. Call `transform` directly; the configured pipeline is initialized on its first use. Model execution is intentionally provider-neutral: pass an object with `generate(...)` or a compatible callable.

```python
from semantic_graphicalizer import SemanticGraphicalizer

graphicalizer = SemanticGraphicalizer(
    ontology="configs/ontologies/aesop.yaml",
    prompts="configs/prompts/aesop.yaml",
)
graphs = graphicalizer.transform(["The fox met the crow."])
```

When `model` is omitted, the package uses OpenAI `gpt-4.1-mini` through the
Responses API. Set `OPENAI_API_KEY` in the environment before running it:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Pass `model=...` to inject a fake client for tests or another provider.

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
round-trip serialization. Visualization keeps the reified argument edges and
also derives colored temporal and causal links when the ontology declares a
category and ordered binary projection; pass `show_derived_links=False` to
hide those display-only overlays.
Document IDs are stable content-derived IDs by default; pass
`document_id_fn=(text, index) -> str` when an external identifier is available.
Node and edge provenance includes source spans when the model's source text can
be aligned to the original document.

Install the package and development tools with Python 3.12:

```bash
~/.venvs/py312/bin/python -m pip install -e '.[dev]'
~/.venvs/py312/bin/python -m pytest
```

See `notebooks/aesop_tales.ipynb` for the Gutenberg download, fable parsing, graph construction, and visualization workflow.

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
`data/raw/aesop_300_fables.json`, and the source under `data/raw/pg21.txt`;
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
during conversion:

```python
abstract = graphicalizer.to_abstract_graph(graphs[0], embed_nodes=True)
matrix = abstract.to_array()
story_vector = matrix.sum(axis=0)
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
