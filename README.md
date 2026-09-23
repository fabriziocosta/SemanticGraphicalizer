# Semantic Graphicalizer

Ontology-guided text graphicalization exposed as a scikit-learn-compatible transformer.

The transformer accepts complete documents and returns one `networkx.MultiDiGraph` per document. Model execution is intentionally provider-neutral: pass an object with `generate(...)` or a compatible callable.

```python
from semantic_graphicalizer import SemanticGraphicalizer

graphicalizer = SemanticGraphicalizer(
    ontology="configs/ontologies/aesop.yaml",
    prompts="configs/prompts/aesop.yaml",
)
graphs = graphicalizer.fit_transform(["The fox met the crow."])
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
graph mechanism. `transform_with_trace` exposes extracted `entities`,
`relations`, stage statistics, and the canonical graph.

Use `project_binary_relations(graph, ontology)` for an explicit, schema-driven
direct-edge view. Use `graph_to_dict` and `graph_from_dict` for JSON-safe
round-trip serialization.
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
```

This returns a `list[str]`, one complete story per item, and caches the parsed
results under `data/raw/aesop_fables.json`. The Gutenberg source is also kept
under `data/raw/pg53103.txt`; after the first call, subsequent calls do not
download anything.

Graphs can be rendered inline in a notebook with either the interactive D3
layout or a deterministic static SVG layout:

```python
graphicalizer.display(graphs[0], mode="dynamic")
graphicalizer.display(graphs[0], mode="static")
graphicalizer.display(graphs[0], mode="text")
```

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
stage timings and counts remain available on each `DocumentTrace.stats` item
when using `transform_with_trace`.

Long paragraphs are split at word boundaries, and transient model-provider
failures are retried with exponential backoff. Configure `max_retries` and
`retry_backoff` on `SemanticGraphicalizer` when a provider needs different
limits; set `max_retries=0` to disable retries.
