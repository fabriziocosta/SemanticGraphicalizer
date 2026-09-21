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

Nodes store the ontology term in `node["label"]`. Nodes and edges also carry
`document_id` and the original `document_text`; chunk-level excerpts remain in
their provenance fields. Edges store the complete proposition in
`edge["label"]`, while `edge["predicate"]` contains the ontology relation ID.
Nodes carry a first-seen `sequence`; visual layouts order disconnected
components from left to right by the earliest node sequence in the document.
`transform_with_trace` exposes all intermediate stages and provenance.
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

Graphs can be rendered inline in a notebook with either the interactive D3 force
layout or a deterministic static NetworkX Kamada-Kawai layout:

```python
graphicalizer.display(graphs[0], mode="dynamic")
graphicalizer.display(graphs[0], mode="static")
graphicalizer.display(graphs[0], mode="text")
```

Text mode prints each node as `Ontology: surface text`, followed by indented
`relation: TargetOntology: target text` lines. Incoming relations are marked
with `←`.

The default view shows both values: ontology IDs plus surface mentions on
text-only nodes, and ontology relation IDs plus proposition fragments on thin
gray edges. Use `show_source=False` to display only ontology IDs and relation
IDs.

Progress reporting is enabled by default. Set `verbose=False` to suppress it;
stage timings and counts remain available on each `DocumentTrace.stats` item
when using `transform_with_trace`.

Long paragraphs are split at word boundaries, and transient model-provider
failures are retried with exponential backoff. Configure `max_retries` and
`retry_backoff` on `SemanticGraphicalizer` when a provider needs different
limits; set `max_retries=0` to disable retries.
