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

Nodes store the ontology term in `node["label"]`. Edges store the complete proposition in `edge["label"]`; `edge["predicate"]` contains the ontology relation ID. `transform_with_trace` exposes all intermediate stages and provenance.

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

This returns a `list[str]`, one complete story per item, and caches the source
under `data/raw/pg53103.txt`.

Graphs can be rendered inline in a notebook with a D3 force layout:

```python
graphicalizer.display(graphs[0])
```

The default view uses ontology IDs as text-only nodes, proposition labels on
thin gray edges, and no filled node circles.
