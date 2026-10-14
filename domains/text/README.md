# Text domain

Text-specific resources are grouped here so another domain can add its own
configuration, examples, experiments, and data without competing for root
paths.

- `configs/` — text extraction ontologies and prompt packs.
- `notebooks/` — text-focused examples and experiments.
- `experiments/` — reusable code for the Aesop graph and text clustering study.
- `data/` — raw caches and generated experiment artifacts; these are ignored by Git.

The reusable Python package remains at the repository root in
`semantic_graphicalizer/`.

## Node embeddings

`SemanticTextGraphicalizer` uses local EmbeddingGemma 2 for graph node text
embeddings by default. Install the optional runtime with
`pip install -e '.[text-embeddings]'`. Choose `embedding_model`,
`embedding_dim`, and optionally `embedding_source_dim` and
`embedding_projection_seed` when constructing the graphicalizer. Native output
dimensions (`128`, `256`, `512`, `768`) use Matryoshka truncation; other
positive dimensions use a reproducible Gaussian random projection. Model
weights load only when `compute_embeddings` is called.

```python
from semantic_graphicalizer import SemanticTextGraphicalizer

graphicalizer = SemanticTextGraphicalizer(
    ontology="domains/text/configs/ontologies/aesop.yaml",
    prompts="domains/text/configs/prompts/aesop.yaml",
    embedding_model="google/embeddinggemma-2",
    embedding_dim=10,
    embedding_source_dim=128,
    embedding_projection_seed=17,
)
graphs = graphicalizer.transform(["The fox met the crow."])
graphicalizer.compute_embeddings(graphs)
```

Pass a custom client using `embedder=...` to select another provider. For
OpenAI embeddings, inject `OpenAIEmbeddingClient` explicitly.
