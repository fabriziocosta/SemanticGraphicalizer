# Image domain

The image workflow accepts local scene photographs and returns the shared
canonical NetworkX `MultiDiGraph` representation. It uses three model stages:

1. Describe the full scene and locate foreground entities with normalized
   bounding boxes.
2. Save each region crop and analyze it against the configured ontology.
3. Extract ontology-defined relations across entities and the scene node.

Install the optional image dependency with `pip install -e '.[image]'`. The
default VLM is `gpt-5.6-terra`; the OpenAI Responses client reads
`OPENAI_API_KEY` from the environment. Set `vlm_model` to select another
OpenAI model. Pass a request client via `model` implementing
`generate_request(request)` to use another provider or a local/fake VLM.

```python
from pathlib import Path
from semantic_graphicalizer import SemanticImageGraphicalizer

graphicalizer = SemanticImageGraphicalizer(vlm_model="gpt-5.6-terra", verbose=True)
graphs = graphicalizer.transform(Path("domains/image/data/raw/scene.jpg"))
graph = graphs[0]
```

The repository includes an office sample image at
`data/raw/office_scene.png`. To run the example against it, pass
`Path("domains/image/data/raw/office_scene.png")` to `transform`.

The default visual ontology and image-specific prompts live under
`configs/ontologies/` and `configs/prompts/`; pass custom YAML paths or mappings
to `SemanticImageGraphicalizer` to adapt them. The built-in ontology includes `scene`,
a `visual_entity` fallback, and specific types such as `person`, `animal`,
`plant`, `vehicle`, `tool`, `building`, `road`, `furniture`, `object`, and
`landscape_feature`. Office types include `chair`, `desk`, `table`, `monitor`,
`computer`, `laptop`, `keyboard`, `mouse`, `printer`, `telephone`,
`filing_cabinet`, `shelf`, `bookshelf`, `whiteboard`, `desk_lamp`, `table_lamp`,
`cup`, `drinking_glass`, `document`, and `office_supply`. It defines containment,
spatial, and interaction relations.

- `data/raw/` — local inputs or source manifests. Keep private or large files
  out of version control.
- `data/processed/regions/` — persisted PNG crops and image provenance. This
  output is ignored by Git.
- `notebooks/` — end-to-end examples and exploration.
- `experiments/` — reusable image-domain experiments.

Each graph records its image source, checksum, dimensions, scene summary,
region parts, crop paths, normalized boxes, pixel boxes, and skipped-region
warnings. Crop analysis also returns an entity box within each crop. These boxes
are mapped to the source image and used to merge same-type, semantically matching
detections whose boxes overlap across crops. The merged node retains provenance
from each crop, and `graph.graph["duplicate_entities_merged"]` records how many
detections were merged. Use `display_image_graph(graph)` or
`SemanticImageGraphicalizer.display_on_image(graph)` to overlay detected boxes, graph
nodes, relation nodes, and argument links on the source image. Whole-scene nodes
and their relations are hidden by default to reduce clutter; pass
`show_scene_nodes=True` to include them. The standard `display_graph` renderer
also accepts explicit node IDs through `exclude_nodes`, for example to omit
the scene node from a graph view. Image overlay labels use separate lines and
are repositioned to avoid colliding with other labels. OCR, video, and remote
image loading are not part of this workflow.

### Image part embeddings

Install the local EmbeddingGemma 2 runtime with
`pip install -e '.[image-embeddings]'`. The model and its weights load only
when the embedding method is called. After creating graphs, explicitly embed
their saved region crops. Choose the embedder and output dimensions when
constructing `SemanticImageGraphicalizer`; the defaults are EmbeddingGemma 2 and 768
dimensions. Set `image_embedding_model` to choose a Sentence Transformers model,
or pass a custom client as `image_embedder`. Native output dimensions
(`128`, `256`, `512`, `768`) use the model's Matryoshka truncation. Any smaller
positive target can instead be created with a reproducible Gaussian random
projection from `image_embedding_source_dim`:

```python
graphicalizer = SemanticImageGraphicalizer(
    image_embedding_model="google/embeddinggemma-2",
    image_embedding_dim=10,
    image_embedding_source_dim=128,
)
embedded_graphs = graphicalizer.compute_part_embeddings(graphs, batch_size=32)
part = embedded_graphs[0].graph["parts"][0]
print(part["part_id"], len(part["embedding"]))  # 10 dimensions in this setup
```

Set `image_embedding_dim` to the desired positive output dimension. For native
Matryoshka dimensions, `image_embedding_source_dim` can be left unset. For
other dimensions, set the source to a supported model dimension greater than
the target. The optional `embedding_dim` argument on
`compute_part_embeddings` overrides the target for a one-off call. Projected
vectors are normalized for cosine similarity; the random seed is recorded in
their metadata.

Vectors are stored on graph part records with model and crop checksum metadata.
Entity provenance already contains the matching `part_id`, so consumers can
resolve an entity to its region vector. Unchanged crops with matching embedding
settings reuse their saved vectors. The model returns normalized image vectors
in a shared text and image space; vectors are not attached to entity nodes.
