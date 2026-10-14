# Image domain

The image workflow accepts local scene photographs and returns the shared
canonical NetworkX `MultiDiGraph` representation. It uses three model stages:

1. Describe the full scene and locate foreground entities with normalized
   bounding boxes.
2. Save each region crop and analyze it against the configured ontology.
3. Extract ontology-defined relations across entities and the scene node.

Install the optional image dependency with `pip install -e '.[image]'`. The
default model client uses OpenAI Responses and reads `OPENAI_API_KEY` from the
environment. Pass a request client implementing `generate_request(request)` to
use another provider or a local/fake model.

```python
from pathlib import Path
from semantic_graphicalizer import ImageGraphicalizer

graphicalizer = ImageGraphicalizer(verbose=True)
graphs = graphicalizer.transform(Path("domains/image/data/raw/scene.jpg"))
graph = graphs[0]
```

The repository includes an office sample image at
`data/raw/office_scene.png`. To run the example against it, pass
`Path("domains/image/data/raw/office_scene.png")` to `transform`.

The default visual ontology and image-specific prompts live under
`configs/ontologies/` and `configs/prompts/`; pass custom YAML paths or mappings
to `ImageGraphicalizer` to adapt them. The built-in ontology includes `scene`,
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
`ImageGraphicalizer.display_on_image(graph)` to overlay detected boxes, graph
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
their saved region crops:

```python
embedded_graphs = graphicalizer.compute_part_embeddings(graphs, batch_size=32)
part = embedded_graphs[0].graph["parts"][0]
print(part["part_id"], len(part["embedding"]))  # 768 dimensions
```

Vectors are stored on graph part records with model and crop checksum metadata.
Entity provenance already contains the matching `part_id`, so consumers can
resolve an entity to its region vector. Unchanged crops with matching embedding
settings reuse their saved vectors. The model returns normalized image vectors
in a shared text and image space; vectors are not attached to entity nodes.
