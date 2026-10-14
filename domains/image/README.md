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

The default visual ontology and image-specific prompts live under
`configs/ontologies/` and `configs/prompts/`; pass custom YAML paths or mappings
to `ImageGraphicalizer` to adapt them. The built-in ontology uses `scene` and
`visual_entity` nodes with containment, spatial, and interaction relations.

- `data/raw/` — local inputs or source manifests. Keep private or large files
  out of version control.
- `data/processed/regions/` — persisted PNG crops and image provenance. This
  output is ignored by Git.
- `notebooks/` — end-to-end examples and exploration.
- `experiments/` — reusable image-domain experiments.

Each graph records its image source, checksum, dimensions, scene summary,
region parts, crop paths, normalized boxes, pixel boxes, and skipped-region
warnings. OCR, video, and remote image loading are not part of this workflow.
