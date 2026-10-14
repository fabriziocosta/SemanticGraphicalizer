from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

import pytest

from semantic_graphicalizer import (
    DEFAULT_OPENAI_IMAGE_MODEL,
    IMAGE_EMBEDDING_DIMENSIONS,
    EmbeddingGemma2ImageClient,
    ImageGraphicalizer,
    ModelContent,
    ModelRequest,
    OpenAIModelClient,
    graph_from_dict,
    graph_to_dict,
    load_ontology,
    validate_graph,
)


PIL = pytest.importorskip("PIL.Image")


class FakeImageModel:
    def __init__(self):
        self.requests = []

    def generate_request(self, request):
        self.requests.append(request)
        if request.stage == "inventory":
            return {
                "scene_description": "A leafy plant sits on a table in a bright room.",
                "regions": [
                    {"id": "plant", "label": "plant", "bbox": [0.1, 0.2, 0.5, 0.575], "confidence": 0.9},
                    {"id": "bad-box", "label": "invalid", "bbox": [0.4, 0.4, 0.4, 0.7], "confidence": None},
                ],
            }
        if request.stage == "analyze_entity":
            return {
                "entities": [{
                    "id": "leafy-plant",
                    "type": "visual_entity",
                    "mention": "potted plant",
                    "key": "potted-plant",
                    "description": "A small green plant in a pot.",
                    "properties": [{"name": "color", "value": "green"}],
                    "confidence": 0.95,
                }]
            }
        if request.stage == "relate":
            scene_id, entity_id = request.context["entity_ids"]
            return {
                "relations": [{
                    "id": "scene-contains-plant",
                    "type": "visual_entity",
                    "relation": "contains",
                    "arguments": [
                        {"role": "container", "entity_id": scene_id},
                        {"role": "item", "entity_id": entity_id},
                    ],
                    "description": "The plant is in the room scene.",
                    "confidence": 0.9,
                }]
            }
        raise AssertionError(request.stage)


def test_image_workflow_crops_analyzes_relates_and_preserves_provenance(tmp_path: Path) -> None:
    source = tmp_path / "garden.png"
    PIL.new("RGB", (100, 80), (10, 140, 40)).save(source)
    model = FakeImageModel()
    crop_dir = tmp_path / "crops"
    graphicalizer = ImageGraphicalizer(model=model, crop_dir=crop_dir, verbose=False)

    [graph] = graphicalizer.transform(source)

    assert graph.graph["modality"] == "image"
    assert graph.graph["scene_summary"].startswith("A leafy plant")
    assert graph.graph["warnings"] and "bad-box" in graph.graph["warnings"][0]
    assert graphicalizer.image_embedder is None
    assert [request.stage for request in model.requests] == ["inventory", "analyze_entity", "relate"]
    assert model.requests[0].content[0].value == source
    assert model.requests[0].content[0].modality == "image"
    [part] = graph.graph["parts"]
    assert part["modality"] == "image"
    assert part["source_locator"]["normalized_bbox"] == [0.1, 0.2, 0.5, 0.575]
    assert part["source_locator"]["pixel_bbox"] == [10, 16, 50, 46]
    crop_path = Path(part["metadata"]["crop_path"])
    assert crop_path.is_file()
    with PIL.open(crop_path) as crop:
        assert crop.size == (40, 30)
    crop_request = model.requests[1]
    assert crop_request.content[0].modality == "image"
    assert crop_request.content[0].value == crop_path
    assert crop_request.context["scene_summary"] == graph.graph["scene_summary"]
    entity = next(data for _node, data in graph.nodes(data=True) if data["type"] == "visual_entity" and data["relation"] is None)
    assert entity["attributes"]["properties"] == {"color": "green"}
    assert entity["attributes"]["provenance"][0]["source_locator"]["pixel_bbox"] == [10, 16, 50, 46]
    relation = next(data for _node, data in graph.nodes(data=True) if data["relation"] == "contains")
    assert len(list(graph.out_edges(relation["id"]))) == 2
    validate_graph(graph, load_ontology(graphicalizer.ontology))
    restored = graph_from_dict(graph_to_dict(graph))
    assert restored.graph["source_image"] == str(source)
    assert restored.number_of_nodes() == graph.number_of_nodes()


def test_image_transform_accepts_multiple_paths_and_hashes_documents(tmp_path: Path) -> None:
    image_path = tmp_path / "small.png"
    PIL.new("RGB", (2, 2), (0, 0, 0)).save(image_path)
    model = FakeImageModel()
    graphicalizer = ImageGraphicalizer(model=model, crop_dir=tmp_path / "crops", verbose=False)

    graphs = graphicalizer.transform([image_path, image_path])

    assert len(graphs) == 2
    assert graphs[0].graph["document_id"] != graphs[1].graph["document_id"]


def test_image_part_embeddings_are_attached_cached_and_serializable(tmp_path: Path) -> None:
    source = tmp_path / "office.png"
    PIL.new("RGB", (40, 30), (180, 160, 120)).save(source)

    class FakeImageEmbedder:
        model_id = "fake/image-embedder"

        def __init__(self):
            self.calls = []

        def embed_images(self, image_paths, *, batch_size=32, embedding_dim=IMAGE_EMBEDDING_DIMENSIONS):
            self.calls.append((list(image_paths), batch_size))
            return [[float(index + 1)] * embedding_dim for index, _ in enumerate(image_paths)]

    embedder = FakeImageEmbedder()
    graphicalizer = ImageGraphicalizer(
        model=FakeImageModel(),
        image_embedder=embedder,
        crop_dir=tmp_path / "crops",
        verbose=False,
    )
    graphs = graphicalizer.transform([source, source])
    graph = graphs[0]

    assert graphicalizer.compute_part_embeddings(graphs, batch_size=1) == graphs
    [part] = graph.graph["parts"]
    assert len(embedder.calls) == 2
    assert embedder.calls[0][1] == 1
    assert len(part["embedding"]) == IMAGE_EMBEDDING_DIMENSIONS
    assert sum(value * value for value in part["embedding"]) == pytest.approx(1.0)
    assert part["embedding_metadata"]["model"] == embedder.model_id
    entity = next(data for _node, data in graph.nodes(data=True) if data["type"] == "visual_entity" and data["relation"] is None)
    assert entity["attributes"]["provenance"][0]["part_id"] == part["part_id"]
    assert "embedding" not in entity["attributes"]
    restored = graph_from_dict(graph_to_dict(graph))
    assert restored.graph["parts"][0]["embedding"] == part["embedding"]

    graphicalizer.compute_part_embeddings(graphs)
    assert len(embedder.calls) == 2
    graphicalizer.compute_part_embeddings(graphs, embedding_dim=128, batch_size=1)
    assert len(embedder.calls) == 4
    assert len(part["embedding"]) == 128
    assert part["embedding_metadata"]["dimension"] == 128
    graphicalizer.compute_part_embeddings(graphs, embedding_dim=128, batch_size=1)
    assert len(embedder.calls) == 4
    Path(part["content"]).write_bytes(Path(part["content"]).read_bytes() + b"changed")
    graphicalizer.compute_part_embeddings(graphs, embedding_dim=128, batch_size=1)
    assert len(embedder.calls) == 5


def test_empty_image_graph_does_not_load_embedding_model(tmp_path: Path) -> None:
    source = tmp_path / "empty.png"
    PIL.new("RGB", (8, 8), (20, 20, 20)).save(source)

    class NoRegionModel:
        def generate_request(self, request):
            if request.stage == "inventory":
                return {"scene_description": "A blank gray image.", "regions": []}
            if request.stage == "relate":
                return {"relations": []}
            raise AssertionError(request.stage)

    graphicalizer = ImageGraphicalizer(model=NoRegionModel(), crop_dir=tmp_path / "crops", verbose=False)
    [graph] = graphicalizer.transform(source)
    assert graphicalizer.image_embedder is None
    assert graphicalizer.compute_part_embeddings(graph) == [graph]
    assert graphicalizer.image_embedder is None


@pytest.mark.parametrize("result, message", [
    ([], "different number of vectors"),
    ([[0.0] * (IMAGE_EMBEDDING_DIMENSIONS - 1)], "must contain 768 finite values"),
])
def test_image_part_embedding_rejects_invalid_client_output(tmp_path: Path, result, message: str) -> None:
    source = tmp_path / "office.png"
    PIL.new("RGB", (40, 30), (180, 160, 120)).save(source)

    class BadImageEmbedder:
        model_id = "fake/bad-image-embedder"

        def embed_images(self, image_paths, *, batch_size=32, embedding_dim=IMAGE_EMBEDDING_DIMENSIONS):
            del image_paths, batch_size, embedding_dim
            return result

    graphicalizer = ImageGraphicalizer(
        model=FakeImageModel(),
        image_embedder=BadImageEmbedder(),
        crop_dir=tmp_path / "crops",
        verbose=False,
    )
    [graph] = graphicalizer.transform(source)
    with pytest.raises(ValueError, match=message):
        graphicalizer.compute_part_embeddings(graph)


def test_embedding_gemma_client_uses_normalized_local_image_embeddings() -> None:
    class FakeSentenceTransformer:
        def encode(self, paths, **kwargs):
            assert paths == ["crop-a.png", "crop-b.png"]
            assert kwargs["batch_size"] == 2
            assert kwargs["truncate_dim"] == 128
            assert kwargs["normalize_embeddings"] is True
            assert kwargs["show_progress_bar"] is False
            return [[0.25] * 128, [0.5] * 128]

    client = EmbeddingGemma2ImageClient(model=FakeSentenceTransformer())
    assert client.embed_images(["crop-a.png", "crop-b.png"], batch_size=2, embedding_dim=128) == [
        [0.25] * 128, [0.5] * 128
    ]


def test_image_part_embedding_reports_missing_crop_path(tmp_path: Path) -> None:
    source = tmp_path / "office.png"
    PIL.new("RGB", (40, 30), (180, 160, 120)).save(source)
    graphicalizer = ImageGraphicalizer(
        model=FakeImageModel(),
        crop_dir=tmp_path / "crops",
        verbose=False,
    )
    [graph] = graphicalizer.transform(source)
    crop_path = Path(graph.graph["parts"][0]["content"])
    crop_path.unlink()

    with pytest.raises(ValueError, match="does not exist"):
        graphicalizer.compute_part_embeddings(graph)


def test_image_without_detected_regions_still_returns_scene_graph(tmp_path: Path) -> None:
    image_path = tmp_path / "empty.png"
    PIL.new("RGB", (10, 10), (20, 20, 20)).save(image_path)

    class NoRegionModel:
        def generate_request(self, request):
            if request.stage == "inventory":
                return {"scene_description": "An empty gray wall.", "regions": []}
            if request.stage == "relate":
                assert len(request.context["entity_ids"]) == 1
                return {"relations": []}
            raise AssertionError(f"unexpected stage {request.stage}")

    graph = ImageGraphicalizer(model=NoRegionModel(), crop_dir=tmp_path / "crops", verbose=False).transform(image_path)[0]

    assert graph.number_of_nodes() == 1
    assert graph.nodes[f"{graph.graph['document_id']}:scene"]["type"] == "scene"
    assert graph.graph["parts"] == []


def test_openai_request_client_encodes_local_image_and_structured_schema(tmp_path: Path) -> None:
    image_path = tmp_path / "evidence.png"
    image_path.write_bytes(b"fake image bytes")
    calls = []

    class Responses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_text='{"ok": true}')

    client = OpenAIModelClient(model=DEFAULT_OPENAI_IMAGE_MODEL, client=SimpleNamespace(responses=Responses()))
    request = ModelRequest(
        stage="inventory",
        prompt="Inspect this image.",
        schema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False},
        content=(ModelContent("image", image_path, {"detail": "high"}),),
    )

    assert client.generate_request(request) == {"ok": True}
    assert calls[0]["model"] == "gpt-5.6-terra"
    image_part = calls[0]["input"][0]["content"][1]
    assert image_part["type"] == "input_image"
    assert image_part["detail"] == "high"
    assert image_part["image_url"] == "data:image/png;base64," + base64.b64encode(b"fake image bytes").decode("ascii")
    assert calls[0]["text"]["format"]["type"] == "json_schema"


def test_openai_image_request_rejects_unsupported_modalities() -> None:
    client = OpenAIModelClient(client=SimpleNamespace(responses=SimpleNamespace(create=lambda **_kwargs: None)))
    request = ModelRequest("inventory", "inspect", {}, content=(ModelContent("audio", b"x"),))

    with pytest.raises(TypeError, match="does not support 'audio'"):
        client.generate_request(request)


def test_legacy_text_model_cannot_receive_image_content(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.png"
    PIL.new("RGB", (2, 2)).save(image_path)

    class LegacyTextModel:
        def generate(self, **_kwargs):
            raise AssertionError("legacy model should not be called")

    graphicalizer = ImageGraphicalizer(model=LegacyTextModel(), crop_dir=tmp_path / "crops", verbose=False)
    with pytest.raises(TypeError, match="legacy text model clients cannot consume non-text content"):
        graphicalizer.transform(image_path)
