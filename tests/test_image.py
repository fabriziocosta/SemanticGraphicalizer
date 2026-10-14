from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

import pytest

from semantic_graphicalizer import (
    DEFAULT_OPENAI_IMAGE_MODEL,
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
