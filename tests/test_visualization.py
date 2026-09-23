import shutil
import subprocess

import networkx as nx
import pytest

from semantic_graphicalizer import (
    display_graph,
    graph_to_d3_data,
    graph_to_d3_html,
    graph_to_d3_iframe,
    graph_to_d3_javascript,
    graph_to_static_svg,
    graph_to_text,
    load_ontology,
)
from semantic_graphicalizer.graph import materialize_graph, project_binary_relations
from semantic_graphicalizer.types import Argument, Entity, RelationInstance


def test_reified_temporal_and_causal_links_are_derived_for_display() -> None:
    ontology = load_ontology("configs/ontologies/aesop.yaml")
    graph = materialize_graph(
        [
            Entity("event-a", "Action"),
            Entity("event-b", "Action"),
            Entity("animal", "Animal"),
            Entity("trait", "Trait"),
        ],
        [
            RelationInstance(
                "causal-assertion",
                "Action",
                "causes",
                (Argument("cause", "event-a"), Argument("effect", "event-b")),
            ),
            RelationInstance(
                "temporal-assertion",
                "Action",
                "before",
                (Argument("earlier", "event-a"), Argument("later", "event-b")),
            ),
            RelationInstance(
                "trait-assertion",
                "Animal",
                "has_trait",
                (Argument("bearer", "animal"), Argument("trait", "trait")),
            ),
        ],
        ontology,
        document_id="doc",
        document_text="Event A causes and precedes Event B.",
    )

    data = graph_to_d3_data(graph)
    derived = [link for link in data["links"] if link["edge_type"] == "derived_projection"]
    assert {(link["predicate"], link["category"]) for link in derived} == {
        ("causes", "causal"),
        ("before", "temporal"),
    }
    assert sum(link["edge_type"] == "argument" for link in data["links"]) == 0
    assert {node["id"] for node in data["nodes"]} == {"event-a", "event-b", "animal", "trait"}
    projected_trait = next(link for link in data["links"] if link["predicate"] == "has_trait")
    assert projected_trait["edge_type"] == "projection"
    assert projected_trait["category"] == "semantic"
    unprojected = graph_to_d3_data(graph, show_derived_links=False)
    assert "trait-assertion" in {node["id"] for node in unprojected["nodes"]}
    assert any(
        link["source"] == "trait-assertion" and link["predicate"] == "bearer"
        for link in unprojected["links"]
    )

    projected = project_binary_relations(graph, ontology)
    assert projected["event-a"]["event-b"]["projection:causal-assertion"]["category"] == "causal"
    assert projected["event-a"]["event-b"]["projection:temporal-assertion"]["category"] == "temporal"

    svg = graph_to_static_svg(graph, show_source=False)
    assert "#c2410c" in svg
    assert "#2563eb" in svg
    assert "causes" in svg
    assert "before" in svg


def test_d3_data_uses_node_and_relation_labels() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node("fox", label="Animal", mentions=["fox"])
    graph.add_node("crow", label="Animal", mentions=["crow"])
    graph.add_edge(
        "fox",
        "crow",
        label="The fox interacts with the crow.",
        predicate="interacts_with",
    )

    data = graph_to_d3_data(graph)

    assert data == {
        "nodes": [
            {
                "id": "fox",
                "label": "animal\nfox",
                "ontology_label": "animal",
                "source_fragment": "fox",
                "node_type": "entity",
                "sequence": 0,
                "component_order": 0,
                "component_index": 0,
                "component_size": 2,
            },
            {
                "id": "crow",
            "label": "animal\ncrow",
                "ontology_label": "animal",
                "source_fragment": "crow",
                "node_type": "entity",
                "sequence": 1,
                "component_order": 0,
                "component_index": 1,
                "component_size": 2,
            },
        ],
        "links": [{
            "source": "fox",
            "target": "crow",
            "label": "interacts_with\nThe fox interacts with the crow.",
            "predicate": "interacts_with",
            "source_fragment": "The fox interacts with the crow.",
            "edge_type": "semantic",
            "category": "semantic",
            "directed": True,
        }],
    }


def test_parallel_labels_receive_separate_offsets() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node("source", label="Source")
    graph.add_node("target", label="Target")
    graph.add_edge("source", "target", predicate="causes", label="first evidence")
    graph.add_edge("source", "target", predicate="before", label="second evidence")

    data = graph_to_d3_data(graph)
    assert {(link["parallel_index"], link["parallel_count"]) for link in data["links"]} == {
        (0, 2),
        (1, 2),
    }
    html = graph_to_d3_html(graph)
    assert "const edgeOffset = d =>" in html
    assert "const edgeDistance = d =>" in html
    assert "linkGeometry(d).labelX" in html


def test_parallel_edges_use_curved_paths() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node("source", label="Source")
    graph.add_node("target", label="Target")
    graph.add_edge("source", "target", predicate="causes", label="first")
    graph.add_edge("source", "target", predicate="before", label="second")

    html = graph_to_d3_html(graph)
    svg = graph_to_static_svg(graph)

    assert "const parallelCurvature = count > 1" in html
    assert '.attr("d", d => linkGeometry(d).path)' in html
    assert " Q" in svg


def test_parallel_edge_spacing_is_exposed_across_renderers() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node("source", label="Source")
    graph.add_node("target", label="Target")
    graph.add_edge("source", "target", predicate="causes", label="first")
    graph.add_edge("source", "target", predicate="before", label="second")

    html = graph_to_d3_html(graph, parallel_edge_spacing=72)
    javascript = graph_to_d3_javascript(graph, parallel_edge_spacing=72)
    iframe = graph_to_d3_iframe(graph, parallel_edge_spacing=72)
    narrow_svg = graph_to_static_svg(graph, parallel_edge_spacing=24)
    wide_svg = graph_to_static_svg(graph, parallel_edge_spacing=72)

    assert "const parallelEdgeSpacing = 72.0;" in html
    assert "const parallelEdgeSpacing = 72.0;" in javascript
    assert "parallelEdgeSpacing" in iframe
    assert narrow_svg != wide_svg
    with pytest.raises(ValueError, match="parallel_edge_spacing"):
        graph_to_d3_html(graph, parallel_edge_spacing=0)
    with pytest.raises(ValueError, match="parallel_edge_spacing"):
        graph_to_static_svg(graph, parallel_edge_spacing=0)


def test_self_loop_uses_offset_arc_and_label_anchor() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node("fox", label="Animal", mentions=["fox"])
    graph.add_edge("fox", "fox", predicate="interacts_with", label="The fox interacts with itself.")

    html = graph_to_d3_html(graph)
    svg = graph_to_static_svg(graph)
    assert "const isSelfLoop = d =>" in html
    assert ".data(data.links.filter(isSelfLoop))" in html
    assert "selfLoop" in html
    assert "C" in svg
    assert "<path" in svg
    assert "interacts_with" in svg


def test_d3_data_exposes_reified_entity_and_argument_categories() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "relation::r1",
        label="Event",
        type="Event",
        relation="causes",
        mentions=["The fox runs."],
        node_type="entity",
        sequence=0,
    )
    graph.add_node(
        "event::p2",
        label="Event",
        type="Event",
        relation=None,
        mentions=["The fox arrives."],
        node_type="entity",
        sequence=1,
    )
    graph.add_edge(
        "relation::r1",
        "event::p2",
        role="effect",
        category="semantic",
        edge_type="argument",
    )

    data = graph_to_d3_data(graph)

    assert data["nodes"][0]["node_type"] == "entity"
    assert "event : causes" in data["nodes"][0]["label"]
    assert data["links"][0]["predicate"] == "effect"
    assert data["links"][0]["edge_type"] == "argument"


def test_d3_html_has_text_only_nodes_and_gray_thin_edges() -> None:
    graph = nx.MultiDiGraph()
    graph.add_edge("fox", "crow", label="The fox interacts with the crow.", predicate="interacts_with")

    html = graph_to_d3_html(graph)

    assert "https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js" in html
    assert '.selectAll("path")' in html
    assert '.selectAll("text")' in html
    assert 'attr("stroke", d => d.category === "causal"' in html
    assert 'attr("stroke-width", d => d.category === "causal"' in html
    assert 'attr("marker-end", d => d.directed' in html
    assert 'append("marker")' in html
    assert 'force("x", d3.forceX(xTarget)' in html
    assert 'force("y", useTimeline ? d3.forceY(timelineYTarget)' in html
    assert 'category === "causal" ? "#c2410c"' in html
    assert 'category === "temporal" ? "#2563eb"' in html
    assert "nodeRadius" in html
    assert "setMultilineText" in html
    assert ".append(\"tspan\")" in html
    assert '.attr("font-family", index === 0 ? "monospace" : "sans-serif")' in html
    assert "d3.zoom()" in html
    assert "scaleExtent([0.25, 4])" in html
    assert 'event => viewport.attr("transform", event.transform)' in html
    assert 'selectAll("tspan").attr("x", x)' in html
    assert "d.pinned = true" in html
    assert '.on("dblclick"' in html
    assert "drag the background to pan" in html
    assert "temporalArrowId" in html
    assert "causalArrowId" in html
    assert "legend" in html
    assert "<circle" not in html


def test_d3_charge_strength_is_tunable_and_less_repelled_by_default() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node("fox", label="Animal")

    html = graph_to_d3_html(graph)
    tuned = graph_to_d3_html(
        graph,
        charge_strength=-1,
        link_distance=70,
        component_spacing=120,
        component_strength=0.1,
    )

    assert 'force("charge", d3.forceManyBody().strength(-180.0))' in html
    assert 'force("link", d3.forceLink(data.links).id(d => d.id).distance(d =>' in tuned
    assert "const linkDistance = 70.0;" in tuned
    assert "const requestedComponentSpacing = 120.0;" in tuned
    assert "const componentStrength = 0.1;" in tuned
    assert 'force("charge", d3.forceManyBody().strength(-1.0))' in tuned
    timeline = graph_to_d3_html(graph, timeline_stiffness=1.0)
    assert "const timelineStiffness = 1.0;" in timeline
    assert "const hardTimeline = useTimeline && timelineStiffness >= 0.999;" in timeline
    assert "d.x = timelineX(d);" in timeline
    assert "d.y = timelineY;" in timeline
    with pytest.raises(ValueError, match="timeline_stiffness"):
        graph_to_d3_html(graph, timeline_stiffness=1.1)
    with pytest.raises(ValueError, match="timeline_stiffness"):
        graph_to_d3_html(graph, timeline_stiffness=-0.1)
    with pytest.raises(ValueError, match="charge_strength"):
        graph_to_d3_html(graph, charge_strength=10)
    with pytest.raises(ValueError, match="link_distance"):
        graph_to_d3_html(graph, link_distance=0)


def test_timeline_layout_is_explicit_and_type_agnostic() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "event::p1",
        label="event",
        mentions=["The fox runs."],
        node_type="entity",
        sequence=0,
    )
    graph.add_node(
        "event::p2",
        label="event",
        mentions=["The fox arrives."],
        node_type="entity",
        sequence=1,
    )
    graph.add_edge(
        "event::p1",
        "event::p2",
        predicate="next_in_narrative",
        category="temporal",
        edge_type="temporal",
    )

    html = graph_to_d3_html(graph, layout="timeline")

    assert 'const layoutMode = "timeline";' in html
    assert "const useTimeline = layoutMode === \"timeline\"" in html
    assert "timelineScale" in html
    assert "timelineYTarget" in html
    assert 'stroke-dasharray", d => d.category === "temporal"' in html


def test_labels_use_new_lines_and_wrap_at_max_width() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node("fox", label="Animal", mentions=["fox"])
    graph.add_edge(
        "fox",
        "fox",
        label="The fox interacts with the crow.",
        predicate="interacts_with",
    )

    data = graph_to_d3_data(graph, max_width=16)

    assert data["nodes"][0]["label"] == "animal\nfox"
    assert data["links"][0]["label"] == (
        "interacts_with\nThe fox\ninteracts with\nthe crow."
    )


def test_max_width_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_width"):
        graph_to_d3_data(nx.MultiDiGraph(), max_width=0)


def test_display_graph_returns_html_iframe_for_dynamic_mode() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node("fox", label="Animal")

    html = display_graph(graph)
    rendered_html = html._repr_html_()

    assert "<iframe" in rendered_html
    assert "data:text/html;charset=utf-8," in rendered_html
    assert "sandbox=\"allow-scripts allow-same-origin\"" in rendered_html
    assert "100%" in rendered_html


def test_display_graph_does_not_emit_a_second_notebook_output(monkeypatch) -> None:
    import IPython.display

    emitted = []
    monkeypatch.setattr(IPython.display, "display", emitted.append)

    graph = nx.MultiDiGraph()
    graph.add_node("fox", label="Animal")

    rendered = display_graph(graph)

    assert rendered._repr_html_().startswith("\n        <iframe")
    assert emitted == []


def test_d3_iframe_contains_force_layout_document() -> None:
    graph = nx.MultiDiGraph()
    graph.add_edge("fox", "crow", predicate="interacts_with", label="The fox interacts with the crow.")

    iframe = graph_to_d3_iframe(graph)

    assert iframe.startswith('<iframe title="Ontology graph"')
    assert 'height="900"' in iframe
    assert "srcdoc=" in iframe
    assert "d3.forceSimulation" in iframe


def test_d3_javascript_reuses_interactive_renderer() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node("fox", label="Animal", mentions=["fox"])

    javascript = graph_to_d3_javascript(graph)

    assert javascript.startswith("(() => {")
    assert "element.appendChild(root)" in javascript
    assert "d3.forceSimulation" in javascript
    assert "d3.zoom()" in javascript
    assert "selectAll(\"tspan\").attr(\"x\", d.x)" in javascript
    assert "<script" not in javascript


def test_d3_javascript_is_valid_javascript_when_node_is_available(tmp_path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")
    graph = nx.MultiDiGraph()
    graph.add_node("fox", label="Animal", mentions=["fox"])
    script_path = tmp_path / "graph.js"
    script_path.write_text(graph_to_d3_javascript(graph), encoding="utf-8")

    checked = subprocess.run([node, "--check", str(script_path)], capture_output=True, text=True)

    assert checked.returncode == 0, checked.stderr


def test_visualization_can_hide_source_fragments() -> None:
    graph = nx.MultiDiGraph()
    graph.add_edge("fox", "crow", label="The fox interacts with the crow.", predicate="interacts_with")

    data = graph_to_d3_data(graph, show_source=False)

    assert data["links"][0]["label"] == "interacts_with"


def test_components_are_ordered_by_first_sequence_position() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node("late", label="Animal", mentions=["late"], sequence=3)
    graph.add_node("early", label="Animal", mentions=["early"], sequence=0)
    graph.add_edge("late", "late", predicate="causes", label="A late event.")
    graph.add_edge("early", "early", predicate="causes", label="An early event.")

    data = graph_to_d3_data(graph)
    by_id = {node["id"]: node for node in data["nodes"]}

    assert by_id["early"]["component_order"] == 0
    assert by_id["late"]["component_order"] == 1


def test_static_display_uses_kamada_kawai_and_text_only_svg() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node("fox", label="Animal", mentions=["fox"])
    graph.add_node("action", label="Action", mentions=["an action"])
    graph.add_edge("fox", "action", label="The fox performs an action.", predicate="performs")

    svg = graph_to_static_svg(graph)
    rendered = display_graph(graph, mode="static")

    assert "Kamada-Kawai" in svg
    assert 'stroke="#9aa0a6"' in svg
    assert '<text' in svg
    assert "animal" in svg
    assert 'font-family="monospace"' in svg
    assert ">fox</tspan>" in svg
    assert "performs" in svg
    assert "The fox performs an action." in svg
    assert "<tspan" in svg
    assert "<circle" not in svg
    assert rendered.data.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert "Kamada-Kawai" in rendered.data


def test_display_graph_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="dynamic.*static"):
        display_graph(nx.MultiDiGraph(), mode="unknown")


def test_text_mode_lists_nodes_mentions_and_relations() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node("fox", label="Animal", mentions=["the fox"])
    graph.add_node("action", label="Action", mentions=["an action"])
    graph.add_edge("fox", "action", predicate="performs", label="The fox performs an action.")

    rendered = graph_to_text(graph)

    assert rendered == (
        "animal: the fox\n"
        "    performs: action: an action"
    )


def test_display_graph_supports_text_mode() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node("fox", label="Animal", mentions=["fox"])

    rendered = display_graph(graph, mode="text")

    assert rendered.data == ""
