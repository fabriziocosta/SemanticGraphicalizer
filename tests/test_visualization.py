import networkx as nx
import pytest

from semantic_graphicalizer import (
    display_graph,
    graph_to_d3_data,
    graph_to_d3_html,
    graph_to_d3_iframe,
    graph_to_static_svg,
)


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
                "label": "Animal\nfox",
                "ontology_label": "Animal",
                "source_fragment": "fox",
                "sequence": 0,
                "component_order": 0,
                "component_index": 0,
                "component_size": 2,
            },
            {
                "id": "crow",
            "label": "Animal\ncrow",
                "ontology_label": "Animal",
                "source_fragment": "crow",
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
        }],
    }


def test_d3_html_has_text_only_nodes_and_gray_thin_edges() -> None:
    graph = nx.MultiDiGraph()
    graph.add_edge("fox", "crow", label="The fox interacts with the crow.", predicate="interacts_with")

    html = graph_to_d3_html(graph)

    assert "https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js" in html
    assert '.selectAll("line")' in html
    assert '.selectAll("text")' in html
    assert 'attr("stroke", "#9aa0a6")' in html
    assert 'attr("stroke-width", 1)' in html
    assert 'force("component-order"' in html
    assert "nodeRadius" in html
    assert "setMultilineText" in html
    assert ".append(\"tspan\")" in html
    assert "<circle" not in html


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

    assert data["nodes"][0]["label"] == "Animal\nfox"
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
    assert "srcdoc=" in iframe
    assert "d3.forceSimulation" in iframe


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
    assert "Animal" in svg
    assert "Animal" in svg
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
