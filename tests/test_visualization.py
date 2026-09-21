import networkx as nx
import pytest

from semantic_graphicalizer import (
    display_graph,
    graph_to_d3_data,
    graph_to_d3_html,
    graph_to_static_svg,
)


def test_d3_data_uses_node_and_edge_labels() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node("fox", label="Animal")
    graph.add_node("crow", label="Animal")
    graph.add_edge("fox", "crow", label="The fox interacts with the crow.")

    data = graph_to_d3_data(graph)

    assert data == {
        "nodes": [
            {"id": "fox", "label": "Animal"},
            {"id": "crow", "label": "Animal"},
        ],
        "links": [{
            "source": "fox",
            "target": "crow",
            "label": "The fox interacts with the crow.",
        }],
    }


def test_d3_html_has_text_only_nodes_and_gray_thin_edges() -> None:
    graph = nx.MultiDiGraph()
    graph.add_edge("fox", "crow", label="interacts_with")

    html = graph_to_d3_html(graph)

    assert "https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js" in html
    assert '.selectAll("line")' in html
    assert '.selectAll("text")' in html
    assert 'attr("stroke", "#9aa0a6")' in html
    assert 'attr("stroke-width", 1)' in html
    assert "<circle" not in html


def test_display_graph_returns_ipython_html() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node("fox", label="Animal")

    html = display_graph(graph)

    assert "semantic-graphicalizer-" in html.data
    assert "d3.forceSimulation" in html.data
    assert html.lib == ["https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"]


def test_static_display_uses_kamada_kawai_and_text_only_svg() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node("fox", label="Animal")
    graph.add_node("action", label="Action")
    graph.add_edge("fox", "action", label="The fox performs an action.")

    svg = graph_to_static_svg(graph)
    rendered = display_graph(graph, mode="static")

    assert "Kamada-Kawai" in svg
    assert 'stroke="#9aa0a6"' in svg
    assert '<text' in svg
    assert "Animal" in svg
    assert "The fox performs an action." in svg
    assert "<circle" not in svg
    assert rendered.data.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert "Kamada-Kawai" in rendered.data


def test_display_graph_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="dynamic.*static"):
        display_graph(nx.MultiDiGraph(), mode="unknown")
