import networkx as nx

from semantic_graphicalizer import display_graph, graph_to_d3_data, graph_to_d3_html


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
