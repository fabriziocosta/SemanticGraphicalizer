"""Inline dynamic and static rendering for NetworkX graphs."""

from __future__ import annotations

import json
from html import escape
from typing import Any
from uuid import uuid4

import networkx as nx


D3_CDN_URL = "https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"
_NETWORKX_GRAPH_TYPES = (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)


def _graph_from_value(value: Any) -> nx.Graph:
    if isinstance(value, _NETWORKX_GRAPH_TYPES):
        return value
    graph = getattr(value, "graph", None)
    if not isinstance(graph, _NETWORKX_GRAPH_TYPES):
        raise TypeError("value must be a NetworkX graph or a DocumentTrace")
    return graph


def graph_to_d3_data(
    value: nx.Graph | Any,
    *,
    node_label_attr: str = "label",
    edge_label_attr: str = "label",
    show_source: bool = True,
) -> dict[str, list[dict[str, str]]]:
    """Convert a graph or trace into D3 data with ontology and source labels."""

    graph = _graph_from_value(value)

    def combined_label(primary: Any, source: Any) -> str:
        primary_text = str(primary)
        source_text = str(source) if source else ""
        if not show_source or not source_text:
            return primary_text
        return f"{primary_text} — {source_text}"

    def node_source(data: dict[str, Any]) -> str:
        mentions = data.get("mentions")
        if isinstance(mentions, (list, tuple)):
            return "; ".join(str(mention) for mention in mentions if mention)
        return str(mentions) if mentions else ""

    nodes = [
        {
            "id": str(node_id),
            "label": combined_label(data.get(node_label_attr, node_id), node_source(data)),
            "ontology_label": str(data.get(node_label_attr, node_id)),
            "source_fragment": node_source(data),
        }
        for node_id, data in graph.nodes(data=True)
    ]
    if graph.is_multigraph():
        edges = graph.edges(data=True, keys=True)
        links = [
            {
                "source": str(source),
                "target": str(target),
                "label": combined_label(data.get("predicate", ""), data.get(edge_label_attr, "")),
                "predicate": str(data.get("predicate", "")),
                "source_fragment": str(data.get(edge_label_attr, "")),
            }
            for source, target, _key, data in edges
        ]
    else:
        links = [
            {
                "source": str(source),
                "target": str(target),
                "label": combined_label(data.get("predicate", ""), data.get(edge_label_attr, "")),
                "predicate": str(data.get("predicate", "")),
                "source_fragment": str(data.get(edge_label_attr, "")),
            }
            for source, target, data in graph.edges(data=True)
        ]
    return {"nodes": nodes, "links": links}


def graph_to_d3_html(
    value: nx.Graph | Any,
    *,
    width: int = 900,
    height: int = 600,
    node_label_attr: str = "label",
    edge_label_attr: str = "label",
    show_source: bool = True,
    d3_url: str = D3_CDN_URL,
) -> str:
    """Return a self-contained inline HTML fragment containing a D3 graph."""

    if width < 1 or height < 1:
        raise ValueError("width and height must be positive")
    if not isinstance(d3_url, str) or not d3_url.strip():
        raise ValueError("d3_url must be a non-empty string")

    root_id = f"semantic-graphicalizer-{uuid4().hex}"
    payload = json.dumps(
        graph_to_d3_data(
            value,
            node_label_attr=node_label_attr,
            edge_label_attr=edge_label_attr,
            show_source=show_source,
        ),
        ensure_ascii=False,
    ).replace("<", "\\u003c")
    width_json = json.dumps(int(width))
    height_json = json.dumps(int(height))

    return f'''<div id="{root_id}" role="img" aria-label="Ontology graph"></div>
<script src="{d3_url}"></script>
<script>
(() => {{
  const root = document.getElementById({json.dumps(root_id)});
  const data = {payload};
  const width = {width_json};
  const height = {height_json};
  const svg = d3.select(root)
    .append("svg")
    .attr("viewBox", `0 0 ${{width}} ${{height}}`)
    .attr("width", "100%")
    .attr("height", height)
    .attr("role", "img")
    .attr("aria-label", "Ontology graph with ontology-relation-labelled edges");

  svg.append("title").text("Ontology graph");
  svg.append("desc").text("A force-directed graph with ontology terms as text-only nodes and ontology relation IDs as edge labels.");

  const link = svg.append("g")
    .attr("aria-hidden", "true")
    .selectAll("line")
    .data(data.links)
    .join("line")
    .attr("stroke", "#9aa0a6")
    .attr("stroke-width", 1)
    .attr("stroke-opacity", 0.85);

  const edgeLabel = svg.append("g")
    .selectAll("text")
    .data(data.links)
    .join("text")
    .attr("fill", "#6b7280")
    .attr("font-size", 11)
    .attr("text-anchor", "middle")
    .attr("dy", -4)
    .text(d => d.label);

  const node = svg.append("g")
    .selectAll("text")
    .data(data.nodes)
    .join("text")
    .attr("fill", "currentColor")
    .attr("font-size", 13)
    .attr("font-weight", 500)
    .attr("text-anchor", "middle")
    .attr("dominant-baseline", "central")
    .text(d => d.label)
    .call(d3.drag()
      .on("start", (event, d) => {{
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      }})
      .on("drag", (event, d) => {{
        d.fx = event.x;
        d.fy = event.y;
      }})
      .on("end", (event, d) => {{
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      }}));

  const simulation = d3.forceSimulation(data.nodes)
    .force("link", d3.forceLink(data.links).id(d => d.id).distance(150).strength(0.65))
    .force("charge", d3.forceManyBody().strength(-280))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collision", d3.forceCollide().radius(48))
    .on("tick", () => {{
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);
      edgeLabel
        .attr("x", d => (d.source.x + d.target.x) / 2)
        .attr("y", d => (d.source.y + d.target.y) / 2);
      node
        .attr("x", d => d.x)
        .attr("y", d => d.y);
    }});
}})();
</script>'''


def graph_to_d3_javascript(
    value: nx.Graph | Any,
    *,
    width: int = 900,
    height: int = 600,
    node_label_attr: str = "label",
    edge_label_attr: str = "label",
    show_source: bool = True,
) -> str:
    """Return JavaScript that renders the graph into an IPython output area."""

    if width < 1 or height < 1:
        raise ValueError("width and height must be positive")
    root_id = f"semantic-graphicalizer-{uuid4().hex}"
    payload = json.dumps(
        graph_to_d3_data(
            value,
            node_label_attr=node_label_attr,
            edge_label_attr=edge_label_attr,
            show_source=show_source,
        ),
        ensure_ascii=False,
    ).replace("<", "\\u003c")
    width_json = json.dumps(int(width))
    height_json = json.dumps(int(height))

    return f'''(() => {{
  const root = document.createElement("div");
  root.id = {json.dumps(root_id)};
  root.setAttribute("role", "img");
  root.setAttribute("aria-label", "Ontology graph");
  element.appendChild(root);

  const data = {payload};
  const width = {width_json};
  const height = {height_json};
  const svg = d3.select(root)
    .append("svg")
    .attr("viewBox", `0 0 ${{width}} ${{height}}`)
    .attr("width", "100%")
    .attr("height", height)
    .attr("role", "img")
    .attr("aria-label", "Ontology graph with ontology-relation-labelled edges");

  svg.append("title").text("Ontology graph");
  svg.append("desc").text("A force-directed graph with ontology terms as text-only nodes and ontology relation IDs as edge labels.");

  const link = svg.append("g")
    .attr("aria-hidden", "true")
    .selectAll("line")
    .data(data.links)
    .join("line")
    .attr("stroke", "#9aa0a6")
    .attr("stroke-width", 1)
    .attr("stroke-opacity", 0.85);

  const edgeLabel = svg.append("g")
    .selectAll("text")
    .data(data.links)
    .join("text")
    .attr("fill", "#6b7280")
    .attr("font-size", 11)
    .attr("text-anchor", "middle")
    .attr("dy", -4)
    .text(d => d.label);

  let simulation;
  const node = svg.append("g")
    .selectAll("text")
    .data(data.nodes)
    .join("text")
    .attr("fill", "currentColor")
    .attr("font-size", 13)
    .attr("font-weight", 500)
    .attr("text-anchor", "middle")
    .attr("dominant-baseline", "central")
    .text(d => d.label)
    .call(d3.drag()
      .on("start", (event, d) => {{
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      }})
      .on("drag", (event, d) => {{
        d.fx = event.x;
        d.fy = event.y;
      }})
      .on("end", (event, d) => {{
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      }}));

  simulation = d3.forceSimulation(data.nodes)
    .force("link", d3.forceLink(data.links).id(d => d.id).distance(150).strength(0.65))
    .force("charge", d3.forceManyBody().strength(-280))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collision", d3.forceCollide().radius(48))
    .on("tick", () => {{
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);
      edgeLabel
        .attr("x", d => (d.source.x + d.target.x) / 2)
        .attr("y", d => (d.source.y + d.target.y) / 2);
      node
        .attr("x", d => d.x)
        .attr("y", d => d.y);
    }});
}})();'''


def graph_to_static_svg(
    value: nx.Graph | Any,
    *,
    width: int = 900,
    height: int = 600,
    node_label_attr: str = "label",
    edge_label_attr: str = "label",
    show_source: bool = True,
) -> str:
    """Return an SVG using NetworkX's deterministic Kamada-Kawai layout."""

    if width < 1 or height < 1:
        raise ValueError("width and height must be positive")

    graph = _graph_from_value(value)
    display_data = graph_to_d3_data(
        graph,
        node_label_attr=node_label_attr,
        edge_label_attr=edge_label_attr,
        show_source=show_source,
    )
    node_items = list(graph.nodes(data=True))
    positions = nx.kamada_kawai_layout(graph, weight=None) if node_items else {}
    margin = 48
    usable_width = max(width - 2 * margin, 1)
    usable_height = max(height - 2 * margin, 1)

    def point(node_id: Any) -> tuple[float, float]:
        x, y = positions[node_id]
        return (
            margin + (float(x) + 1.0) * usable_width / 2.0,
            margin + (1.0 - (float(y) + 1.0) / 2.0) * usable_height,
        )

    if graph.is_multigraph():
        edge_items = [(source, target, data) for source, target, _key, data in graph.edges(data=True, keys=True)]
    else:
        edge_items = list(graph.edges(data=True))

    def xml_text(value: Any) -> str:
        return escape(str(value), quote=True)

    elements: list[str] = []
    if graph.is_directed():
        elements.append(
            '<defs><marker id="semantic-graphicalizer-arrow" markerWidth="8" '
            'markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">'
            '<path d="M0,0 L8,4 L0,8 z" fill="#9aa0a6" /></marker></defs>'
        )

    for source, target, data in edge_items:
        x1, y1 = point(source)
        x2, y2 = point(target)
        marker = ' marker-end="url(#semantic-graphicalizer-arrow)"' if graph.is_directed() else ""
        elements.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="#9aa0a6" stroke-width="1" stroke-opacity="0.85"{marker} />'
        )
        label = data.get(edge_label_attr, "")
        if label:
            elements.append(
                f'<text x="{(x1 + x2) / 2:.2f}" y="{(y1 + y2) / 2 - 4:.2f}" '
                'fill="#6b7280" font-size="11" text-anchor="middle" '
                'font-family="sans-serif">'
                f"{xml_text(label)}</text>"
            )

    for node_id, data in node_items:
        x, y = point(node_id)
        label = data.get(node_label_attr, node_id)
        elements.append(
            f'<text x="{x:.2f}" y="{y:.2f}" fill="currentColor" font-size="13" '
            'font-weight="500" text-anchor="middle" dominant-baseline="central" '
            'font-family="sans-serif">'
            f"{xml_text(label)}</text>"
        )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="{int(height)}" '
        f'viewBox="0 0 {int(width)} {int(height)}" role="img" aria-label="Ontology graph">'
        '<title>Ontology graph</title>'
        '<desc>A deterministic Kamada-Kawai graph with ontology terms as text-only nodes '
        'and ontology relation IDs as edge labels.</desc>'
        + "".join(elements)
        + "</svg>"
    )


def display_graph(value: nx.Graph | Any, *, mode: str = "dynamic", **kwargs: Any) -> Any:
    """Return a dynamic D3 or static Kamada-Kawai notebook visualization."""

    if mode not in {"dynamic", "static"}:
        raise ValueError("mode must be either 'dynamic' or 'static'")

    if mode == "static":
        markup = graph_to_static_svg(value, **kwargs)
        try:
            from IPython.display import SVG
        except ImportError:  # pragma: no cover - depends on environment
            return markup
        return SVG(markup)

    try:
        from IPython.display import Javascript
    except ImportError:  # pragma: no cover - IPython is an optional notebook dependency
        return graph_to_d3_html(value, **kwargs)
    d3_url = kwargs.pop("d3_url", D3_CDN_URL)
    return Javascript(graph_to_d3_javascript(value, **kwargs), lib=[d3_url])


__all__ = [
    "D3_CDN_URL",
    "display_graph",
    "graph_to_d3_data",
    "graph_to_d3_html",
    "graph_to_d3_javascript",
    "graph_to_static_svg",
]
