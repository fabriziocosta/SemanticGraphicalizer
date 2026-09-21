"""Inline dynamic and static rendering for NetworkX graphs."""

from __future__ import annotations

import json
from html import escape
from textwrap import wrap
from typing import Any
from urllib.parse import quote
from uuid import uuid4

import networkx as nx


D3_CDN_URL = "https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"
_NETWORKX_GRAPH_TYPES = (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)


def _validate_max_width(max_width: int) -> int:
    if isinstance(max_width, bool) or not isinstance(max_width, int) or max_width < 1:
        raise ValueError("max_width must be a positive integer")
    return max_width


def _wrap_text(value: Any, max_width: int) -> str:
    lines: list[str] = []
    for paragraph in str(value).splitlines() or [""]:
        lines.extend(
            wrap(
                paragraph,
                width=max_width,
                break_long_words=False,
                break_on_hyphens=False,
            )
            or [""]
        )
    return "\n".join(lines)


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
    max_width: int = 80,
) -> dict[str, list[dict[str, Any]]]:
    """Convert a graph or trace into D3 data with wrapped multiline labels."""

    graph = _graph_from_value(value)
    max_width = _validate_max_width(max_width)

    def combined_label(primary: Any, source: Any) -> str:
        primary_text = _wrap_text(primary, max_width)
        source_text = _wrap_text(source, max_width) if source else ""
        if not show_source or not source_text:
            return primary_text
        return f"{primary_text}\n{source_text}"

    def node_source(data: dict[str, Any]) -> str:
        mentions = data.get("mentions")
        if isinstance(mentions, (list, tuple)):
            return "; ".join(str(mention) for mention in mentions if mention)
        return str(mentions) if mentions else ""

    node_items = list(graph.nodes(data=True))
    sequence_by_node = {
        node_id: data.get("sequence", index)
        if isinstance(data.get("sequence", index), (int, float))
        else index
        for index, (node_id, data) in enumerate(node_items)
    }
    components = list(nx.connected_components(graph.to_undirected()))
    components.sort(key=lambda component: min(sequence_by_node[node] for node in component))
    component_info: dict[Any, tuple[int, int, int]] = {}
    for component_order, component in enumerate(components):
        ordered_nodes = sorted(component, key=lambda node: (sequence_by_node[node], str(node)))
        for component_index, node_id in enumerate(ordered_nodes):
            component_info[node_id] = (component_order, component_index, len(ordered_nodes))

    nodes = [
        {
            "id": str(node_id),
            "label": combined_label(data.get(node_label_attr, node_id), node_source(data)),
            "ontology_label": str(data.get(node_label_attr, node_id)),
            "source_fragment": node_source(data),
            "sequence": sequence_by_node[node_id],
            "component_order": component_info[node_id][0],
            "component_index": component_info[node_id][1],
            "component_size": component_info[node_id][2],
        }
        for node_id, data in node_items
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
    max_width: int = 80,
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
            max_width=max_width,
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
  svg.append("desc").text("A force-directed graph with ontology terms and surface mentions as text-only nodes, and ontology relation IDs with proposition fragments as edge labels.");

  const setMultilineText = selection => {{
    selection.each(function(d) {{
      const lines = String(d.label).split("\\n");
      const lineHeight = 1.2;
      const text = d3.select(this);
      text.selectAll("*").remove();
      lines.forEach((line, index) => {{
        text.append("tspan")
          .attr("dy", index === 0
            ? `${{-(lines.length - 1) * lineHeight / 2}}em`
            : `${{lineHeight}}em`)
          .text(line);
      }});
    }});
  }};

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
    .call(setMultilineText);

  const labelWidth = d => Math.max(...String(d.label).split("\\n").map(line => line.length), 1);
  const nodeRadius = d => Math.max(42, Math.min(180, labelWidth(d) * 3.8));
  const componentValues = [...new Set(data.nodes.map(d => d.component_order))].sort((a, b) => a - b);
  const componentScale = d3.scalePoint()
    .domain(componentValues)
    .range([90, width - 90])
    .padding(0.5);
  const componentTarget = d => {{
    const center = componentScale(d.component_order) ?? width / 2;
    const slots = Math.max(1, d.component_size - 1);
    const local = (d.component_index / slots - 0.5)
      * Math.min(140, width / Math.max(2 * componentValues.length, 1));
    return center + local;
  }};

  const node = svg.append("g")
    .selectAll("text")
    .data(data.nodes)
    .join("text")
    .attr("fill", "currentColor")
    .attr("font-size", 13)
    .attr("font-weight", 500)
    .attr("text-anchor", "middle")
    .attr("dominant-baseline", "central")
    .call(setMultilineText)
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
    .force("charge", d3.forceManyBody().strength(-420))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("component-order", d3.forceX(componentTarget).strength(0.8))
    .force("collision", d3.forceCollide().radius(nodeRadius).strength(1).iterations(4))
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
    max_width: int = 80,
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
            max_width=max_width,
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
  svg.append("desc").text("A force-directed graph with ontology terms and surface mentions as text-only nodes, and ontology relation IDs with proposition fragments as edge labels.");

  const setMultilineText = selection => {{
    selection.each(function(d) {{
      const lines = String(d.label).split("\\n");
      const lineHeight = 1.2;
      const text = d3.select(this);
      text.selectAll("*").remove();
      lines.forEach((line, index) => {{
        text.append("tspan")
          .attr("dy", index === 0
            ? `${{-(lines.length - 1) * lineHeight / 2}}em`
            : `${{lineHeight}}em`)
          .text(line);
      }});
    }});
  }};

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
    .call(setMultilineText);

  const labelWidth = d => Math.max(...String(d.label).split("\\n").map(line => line.length), 1);
  const nodeRadius = d => Math.max(42, Math.min(180, labelWidth(d) * 3.8));
  const componentValues = [...new Set(data.nodes.map(d => d.component_order))].sort((a, b) => a - b);
  const componentScale = d3.scalePoint()
    .domain(componentValues)
    .range([90, width - 90])
    .padding(0.5);
  const componentTarget = d => {{
    const center = componentScale(d.component_order) ?? width / 2;
    const slots = Math.max(1, d.component_size - 1);
    const local = (d.component_index / slots - 0.5)
      * Math.min(140, width / Math.max(2 * componentValues.length, 1));
    return center + local;
  }};

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
    .call(setMultilineText)
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
    .force("charge", d3.forceManyBody().strength(-420))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("component-order", d3.forceX(componentTarget).strength(0.8))
    .force("collision", d3.forceCollide().radius(nodeRadius).strength(1).iterations(4))
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


def graph_to_d3_iframe(
    value: nx.Graph | Any,
    *,
    width: int = 900,
    height: int = 600,
    node_label_attr: str = "label",
    edge_label_attr: str = "label",
    show_source: bool = True,
    max_width: int = 80,
    d3_url: str = D3_CDN_URL,
) -> str:
    """Return an HTML iframe that runs the D3 graph in notebook frontends."""

    if width < 1 or height < 1:
        raise ValueError("width and height must be positive")
    if not isinstance(d3_url, str) or not d3_url.strip():
        raise ValueError("d3_url must be a non-empty string")

    document = _graph_to_d3_document(
        value,
        width=width,
        height=height,
        node_label_attr=node_label_attr,
        edge_label_attr=edge_label_attr,
        show_source=show_source,
        max_width=max_width,
        d3_url=d3_url,
    )
    return (
        f'<iframe title="Ontology graph" width="100%" height="{int(height)}" '
        'style="border:0; display:block;" sandbox="allow-scripts allow-same-origin" '
        f'srcdoc="{escape(document, quote=True)}"></iframe>'
    )


def _graph_to_d3_document(
    value: nx.Graph | Any,
    *,
    width: int,
    height: int,
    node_label_attr: str,
    edge_label_attr: str,
    show_source: bool,
    max_width: int,
    d3_url: str,
) -> str:
    """Build the document used by notebook iframe renderers."""

    return f'''<!doctype html>
<html>
<head><meta charset="utf-8"><style>html, body {{ margin: 0; padding: 0; overflow: hidden; }}</style></head>
<body>{graph_to_d3_html(
    value,
    width=width,
    height=height,
    node_label_attr=node_label_attr,
    edge_label_attr=edge_label_attr,
    show_source=show_source,
    max_width=max_width,
    d3_url=d3_url,
)}</body>
</html>'''


def graph_to_static_svg(
    value: nx.Graph | Any,
    *,
    width: int = 900,
    height: int = 600,
    node_label_attr: str = "label",
    edge_label_attr: str = "label",
    show_source: bool = True,
    max_width: int = 80,
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
        max_width=max_width,
    )
    node_items = list(graph.nodes(data=True))
    positions = nx.kamada_kawai_layout(graph, weight=None) if node_items else {}
    display_nodes = {item["id"]: item for item in display_data["nodes"]}
    display_links = display_data["links"]
    component_count = max(
        (int(item["component_order"]) for item in display_data["nodes"]),
        default=-1,
    ) + 1
    margin = 48
    usable_width = max(width - 2 * margin, 1)
    usable_height = max(height - 2 * margin, 1)

    def point(node_id: Any) -> tuple[float, float]:
        x, y = positions[node_id]
        node_data = display_nodes[str(node_id)]
        band_width = usable_width / max(component_count, 1)
        slots = max(1, int(node_data["component_size"]) - 1)
        local_x = (
            int(node_data["component_index"]) / slots - 0.5
        ) * min(140.0, band_width * 0.7)
        return (
            margin + (int(node_data["component_order"]) + 0.5) * band_width + local_x,
            margin + (1.0 - (float(y) + 1.0) / 2.0) * usable_height,
        )

    if graph.is_multigraph():
        edge_items = [(source, target, data) for source, target, _key, data in graph.edges(data=True, keys=True)]
    else:
        edge_items = list(graph.edges(data=True))

    def xml_text(value: Any) -> str:
        return escape(str(value), quote=True)

    def svg_label(
        x: float,
        y: float,
        label: str,
        *,
        font_size: float,
        attributes: str,
    ) -> str:
        lines = label.split("\n") or [""]
        line_height = font_size * 1.2
        first_dy = -((len(lines) - 1) * line_height) / 2
        tspans = "".join(
            f'<tspan x="{x:.2f}" dy="{(first_dy if index == 0 else line_height):.2f}">'
            f"{xml_text(line)}</tspan>"
            for index, line in enumerate(lines)
        )
        return (
            f'<text x="{x:.2f}" y="{y:.2f}" {attributes}>'
            f"{tspans}</text>"
        )

    elements: list[str] = []
    if graph.is_directed():
        elements.append(
            '<defs><marker id="semantic-graphicalizer-arrow" markerWidth="8" '
            'markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">'
            '<path d="M0,0 L8,4 L0,8 z" fill="#9aa0a6" /></marker></defs>'
        )

    for edge_index, (source, target, _data) in enumerate(edge_items):
        x1, y1 = point(source)
        x2, y2 = point(target)
        marker = ' marker-end="url(#semantic-graphicalizer-arrow)"' if graph.is_directed() else ""
        elements.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="#9aa0a6" stroke-width="1" stroke-opacity="0.85"{marker} />'
        )
        label = display_links[edge_index]["label"]
        if label:
            elements.append(svg_label(
                (x1 + x2) / 2,
                (y1 + y2) / 2 - 4,
                label,
                font_size=11,
                attributes='fill="#6b7280" text-anchor="middle" font-family="sans-serif"',
            )
            )

    for node_id, _data in node_items:
        x, y = point(node_id)
        label = display_nodes[str(node_id)]["label"]
        elements.append(svg_label(
            x,
            y,
            label,
            font_size=13,
            attributes='fill="currentColor" font-weight="500" text-anchor="middle" '
            'dominant-baseline="central" font-family="sans-serif"',
        )
        )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="{int(height)}" '
        f'viewBox="0 0 {int(width)} {int(height)}" role="img" aria-label="Ontology graph">'
        '<title>Ontology graph</title>'
        '<desc>A deterministic Kamada-Kawai graph with ontology terms and surface mentions '
        'as text-only nodes, and ontology relation IDs with proposition fragments as edge labels.</desc>'
        + "".join(elements)
        + "</svg>"
    )


def display_graph(value: nx.Graph | Any, *, mode: str = "dynamic", **kwargs: Any) -> Any:
    """Return a dynamic D3 or static Kamada-Kawai visualization."""

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
        from IPython.display import IFrame
    except ImportError:  # pragma: no cover - IPython is an optional notebook dependency
        return graph_to_d3_html(value, **kwargs)

    width = int(kwargs.get("width", 900))
    height = int(kwargs.get("height", 600))
    document = _graph_to_d3_document(
        value,
        width=width,
        height=height,
        node_label_attr=kwargs.get("node_label_attr", "label"),
        edge_label_attr=kwargs.get("edge_label_attr", "label"),
        show_source=kwargs.get("show_source", True),
        max_width=kwargs.get("max_width", 80),
        d3_url=kwargs.get("d3_url", D3_CDN_URL),
    )
    data_url = "data:text/html;charset=utf-8," + quote(document, safe="")
    rendered = IFrame(
        data_url,
        width="100%",
        height=height,
        extras=[
            'style="border:0; display:block;"',
            'sandbox="allow-scripts allow-same-origin"',
        ],
    )
    return rendered


__all__ = [
    "D3_CDN_URL",
    "display_graph",
    "graph_to_d3_data",
    "graph_to_d3_html",
    "graph_to_d3_iframe",
    "graph_to_d3_javascript",
    "graph_to_static_svg",
]
