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


def _validate_charge_strength(charge_strength: float) -> float:
    if isinstance(charge_strength, bool) or not isinstance(charge_strength, (int, float)):
        raise ValueError("charge_strength must be a non-positive number")
    if charge_strength > 0:
        raise ValueError("charge_strength must be a non-positive number")
    return float(charge_strength)


def _validate_positive_number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return float(value)


def _validate_nonnegative_number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return float(value)


def _validate_stiffness(value: float, name: str = "timeline_stiffness") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a number between 0 and 1")
    return float(value)


def _validate_layout(layout: str) -> str:
    if layout not in {"auto", "force", "timeline"}:
        raise ValueError("layout must be one of 'auto', 'force', or 'timeline'")
    return layout


def _ontology_text(value: Any) -> str:
    """Normalize ontology labels while leaving source text untouched."""

    return str(value).lower()


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


def graph_to_text(
    value: nx.Graph | Any,
    *,
    node_label_attr: str = "label",
    edge_label_attr: str = "label",
) -> str:
    """Return a readable, indented text view of the graph.

    Each node with outgoing relations is followed by those relations. Incoming
    relations are not repeated, and target nodes are shown inline on the
    relation line. The relation line keeps the predicate and target ontology
    ID together.
    """

    graph = _graph_from_value(value)
    node_items = list(graph.nodes(data=True))
    ordered_nodes = sorted(
        enumerate(node_items),
        key=lambda indexed_item: (
            indexed_item[1][1].get("sequence", indexed_item[0])
            if isinstance(indexed_item[1][1].get("sequence", indexed_item[0]), (int, float))
            else indexed_item[0],
            str(indexed_item[1][0]),
        ),
    )

    lines: list[str] = []
    for _, (node_id, data) in ordered_nodes:
        if graph.is_directed():
            if graph.out_degree(node_id) == 0:
                continue
        elif graph.degree(node_id) == 0:
            continue

        node_label = _ontology_text(data.get(node_label_attr, node_id))
        mentions = data.get("mentions")
        if isinstance(mentions, (list, tuple)):
            surface_text = "; ".join(str(mention) for mention in mentions if mention)
        else:
            surface_text = str(mentions) if mentions else ""
        lines.append(f"{node_label}: {surface_text}" if surface_text else node_label)

        if graph.is_directed():
            relation_items = [
                (target_id, relation_data)
                for _, target_id, relation_data in graph.out_edges(node_id, data=True)
            ]
        else:
            relation_items = [
                (
                    target_id if source_id == node_id else source_id,
                    relation_data,
                )
                for source_id, target_id, relation_data in graph.edges(node_id, data=True)
            ]
        for target_id, relation_data in relation_items:
            predicate_value = relation_data.get("predicate")
            predicate = (
                _ontology_text(predicate_value)
                if predicate_value
                else relation_data.get(edge_label_attr) or "relation"
            )
            target_data = graph.nodes[target_id]
            target_label = _ontology_text(target_data.get(node_label_attr, target_id))
            target_mentions = target_data.get("mentions")
            if isinstance(target_mentions, (list, tuple)):
                target_text = "; ".join(str(mention) for mention in target_mentions if mention)
            else:
                target_text = str(target_mentions) if target_mentions else ""
            relation_text = f"{predicate}: {target_label}"
            category = relation_data.get("category")
            if category in {"temporal", "causal"}:
                relation_text = f"[{category}] {relation_text}"
            if target_text:
                relation_text += f": {target_text}"
            lines.append(f"    {relation_text}")
    return "\n".join(lines)


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
        primary_text = _wrap_text(_ontology_text(primary), max_width)
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
            "ontology_label": _ontology_text(data.get(node_label_attr, node_id)),
            "source_fragment": node_source(data),
            "node_type": str(data.get("node_type", "entity")),
            "proposition_kind": data.get("proposition_kind"),
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
                "predicate": _ontology_text(data.get("predicate", "")),
                "source_fragment": str(data.get(edge_label_attr, "")),
                "edge_type": str(data.get("edge_type", "semantic")),
                "category": str(data.get("category", "semantic")),
                "directed": graph.is_directed(),
            }
            for source, target, _key, data in edges
        ]
    else:
        links = [
            {
                "source": str(source),
                "target": str(target),
                "label": combined_label(data.get("predicate", ""), data.get(edge_label_attr, "")),
                "predicate": _ontology_text(data.get("predicate", "")),
                "source_fragment": str(data.get(edge_label_attr, "")),
                "edge_type": str(data.get("edge_type", "semantic")),
                "category": str(data.get("category", "semantic")),
                "directed": graph.is_directed(),
            }
            for source, target, data in graph.edges(data=True)
        ]
    return {"nodes": nodes, "links": links}


def graph_to_d3_html(
    value: nx.Graph | Any,
    *,
    width: int = 900,
    height: int = 900,
    node_label_attr: str = "label",
    edge_label_attr: str = "label",
    show_source: bool = True,
    max_width: int = 80,
    charge_strength: float = -180,
    link_distance: float = 90,
    component_spacing: float = 180,
    component_strength: float = 0.25,
    timeline_stiffness: float = 0.95,
    layout: str = "auto",
    d3_url: str = D3_CDN_URL,
) -> str:
    """Return a self-contained inline HTML fragment containing a D3 graph."""

    if width < 1 or height < 1:
        raise ValueError("width and height must be positive")
    if not isinstance(d3_url, str) or not d3_url.strip():
        raise ValueError("d3_url must be a non-empty string")
    charge_strength = _validate_charge_strength(charge_strength)
    link_distance = _validate_positive_number(link_distance, "link_distance")
    component_spacing = _validate_nonnegative_number(component_spacing, "component_spacing")
    component_strength = _validate_nonnegative_number(component_strength, "component_strength")
    timeline_stiffness = _validate_stiffness(timeline_stiffness)
    layout = _validate_layout(layout)

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
    script = _d3_script(
        payload=payload,
        width_json=width_json,
        height_json=height_json,
        charge_strength=charge_strength,
        link_distance=link_distance,
        component_spacing=component_spacing,
        component_strength=component_strength,
        timeline_stiffness=timeline_stiffness,
        layout=layout,
        root_setup=f"  const root = document.getElementById({json.dumps(root_id)});",
    )
    return f'''<div id="{root_id}" role="img" aria-label="Ontology graph"></div>
<script src="{d3_url}"></script>
<script>
{script}
</script>'''


def _d3_script(
    *,
    payload: str,
    width_json: str,
    height_json: str,
    charge_strength: float,
    link_distance: float,
    component_spacing: float,
    component_strength: float,
    timeline_stiffness: float,
    layout: str,
    root_setup: str,
) -> str:
    """Return the shared D3 program for HTML and notebook mounts."""

    return f'''(() => {{
{root_setup}
  const data = {payload};
  const width = {width_json};
  const height = {height_json};
  const linkDistance = {link_distance};
  const requestedComponentSpacing = {component_spacing};
  const componentStrength = {component_strength};
  const timelineStiffness = {timeline_stiffness};
  const layoutMode = {json.dumps(layout)};
  const svg = d3.select(root)
    .append("svg")
    .attr("viewBox", `0 0 ${{width}} ${{height}}`)
    .attr("width", "100%")
    .attr("height", height)
    .attr("role", "img")
    .attr("aria-label", "Ontology graph with ontology-relation-labelled edges");

  const arrowId = `${{root.id}}-arrow`;
  const temporalArrowId = `${{root.id}}-temporal-arrow`;
  const causalArrowId = `${{root.id}}-causal-arrow`;
  const marker = (id, fill) => svg.append("defs")
    .append("marker")
    .attr("id", id)
    .attr("viewBox", "0 -5 10 10")
    .attr("refX", 9)
    .attr("refY", 0)
    .attr("markerWidth", 6)
    .attr("markerHeight", 6)
    .attr("orient", "auto")
    .append("path")
    .attr("d", "M0,-5L10,0L0,5")
    .attr("fill", fill);
  marker(arrowId, "#9aa0a6");
  marker(temporalArrowId, "#2563eb");
  marker(causalArrowId, "#c2410c");

  svg.append("title").text("Ontology graph");
  svg.append("desc").text("A semantic narrative graph with temporal and causal relations. Scroll to zoom, drag the background to pan, drag nodes to move and pin them, and double-click a node to unpin it.");

  const viewport = svg.append("g").attr("class", "viewport");
  svg.call(d3.zoom()
    .scaleExtent([0.25, 4])
    .on("zoom", event => viewport.attr("transform", event.transform)));

  const legend = svg.append("g")
    .attr("class", "legend")
    .attr("transform", "translate(24, 24)");
  const legendItems = [
    ["semantic", "#9aa0a6", ""],
    ["temporal", "#2563eb", "6 4"],
    ["causal", "#c2410c", ""],
  ];
  legendItems.forEach((item, index) => {{
    const [label, color, dash] = item;
    const x = index * 120;
    legend.append("line")
      .attr("x1", x)
      .attr("x2", x + 24)
      .attr("y1", 0)
      .attr("y2", 0)
      .attr("stroke", color)
      .attr("stroke-width", label === "causal" ? 2.5 : 1.5)
      .attr("stroke-dasharray", dash || null);
    legend.append("text")
      .attr("x", x + 30)
      .attr("y", 4)
      .attr("font-family", "monospace")
      .attr("font-size", 11)
      .text(label);
  }});

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
          .attr("font-family", index === 0 ? "monospace" : "sans-serif")
          .attr("font-weight", index === 0 ? 600 : 400)
          .text(line);
      }});
    }});
  }};

  const link = viewport.append("g")
    .attr("aria-hidden", "true")
    .selectAll("line")
    .data(data.links)
    .join("line")
    .attr("stroke", d => d.category === "causal" ? "#c2410c" : d.category === "temporal" ? "#2563eb" : "#9aa0a6")
    .attr("stroke-width", d => d.category === "causal" ? 2.5 : d.category === "temporal" ? 1.5 : 1)
    .attr("stroke-opacity", 0.85)
    .attr("stroke-dasharray", d => d.category === "temporal" ? "6 4" : null)
    .attr("marker-end", d => d.directed
      ? `url(#${{d.category === "causal" ? causalArrowId : d.category === "temporal" ? temporalArrowId : arrowId}})`
      : null);

  const edgeLabel = viewport.append("g")
    .selectAll("text")
    .data(data.links)
    .join("text")
    .attr("fill", d => d.category === "causal" ? "#c2410c" : d.category === "temporal" ? "#2563eb" : "#6b7280")
    .attr("font-size", 11)
    .attr("text-anchor", "middle")
    .attr("dy", -4)
    .call(setMultilineText);

  const labelWidth = d => Math.max(...String(d.label).split("\\n").map(line => line.length), 1);
  const nodeRadius = d => Math.max(42, Math.min(180, labelWidth(d) * 3.8));
  const componentValues = [...new Set(data.nodes.map(d => d.component_order))].sort((a, b) => a - b);
  const compactSpacing = componentValues.length > 1
    ? Math.min(requestedComponentSpacing, Math.max(0, (width - 180) / (componentValues.length - 1)))
    : 0;
  const componentScale = d3.scalePoint()
    .domain(componentValues)
    .range([
      width / 2 - compactSpacing * (componentValues.length - 1) / 2,
      width / 2 + compactSpacing * (componentValues.length - 1) / 2,
    ])
    .padding(0);
  const componentTarget = d => {{
    const center = componentScale(d.component_order) ?? width / 2;
    const slots = Math.max(1, d.component_size - 1);
    const local = (d.component_index / slots - 0.5)
      * Math.min(140, width / Math.max(2 * componentValues.length, 1));
    return center + local;
  }};
  const propositionNodes = data.nodes.filter(d => d.node_type === "proposition");
  const sequenceValues = [...new Set(propositionNodes.map(d => d.sequence))].sort((a, b) => a - b);
  const hasPropositions = propositionNodes.length > 0;
  const useTimeline = layoutMode === "timeline" || (layoutMode === "auto" && hasPropositions);
  const hardTimeline = useTimeline && timelineStiffness >= 0.999;
  const timelineY = height * 0.52;
  const timelineScale = d3.scalePoint()
    .domain(sequenceValues)
    .range([80, Math.max(80, width - 80)])
    .padding(0.35);
  const timelineX = d => timelineScale(d.sequence) ?? width / 2;
  const timelineYTarget = d => {{
    if (d.node_type !== "proposition") return height / 2;
    if (d.proposition_kind === "state") return timelineY - 150;
    if (d.proposition_kind === "statement") return timelineY + 150;
    return timelineY;
  }};
  const isTimelineEvent = d => d.node_type === "proposition" && d.proposition_kind === "event";
  const xTarget = d => useTimeline ? timelineX(d) : componentTarget(d);
  const xStrength = d => useTimeline
    ? (isTimelineEvent(d) ? timelineStiffness : d.node_type === "proposition" ? Math.min(0.25, timelineStiffness * 0.25) : 0.08)
    : componentStrength;
  const yStrength = d => useTimeline
    ? (isTimelineEvent(d) ? timelineStiffness : d.node_type === "proposition" ? Math.min(0.25, timelineStiffness * 0.25) : 0.08)
    : 0;
  const linkStrength = d => d.category === "temporal"
    ? 0.35 + 0.65 * timelineStiffness
    : d.category === "causal" ? 0.55 : 0.65;

  const drag = d3.drag()
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
      d.fx = d.x;
      d.fy = d.y;
      d.pinned = true;
    }});

  const node = viewport.append("g")
    .selectAll("text")
    .data(data.nodes)
    .join("text")
    .attr("fill", "currentColor")
    .attr("font-size", 13)
    .attr("font-weight", 500)
    .attr("text-anchor", "middle")
    .attr("dominant-baseline", "central")
    .call(setMultilineText)
    .call(drag)
    .on("dblclick", (event, d) => {{
      event.stopPropagation();
      d.fx = null;
      d.fy = null;
      d.pinned = false;
      simulation.alpha(0.3).restart();
    }});

  const simulation = d3.forceSimulation(data.nodes)
    .force("link", d3.forceLink(data.links).id(d => d.id).distance(d => d.category === "temporal" ? Math.min(linkDistance, 90) : d.category === "causal" ? linkDistance * 1.15 : linkDistance).strength(linkStrength))
    .force("charge", d3.forceManyBody().strength({charge_strength}))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("x", d3.forceX(xTarget).strength(xStrength))
    .force("y", useTimeline ? d3.forceY(timelineYTarget).strength(yStrength) : null)
    .force("collision", d3.forceCollide().radius(nodeRadius).strength(1).iterations(4))
    .on("tick", () => {{
      if (hardTimeline) {{
        data.nodes.forEach(d => {{
          if (isTimelineEvent(d) && !d.pinned) {{
            d.x = timelineX(d);
            d.y = timelineY;
            d.vx = 0;
            d.vy = 0;
          }}
        }});
      }}
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);
      edgeLabel
        .attr("x", d => (d.source.x + d.target.x) / 2)
        .attr("y", d => (d.source.y + d.target.y) / 2)
        .each(function(d) {{
          const x = (d.source.x + d.target.x) / 2;
          d3.select(this).selectAll("tspan").attr("x", x);
        }});
      node
        .attr("x", d => d.x)
        .attr("y", d => d.y)
        .each(function(d) {{
          d3.select(this).selectAll("tspan").attr("x", d.x);
        }});
    }});
}})();'''


def graph_to_d3_javascript(
    value: nx.Graph | Any,
    *,
    width: int = 900,
    height: int = 900,
    node_label_attr: str = "label",
    edge_label_attr: str = "label",
    show_source: bool = True,
    max_width: int = 80,
    charge_strength: float = -180,
    link_distance: float = 90,
    component_spacing: float = 180,
    component_strength: float = 0.25,
    timeline_stiffness: float = 0.95,
    layout: str = "auto",
) -> str:
    """Return JavaScript that renders the graph into an IPython output area."""

    if width < 1 or height < 1:
        raise ValueError("width and height must be positive")
    charge_strength = _validate_charge_strength(charge_strength)
    link_distance = _validate_positive_number(link_distance, "link_distance")
    component_spacing = _validate_nonnegative_number(component_spacing, "component_spacing")
    component_strength = _validate_nonnegative_number(component_strength, "component_strength")
    timeline_stiffness = _validate_stiffness(timeline_stiffness)
    layout = _validate_layout(layout)
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
    return _d3_script(
        payload=payload,
        width_json=json.dumps(int(width)),
        height_json=json.dumps(int(height)),
        charge_strength=charge_strength,
        link_distance=link_distance,
        component_spacing=component_spacing,
        component_strength=component_strength,
        timeline_stiffness=timeline_stiffness,
        layout=layout,
        root_setup=f"""  const root = document.createElement("div");
  root.id = {json.dumps(root_id)};
  root.setAttribute("role", "img");
  root.setAttribute("aria-label", "Ontology graph");
  element.appendChild(root);""",
    )





def graph_to_d3_iframe(
    value: nx.Graph | Any,
    *,
    width: int = 900,
    height: int = 900,
    node_label_attr: str = "label",
    edge_label_attr: str = "label",
    show_source: bool = True,
    max_width: int = 80,
    charge_strength: float = -180,
    link_distance: float = 90,
    component_spacing: float = 180,
    component_strength: float = 0.25,
    timeline_stiffness: float = 0.95,
    layout: str = "auto",
    d3_url: str = D3_CDN_URL,
) -> str:
    """Return an HTML iframe that runs the D3 graph in notebook frontends."""

    if width < 1 or height < 1:
        raise ValueError("width and height must be positive")
    if not isinstance(d3_url, str) or not d3_url.strip():
        raise ValueError("d3_url must be a non-empty string")
    charge_strength = _validate_charge_strength(charge_strength)
    link_distance = _validate_positive_number(link_distance, "link_distance")
    component_spacing = _validate_nonnegative_number(component_spacing, "component_spacing")
    component_strength = _validate_nonnegative_number(component_strength, "component_strength")
    timeline_stiffness = _validate_stiffness(timeline_stiffness)
    layout = _validate_layout(layout)

    document = _graph_to_d3_document(
        value,
        width=width,
        height=height,
        node_label_attr=node_label_attr,
        edge_label_attr=edge_label_attr,
        show_source=show_source,
        max_width=max_width,
        charge_strength=charge_strength,
        link_distance=link_distance,
        component_spacing=component_spacing,
        component_strength=component_strength,
        timeline_stiffness=timeline_stiffness,
        layout=layout,
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
    charge_strength: float,
    link_distance: float,
    component_spacing: float,
    component_strength: float,
    timeline_stiffness: float,
    layout: str,
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
    charge_strength=charge_strength,
    link_distance=link_distance,
    component_spacing=component_spacing,
    component_strength=component_strength,
    timeline_stiffness=timeline_stiffness,
    layout=layout,
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
    layout: str = "auto",
) -> str:
    """Return a deterministic SVG using timeline or Kamada-Kawai layout."""

    if width < 1 or height < 1:
        raise ValueError("width and height must be positive")
    layout = _validate_layout(layout)

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
    use_timeline = layout == "timeline" or (
        layout == "auto"
        and any(item["node_type"] == "proposition" for item in display_data["nodes"])
    )
    component_count = max(
        (int(item["component_order"]) for item in display_data["nodes"]),
        default=-1,
    ) + 1
    margin = 48
    usable_width = max(width - 2 * margin, 1)
    usable_height = max(height - 2 * margin, 1)
    proposition_sequences = sorted(
        item["sequence"]
        for item in display_data["nodes"]
        if item["node_type"] == "proposition"
    )

    def point(node_id: Any) -> tuple[float, float]:
        node_data = display_nodes[str(node_id)]
        if use_timeline and node_data["node_type"] == "proposition":
            sequence_index = proposition_sequences.index(node_data["sequence"])
            x = margin + (
                sequence_index / max(len(proposition_sequences) - 1, 1)
            ) * usable_width
            timeline_y = margin + usable_height * 0.52
            kind_offset = {
                "event": 0,
                "state": -120,
                "statement": 120,
            }.get(node_data.get("proposition_kind"), 0)
            return x, timeline_y + kind_offset
        x, y = positions[node_id]
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
        tspans: list[str] = []
        for index, line in enumerate(lines):
            ontology_attributes = ' font-family="monospace" font-weight="600"' if index == 0 else ""
            tspans.append(
                f'<tspan x="{x:.2f}" dy="{(first_dy if index == 0 else line_height):.2f}"'
                f"{ontology_attributes}>{xml_text(line)}</tspan>"
            )
        return (
            f'<text x="{x:.2f}" y="{y:.2f}" {attributes}>'
            f"{''.join(tspans)}</text>"
        )

    elements: list[str] = []
    if graph.is_directed():
        elements.append(
            '<defs>'
            '<marker id="semantic-graphicalizer-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">'
            '<path d="M0,0 L8,4 L0,8 z" fill="#9aa0a6" /></marker>'
            '<marker id="semantic-graphicalizer-temporal-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">'
            '<path d="M0,0 L8,4 L0,8 z" fill="#2563eb" /></marker>'
            '<marker id="semantic-graphicalizer-causal-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">'
            '<path d="M0,0 L8,4 L0,8 z" fill="#c2410c" /></marker>'
            '</defs>'
        )

    for edge_index, (source, target, _data) in enumerate(edge_items):
        x1, y1 = point(source)
        x2, y2 = point(target)
        category = display_links[edge_index].get("category", "semantic")
        stroke = {"causal": "#c2410c", "temporal": "#2563eb"}.get(category, "#9aa0a6")
        stroke_width = {"causal": 2.5, "temporal": 1.5}.get(category, 1)
        dash = ' stroke-dasharray="6 4"' if category == "temporal" else ""
        marker_id = {
            "causal": "semantic-graphicalizer-causal-arrow",
            "temporal": "semantic-graphicalizer-temporal-arrow",
        }.get(category, "semantic-graphicalizer-arrow")
        marker = f' marker-end="url(#{marker_id})"' if graph.is_directed() else ""
        elements.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}" stroke-opacity="0.85"{dash}{marker} />'
        )
        label = display_links[edge_index]["label"]
        if label:
            elements.append(svg_label(
                (x1 + x2) / 2,
                (y1 + y2) / 2 - 4,
                label,
                font_size=11,
                attributes=f'fill="{stroke}" text-anchor="middle" font-family="sans-serif"',
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

    legend_y = 24
    for index, (label, color, width_value, dash) in enumerate((
        ("semantic", "#9aa0a6", 1, ""),
        ("temporal", "#2563eb", 1.5, "6 4"),
        ("causal", "#c2410c", 2.5, ""),
    )):
        x = 24 + index * 120
        dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
        elements.append(
            f'<line x1="{x}" y1="{legend_y}" x2="{x + 24}" y2="{legend_y}" '
            f'stroke="{color}" stroke-width="{width_value}"'
            f'{dash_attribute} />'
        )
        elements.append(
            f'<text x="{x + 30}" y="{legend_y + 4}" font-family="monospace" '
            f'font-size="11" fill="currentColor">{xml_text(label)}</text>'
        )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="{int(height)}" '
        f'viewBox="0 0 {int(width)} {int(height)}" role="img" aria-label="Ontology graph">'
        '<title>Ontology graph</title>'
        '<desc>A deterministic narrative or Kamada-Kawai graph with ontology terms and '
        'surface mentions as text-only nodes, and ontology relation IDs with proposition '
        'fragments as edge labels.</desc>'
        + "".join(elements)
        + "</svg>"
    )


def display_graph(value: nx.Graph | Any, *, mode: str = "dynamic", **kwargs: Any) -> Any:
    """Return a dynamic D3, static SVG, or indented text visualization."""

    if mode not in {"dynamic", "static", "text"}:
        raise ValueError("mode must be one of 'dynamic', 'static', or 'text'")

    if mode == "text":
        text = graph_to_text(value, **kwargs)
        try:
            from IPython.display import Pretty
        except ImportError:  # pragma: no cover - depends on environment
            return text
        return Pretty(text)

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
    height = int(kwargs.get("height", 900))
    document = _graph_to_d3_document(
        value,
        width=width,
        height=height,
        node_label_attr=kwargs.get("node_label_attr", "label"),
        edge_label_attr=kwargs.get("edge_label_attr", "label"),
        show_source=kwargs.get("show_source", True),
        max_width=kwargs.get("max_width", 80),
        charge_strength=kwargs.get("charge_strength", -180),
        link_distance=kwargs.get("link_distance", 90),
        component_spacing=kwargs.get("component_spacing", 180),
        component_strength=kwargs.get("component_strength", 0.25),
        timeline_stiffness=kwargs.get("timeline_stiffness", 0.95),
        layout=kwargs.get("layout", "auto"),
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
    "graph_to_text",
]
