"""Inline dynamic and static rendering for NetworkX graphs."""

from __future__ import annotations

import json
from collections import defaultdict
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
    raise TypeError("value must be a NetworkX graph")


def _binary_projection_links(graph: nx.Graph) -> list[dict[str, Any]]:
    """Derive display-only links from binary reified relation nodes."""

    if graph.graph.get("projection") == "binary_relations":
        return []
    if not graph.is_directed() or not graph.is_multigraph():
        return []

    def source_fragment(data: dict[str, Any]) -> str:
        """Find the relation's own evidence without conflating it with arguments."""

        direct = data.get("source_text")
        if direct:
            return str(direct)
        attributes = data.get("attributes")
        if isinstance(attributes, dict):
            direct = attributes.get("source_text")
            if direct:
                return str(direct)
            provenance = attributes.get("provenance")
        else:
            provenance = data.get("provenance")
        if isinstance(provenance, dict):
            direct = provenance.get("source_text")
            if direct:
                return str(direct)
        if isinstance(provenance, (list, tuple)):
            for item in provenance:
                if isinstance(item, dict) and item.get("source_text"):
                    return str(item["source_text"])
        return ""

    derived: list[dict[str, Any]] = []
    for relation_id, node_data in graph.nodes(data=True):
        category = node_data.get("relation_category")
        projection_roles = node_data.get("projection_roles")
        relation_name = node_data.get("relation")
        if not relation_name:
            continue
        if not isinstance(projection_roles, (list, tuple)) or len(projection_roles) != 2:
            continue
        source_role, target_role = projection_roles
        source_targets = [
            target
            for _source, target, edge_data in graph.out_edges(relation_id, data=True)
            if edge_data.get("role") == source_role
        ]
        target_targets = [
            target
            for _source, target, edge_data in graph.out_edges(relation_id, data=True)
            if edge_data.get("role") == target_role
        ]
        if len(source_targets) != 1 or len(target_targets) != 1:
            continue
        derived.append({
            "source": str(source_targets[0]),
            "target": str(target_targets[0]),
            "label": str(relation_name),
            "predicate": str(relation_name),
            "source_fragment": source_fragment(node_data),
            "edge_type": "derived_projection" if category in {"temporal", "causal"} else "projection",
            "category": category or "semantic",
            "directed": True,
            "relation_entity_id": str(relation_id),
        })
    return derived


def _display_projection_view(
    graph: nx.Graph,
    *,
    show_derived_links: bool,
) -> tuple[set[Any], list[dict[str, Any]]]:
    """Return visible nodes and links for the reified display.

    Standalone binary relation nodes collapse into direct links. A binary
    relation that is itself used as an argument remains visible so recursive
    assertions retain an identifiable endpoint.
    """

    if not show_derived_links:
        return set(graph.nodes), []
    binary_links = _binary_projection_links(graph)
    if not binary_links:
        return set(graph.nodes), []
    by_relation = {link["relation_entity_id"]: link for link in binary_links}
    collapsed = {
        relation_id
        for relation_id in by_relation
        if relation_id in graph and graph.in_degree(relation_id) == 0
    }
    collapsed_strings = {str(node) for node in collapsed}
    changed = True
    while changed:
        changed = False
        for relation_id in list(collapsed):
            link = by_relation[relation_id]
            if link["source"] in collapsed_strings or link["target"] in collapsed_strings:
                collapsed.remove(relation_id)
                collapsed_strings.remove(str(relation_id))
                changed = True
    visible = set(graph.nodes) - collapsed
    visible_links = [
        link for link in binary_links
        if link["relation_entity_id"] in visible or link["relation_entity_id"] in collapsed
    ]
    return visible, visible_links


def _derived_projection_links(graph: nx.Graph) -> list[dict[str, Any]]:
    """Backward-compatible temporal/causal subset of display projections."""

    return [
        link for link in _binary_projection_links(graph)
        if link["category"] in {"temporal", "causal"}
    ]


def _annotate_parallel_links(links: list[dict[str, Any]]) -> None:
    """Give links sharing endpoints stable offsets for readable labels."""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for link in links:
        source = str(link["source"])
        target = str(link["target"])
        groups[tuple(sorted((source, target)))].append(link)
    for group in groups.values():
        if len(group) < 2:
            continue
        for index, link in enumerate(group):
            link["parallel_index"] = index
            link["parallel_count"] = len(group)
            link["parallel_direction"] = (
                1 if str(link["source"]) <= str(link["target"]) else -1
            )


def graph_to_text(
    value: nx.Graph | Any,
    *,
    node_label_attr: str = "label",
    edge_label_attr: str = "label",
    show_derived_links: bool = True,
) -> str:
    """Return a readable, indented text view of the graph.

    Each node with outgoing relations is followed by those relations. Incoming
    relations are not repeated, and target nodes are shown inline on the
    relation line. The relation line keeps the predicate and target ontology
    ID together.
    """

    graph = _graph_from_value(value)
    visible_nodes, projection_links = _display_projection_view(
        graph,
        show_derived_links=show_derived_links,
    )
    node_items = [
        (node_id, data)
        for node_id, data in graph.nodes(data=True)
        if node_id in visible_nodes
    ]
    node_lookup = {str(node_id): node_id for node_id in visible_nodes}
    projected_by_source: dict[Any, list[tuple[Any, dict[str, Any]]]] = {}
    for link in projection_links:
        source = node_lookup.get(link["source"])
        target = node_lookup.get(link["target"])
        if source is not None and target is not None:
            projected_by_source.setdefault(source, []).append((target, link))
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
                if node_id not in projected_by_source:
                    continue
        elif graph.degree(node_id) == 0:
            continue

        node_label_value = data.get(node_label_attr, node_id)
        if data.get("relation") is not None:
            node_label_value = f"{data.get('type', node_label_value)} [{data['relation']}]"
        node_label = _ontology_text(node_label_value)
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
                if target_id in visible_nodes
            ]
        else:
            relation_items = [
                (
                    target_id if source_id == node_id else source_id,
                    relation_data,
                )
                for source_id, target_id, relation_data in graph.edges(node_id, data=True)
                if source_id in visible_nodes and target_id in visible_nodes
            ]
        relation_items.extend(projected_by_source.get(node_id, []))
        for target_id, relation_data in relation_items:
            predicate_value = relation_data.get("role", relation_data.get("predicate"))
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
    show_derived_links: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Convert a graph into D3 data with wrapped multiline labels."""

    graph = _graph_from_value(value)
    max_width = _validate_max_width(max_width)
    visible_nodes, projection_links = _display_projection_view(
        graph,
        show_derived_links=show_derived_links,
    )

    def combined_label(primary: Any, source: Any) -> str:
        primary_text = _wrap_text(_ontology_text(primary), max_width)
        source_text = _wrap_text(source, max_width) if source else ""
        if not show_source or not source_text:
            return primary_text
        return f"{primary_text}\n{source_text}"

    def node_primary(data: dict[str, Any], fallback: Any) -> Any:
        if data.get("relation") is not None:
            return f"{data.get('type', fallback)} : {data['relation']}"
        return data.get(node_label_attr, fallback)

    def node_source(data: dict[str, Any]) -> str:
        mentions = data.get("mentions")
        if isinstance(mentions, (list, tuple)):
            return "; ".join(str(mention) for mention in mentions if mention)
        return str(mentions) if mentions else ""

    node_items = [
        (node_id, data)
        for node_id, data in graph.nodes(data=True)
        if node_id in visible_nodes
    ]
    sequence_by_node = {
        node_id: data.get("sequence", index)
        if isinstance(data.get("sequence", index), (int, float))
        else index
        for index, (node_id, data) in enumerate(node_items)
    }
    visible_graph = graph.subgraph(visible_nodes)
    components = list(nx.connected_components(visible_graph.to_undirected()))
    components.sort(key=lambda component: min(sequence_by_node[node] for node in component))
    component_info: dict[Any, tuple[int, int, int]] = {}
    for component_order, component in enumerate(components):
        ordered_nodes = sorted(component, key=lambda node: (sequence_by_node[node], str(node)))
        for component_index, node_id in enumerate(ordered_nodes):
            component_info[node_id] = (component_order, component_index, len(ordered_nodes))

    nodes = [
        {
            "id": str(node_id),
            "label": combined_label(node_primary(data, node_id), node_source(data)),
            "ontology_label": _ontology_text(node_primary(data, node_id)),
            "source_fragment": node_source(data),
            "node_type": str(data.get("node_type", "relation" if data.get("relation") is not None else "entity")),
            "sequence": sequence_by_node[node_id],
            "component_order": component_info[node_id][0],
            "component_index": component_info[node_id][1],
            "component_size": component_info[node_id][2],
        }
        for node_id, data in node_items
    ]
    if graph.is_multigraph():
        edges = (
            (source, target, key, data)
            for source, target, key, data in graph.edges(data=True, keys=True)
            if source in visible_nodes and target in visible_nodes
        )
        links = [
            {
                "source": str(source),
                "target": str(target),
                "label": combined_label(data.get("role", data.get("predicate", "")), data.get(edge_label_attr, "")),
                "predicate": _ontology_text(data.get("role", data.get("predicate", ""))),
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
                "label": combined_label(data.get("role", data.get("predicate", "")), data.get(edge_label_attr, "")),
                "predicate": _ontology_text(data.get("role", data.get("predicate", ""))),
                "source_fragment": str(data.get(edge_label_attr, "")),
                "edge_type": str(data.get("edge_type", "semantic")),
                "category": str(data.get("category", "semantic")),
                "directed": graph.is_directed(),
            }
            for source, target, data in graph.edges(data=True)
            if source in visible_nodes and target in visible_nodes
        ]
    if show_derived_links:
        for projection_link in projection_links:
            link = dict(projection_link)
            link["label"] = combined_label(
                link.get("predicate", link.get("label", "")),
                link.get("source_fragment", ""),
            )
            links.append(link)
    _annotate_parallel_links(links)
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
    show_derived_links: bool = True,
    charge_strength: float = -180,
    link_distance: float = 90,
    parallel_edge_spacing: float = 48,
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
    parallel_edge_spacing = _validate_positive_number(
        parallel_edge_spacing, "parallel_edge_spacing"
    )
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
            show_derived_links=show_derived_links,
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
        parallel_edge_spacing=parallel_edge_spacing,
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
    parallel_edge_spacing: float,
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
  const parallelEdgeSpacing = {parallel_edge_spacing};
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

  const endpointId = endpoint => endpoint && typeof endpoint === "object" ? endpoint.id : endpoint;
  const isSelfLoop = d => String(endpointId(d.source)) === String(endpointId(d.target));

  const link = viewport.append("g")
    .attr("aria-hidden", "true")
    .selectAll("path")
    .data(data.links)
    .join("path")
    .attr("fill", "none")
    .attr("stroke", d => d.category === "causal" ? "#c2410c" : d.category === "temporal" ? "#2563eb" : "#9aa0a6")
    .attr("stroke-width", d => d.category === "causal" ? 2.5 : d.category === "temporal" ? 1.5 : 1)
    .attr("stroke-opacity", 0.85)
    .attr("stroke-dasharray", d => d.category === "temporal" ? "6 4" : null)
    .attr("display", d => isSelfLoop(d) ? "none" : null)
    .attr("marker-end", d => d.directed
      ? `url(#${{d.category === "causal" ? causalArrowId : d.category === "temporal" ? temporalArrowId : arrowId}})`
      : null);

  const selfLoop = viewport.append("g")
    .attr("aria-hidden", "true")
    .selectAll("path")
    .data(data.links.filter(isSelfLoop))
    .join("path")
    .attr("fill", "none")
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
  const edgeLabelWidth = d => Math.max(...String(d.label).split("\\n").map(line => line.length), 1);
  const edgeOffset = d => {{
    const count = Number(d.parallel_count || 1);
    const index = Number(d.parallel_index || 0);
    return (index - (count - 1) / 2) * parallelEdgeSpacing;
  }};
  const linkGeometry = d => {{
    if (isSelfLoop(d)) {{
      const radius = Math.max(34, Math.min(86, edgeLabelWidth(d) * 1.6));
      const verticalOffset = edgeOffset(d) * 1.8;
      const x = d.source.x;
      const y = d.source.y + verticalOffset;
      const startX = x + 10;
      const startY = y - 8;
      const endY = y + 8;
      const controlX = x + radius + 30;
      return {{
        selfLoop: true,
        path: `M${{startX}},${{startY}} C${{controlX}},${{y - radius}} ${{controlX}},${{y + radius}} ${{startX}},${{endY}}`,
        labelX: x + radius + 34,
        labelY: y - radius - 8,
      }};
    }}
    const dx = d.target.x - d.source.x;
    const dy = d.target.y - d.source.y;
    const length = Math.sqrt(dx * dx + dy * dy) || 1;
    const normalX = -dy / length;
    const normalY = dx / length;
    const count = Number(d.parallel_count || 1);
    const index = Number(d.parallel_index || 0);
    const parallelCurvature = count > 1
      ? (index - (count - 1) / 2) * parallelEdgeSpacing * Number(d.parallel_direction || 1)
      : 0;
    const controlX = (d.source.x + d.target.x) / 2 + normalX * parallelCurvature;
    const controlY = (d.source.y + d.target.y) / 2 + normalY * parallelCurvature;
    return {{
      selfLoop: false,
      path: `M${{d.source.x}},${{d.source.y}} Q${{controlX}},${{controlY}} ${{d.target.x}},${{d.target.y}}`,
      labelX: (d.source.x + 2 * controlX + d.target.x) / 4,
      labelY: (d.source.y + 2 * controlY + d.target.y) / 4,
    }};
  }};
  const edgeDistance = d => {{
    const parallelCount = Number(d.parallel_count || 1);
    const textDistance = Math.min(320, edgeLabelWidth(d) * 5.5);
    return Math.max(linkDistance, textDistance) + Math.min(90, (parallelCount - 1) * 24);
  }};
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
  const componentSpan = componentValues.length === 1
    ? Math.max(220, width - 160)
    : Math.min(260, Math.max(180, (width - 180) / componentValues.length));
  const componentTarget = d => {{
    const center = componentScale(d.component_order) ?? width / 2;
    const slots = Math.max(1, d.component_size - 1);
    const local = (d.component_index / slots - 0.5)
      * componentSpan;
    return center + local;
  }};
  const sequenceValues = [...new Set(data.nodes.map(d => d.sequence))].sort((a, b) => a - b);
  const useTimeline = layoutMode === "timeline";
  const hardTimeline = useTimeline && timelineStiffness >= 0.999;
  const timelineY = height * 0.52;
  const timelineScale = d3.scalePoint()
    .domain(sequenceValues)
    .range([80, Math.max(80, width - 80)])
    .padding(0.35);
  const timelineX = d => timelineScale(d.sequence) ?? width / 2;
  const timelineYTarget = d => timelineY;
  const xTarget = d => useTimeline ? timelineX(d) : componentTarget(d);
  const xStrength = d => useTimeline ? timelineStiffness : componentStrength;
  const yStrength = d => useTimeline ? timelineStiffness : 0;
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
    .force("link", d3.forceLink(data.links).id(d => d.id).distance(d => d.category === "temporal" ? Math.min(edgeDistance(d), 120) : d.category === "causal" ? edgeDistance(d) * 1.15 : edgeDistance(d)).strength(linkStrength))
    .force("charge", d3.forceManyBody().strength({charge_strength}))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("x", d3.forceX(xTarget).strength(xStrength))
    .force("y", useTimeline ? d3.forceY(timelineYTarget).strength(yStrength) : null)
    .force("collision", d3.forceCollide().radius(d => nodeRadius(d) * 1.08).strength(1).iterations(6))
    .on("tick", () => {{
      if (hardTimeline) {{
        data.nodes.forEach(d => {{
          if (!d.pinned) {{
            d.x = timelineX(d);
            d.y = timelineY;
            d.vx = 0;
            d.vy = 0;
          }}
        }});
      }}
      link
        .attr("d", d => linkGeometry(d).path);
      selfLoop
        .attr("d", d => linkGeometry(d).path);
      edgeLabel
        .attr("x", d => linkGeometry(d).labelX)
        .attr("y", d => linkGeometry(d).labelY)
        .each(function(d) {{
          const x = linkGeometry(d).labelX;
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
    show_derived_links: bool = True,
    charge_strength: float = -180,
    link_distance: float = 90,
    parallel_edge_spacing: float = 48,
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
    parallel_edge_spacing = _validate_positive_number(
        parallel_edge_spacing, "parallel_edge_spacing"
    )
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
            show_derived_links=show_derived_links,
        ),
        ensure_ascii=False,
    ).replace("<", "\\u003c")
    return _d3_script(
        payload=payload,
        width_json=json.dumps(int(width)),
        height_json=json.dumps(int(height)),
        charge_strength=charge_strength,
        link_distance=link_distance,
        parallel_edge_spacing=parallel_edge_spacing,
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
    show_derived_links: bool = True,
    charge_strength: float = -180,
    link_distance: float = 90,
    parallel_edge_spacing: float = 48,
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
    parallel_edge_spacing = _validate_positive_number(
        parallel_edge_spacing, "parallel_edge_spacing"
    )
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
        show_derived_links=show_derived_links,
        charge_strength=charge_strength,
        link_distance=link_distance,
        parallel_edge_spacing=parallel_edge_spacing,
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
    show_derived_links: bool,
    charge_strength: float,
    link_distance: float,
    parallel_edge_spacing: float,
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
    show_derived_links=show_derived_links,
    charge_strength=charge_strength,
    link_distance=link_distance,
    parallel_edge_spacing=parallel_edge_spacing,
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
    show_derived_links: bool = True,
    parallel_edge_spacing: float = 48,
    layout: str = "auto",
) -> str:
    """Return a deterministic SVG using timeline or Kamada-Kawai layout."""

    if width < 1 or height < 1:
        raise ValueError("width and height must be positive")
    parallel_edge_spacing = _validate_positive_number(
        parallel_edge_spacing, "parallel_edge_spacing"
    )
    layout = _validate_layout(layout)

    graph = _graph_from_value(value)
    display_data = graph_to_d3_data(
        graph,
        node_label_attr=node_label_attr,
        edge_label_attr=edge_label_attr,
        show_source=show_source,
        max_width=max_width,
        show_derived_links=show_derived_links,
    )
    display_nodes = {item["id"]: item for item in display_data["nodes"]}
    display_links = display_data["links"]
    node_ids = {item["id"] for item in display_data["nodes"]}
    node_lookup = {
        str(node_id): node_id
        for node_id in graph.nodes
        if str(node_id) in node_ids
    }
    node_items = [
        (node_lookup[item["id"]], graph.nodes[node_lookup[item["id"]]])
        for item in display_data["nodes"]
    ]
    visible_graph = graph.subgraph([node_id for node_id, _data in node_items])
    positions = nx.kamada_kawai_layout(visible_graph, weight=None) if node_items else {}
    use_timeline = layout == "timeline"
    component_count = max(
        (int(item["component_order"]) for item in display_data["nodes"]),
        default=-1,
    ) + 1
    margin = 48
    usable_width = max(width - 2 * margin, 1)
    usable_height = max(height - 2 * margin, 1)
    sequence_values = sorted(item["sequence"] for item in display_data["nodes"])

    def point(node_id: Any) -> tuple[float, float]:
        node_data = display_nodes[str(node_id)]
        if use_timeline:
            sequence_index = sequence_values.index(node_data["sequence"])
            x = margin + (
                sequence_index / max(len(sequence_values) - 1, 1)
            ) * usable_width
            timeline_y = margin + usable_height * 0.52
            return x, timeline_y
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

    def edge_geometry(link: dict[str, Any]) -> dict[str, Any]:
        source = node_lookup[str(link["source"])]
        target = node_lookup[str(link["target"])]
        x1, y1 = point(source)
        x2, y2 = point(target)
        if source == target:
            radius = max(34.0, min(86.0, len(str(link.get("label", ""))) * 1.6))
            count = int(link.get("parallel_count", 1))
            index = int(link.get("parallel_index", 0))
            vertical_offset = (index - (count - 1) / 2) * parallel_edge_spacing
            y = y1 + vertical_offset
            start_x = x1 + 10.0
            start_y = y - 8.0
            end_y = y + 8.0
            control_x = x1 + radius + 30.0
            return {
                "self_loop": True,
                "path": (
                    f"M{start_x:.2f},{start_y:.2f} "
                    f"C{control_x:.2f},{y - radius:.2f} "
                    f"{control_x:.2f},{y + radius:.2f} "
                    f"{start_x:.2f},{end_y:.2f}"
                ),
                "label_x": x1 + radius + 34.0,
                "label_y": y - radius - 8.0,
            }
        dx = x2 - x1
        dy = y2 - y1
        length = (dx * dx + dy * dy) ** 0.5 or 1.0
        normal_x = -dy / length
        normal_y = dx / length
        count = int(link.get("parallel_count", 1))
        index = int(link.get("parallel_index", 0))
        parallel_curvature = (
            (index - (count - 1) / 2)
            * parallel_edge_spacing
            * int(link.get("parallel_direction", 1))
            if count > 1
            else 0.0
        )
        control_x = (x1 + x2) / 2 + normal_x * parallel_curvature
        control_y = (y1 + y2) / 2 + normal_y * parallel_curvature
        return {
            "self_loop": False,
            "path": (
                f"M{x1:.2f},{y1:.2f} Q{control_x:.2f},{control_y:.2f} "
                f"{x2:.2f},{y2:.2f}"
            ),
            "label_x": (x1 + 2 * control_x + x2) / 4,
            "label_y": (y1 + 2 * control_y + y2) / 4,
        }

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

    for link in display_links:
        source = node_lookup.get(str(link["source"]))
        target = node_lookup.get(str(link["target"]))
        if source is None or target is None:
            continue
        geometry = edge_geometry(link)
        category = link.get("category", "semantic")
        stroke = {"causal": "#c2410c", "temporal": "#2563eb"}.get(category, "#9aa0a6")
        stroke_width = {"causal": 2.5, "temporal": 1.5}.get(category, 1)
        dash = ' stroke-dasharray="6 4"' if category == "temporal" else ""
        marker_id = {
            "causal": "semantic-graphicalizer-causal-arrow",
            "temporal": "semantic-graphicalizer-temporal-arrow",
        }.get(category, "semantic-graphicalizer-arrow")
        marker = f' marker-end="url(#{marker_id})"' if graph.is_directed() else ""
        if geometry["self_loop"]:
            elements.append(
                f'<path d="{geometry["path"]}" fill="none" stroke="{stroke}" '
                f'stroke-width="{stroke_width}" stroke-opacity="0.85"{dash}{marker} />'
            )
        else:
            elements.append(
                f'<path d="{geometry["path"]}" fill="none" stroke="{stroke}" '
                f'stroke-width="{stroke_width}" stroke-opacity="0.85"{dash}{marker} />'
            )
        label = link["label"]
        if label:
            elements.append(svg_label(
                geometry["label_x"],
                geometry["label_y"] - 4,
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
        'surface mentions as text-only nodes, and argument roles with source '
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
        show_derived_links=kwargs.get("show_derived_links", True),
        charge_strength=kwargs.get("charge_strength", -180),
        link_distance=kwargs.get("link_distance", 90),
        parallel_edge_spacing=kwargs.get("parallel_edge_spacing", 48),
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
