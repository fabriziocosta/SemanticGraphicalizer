"""Image-specific graph overlay rendering."""

from __future__ import annotations

import math
from pathlib import Path
from collections.abc import Iterable, Mapping
from textwrap import wrap
from typing import Any

import networkx as nx


def display_image_graph(
    graph: nx.MultiDiGraph,
    image: str | Path | Any | None = None,
    *,
    show_boxes: bool = True,
    show_labels: bool = True,
    show_relations: bool = True,
    show_scene_nodes: bool = False,
    show_ids: bool = False,
    line_width: int = 3,
) -> Any:
    """Render a semantic graph over its source image using entity bounding boxes.

    Atomic entity nodes are placed at their bounding-box centers. Reified
    relation nodes are placed at the mean position of their argument entities,
    and their argument edges are drawn over the image. The returned object is
    an IPython image for notebook display, or a Pillow image when IPython is
    unavailable. Whole-scene nodes and relations attached to them are hidden
    by default; set ``show_scene_nodes=True`` to include them.
    """

    if not isinstance(graph, nx.MultiDiGraph):
        raise TypeError("graph must be a NetworkX MultiDiGraph")
    for name, value in (
        ("show_boxes", show_boxes),
        ("show_labels", show_labels),
        ("show_relations", show_relations),
        ("show_scene_nodes", show_scene_nodes),
        ("show_ids", show_ids),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a bool")
    if isinstance(line_width, bool) or not isinstance(line_width, int) or line_width < 1:
        raise ValueError("line_width must be a positive integer")

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError("display_image_graph requires Pillow; install semantic-graphicalizer[image]") from exc

    source = image if image is not None else graph.graph.get("source_image")
    if source is None:
        raise ValueError("provide image=... or graph metadata must contain 'source_image'")
    if isinstance(source, Image.Image):
        canvas = source.convert("RGBA")
    else:
        try:
            with Image.open(source) as opened:
                canvas = opened.convert("RGBA")
        except (OSError, TypeError) as exc:
            raise ValueError(f"could not load overlay image '{source}': {exc}") from exc
    width, height = canvas.size
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = ImageFont.load_default()

    def bbox_for(data: Mapping[str, Any]) -> tuple[tuple[int, int, int, int] | None, bool]:
        attributes = data.get("attributes")
        attributes = attributes if isinstance(attributes, Mapping) else {}
        provenance = attributes.get("provenance", data.get("provenance", []))
        records = provenance if isinstance(provenance, (list, tuple)) else [provenance]
        locator: Mapping[str, Any] = {}
        fallback_locator: Mapping[str, Any] = {}
        for record in records:
            if not isinstance(record, Mapping):
                continue
            candidate = record.get("source_locator", record)
            if not isinstance(candidate, Mapping):
                continue
            if candidate.get("entity_bbox") is not None:
                locator = candidate
                break
            if not fallback_locator and (
                candidate.get("normalized_bbox") is not None
                or candidate.get("pixel_bbox") is not None
                or candidate.get("whole_image") is True
            ):
                fallback_locator = candidate
        if not locator:
            locator = fallback_locator
        if not locator and isinstance(data.get("source_locator"), Mapping):
            locator = data["source_locator"]
        if locator.get("whole_image") is True:
            return (0, 0, width - 1, height - 1), True
        normalized = locator.get(
            "entity_bbox",
            locator.get("normalized_bbox", attributes.get("normalized_bbox", data.get("bbox"))),
        )
        if isinstance(normalized, (list, tuple)) and len(normalized) == 4:
            try:
                x0, y0, x1, y1 = (float(item) for item in normalized)
            except (TypeError, ValueError):
                x0 = y0 = x1 = y1 = math.nan
            if all(math.isfinite(item) for item in (x0, y0, x1, y1)):
                left = max(0, min(width - 1, round(x0 * width)))
                top = max(0, min(height - 1, round(y0 * height)))
                right = max(left + 1, min(width, round(x1 * width)))
                bottom = max(top + 1, min(height, round(y1 * height)))
                return (left, top, right - 1, bottom - 1), False
        pixels = locator.get("pixel_bbox")
        if isinstance(pixels, (list, tuple)) and len(pixels) == 4:
            try:
                left, top, right, bottom = (int(item) for item in pixels)
            except (TypeError, ValueError):
                return None, False
            left, top = max(0, left), max(0, top)
            right, bottom = min(width, right), min(height, bottom)
            if right > left and bottom > top:
                return (left, top, right - 1, bottom - 1), False
        return None, False

    palette = ["#ff5a5f", "#00a6a6", "#f5a623", "#725ac1", "#2e86de", "#20a464", "#db4c77"]
    entity_types = sorted({str(data.get("type", "entity")) for _node, data in graph.nodes(data=True) if data.get("relation") is None})
    type_colors = {entity_type: palette[index % len(palette)] for index, entity_type in enumerate(entity_types)}
    positions: dict[Any, tuple[float, float]] = {}
    boxes: dict[Any, tuple[tuple[int, int, int, int], bool]] = {}
    hidden_scene_nodes: set[Any] = set()

    for node_id, data in graph.nodes(data=True):
        if data.get("relation") is not None:
            continue
        box, whole_image = bbox_for(data)
        is_scene = whole_image or str(data.get("type", "")).casefold() == "scene"
        if is_scene and not show_scene_nodes:
            hidden_scene_nodes.add(node_id)
            continue
        if is_scene and box is None:
            box, whole_image = (0, 0, width - 1, height - 1), True
        if box is None:
            continue
        boxes[node_id] = (box, whole_image)
        left, top, right, bottom = box
        if whole_image:
            # The scene spans the image; anchor its graph marker along the top edge.
            positions[node_id] = (width / 2, max(12, min(height - 12, height * 0.04)))
        else:
            positions[node_id] = ((left + right) / 2, (top + bottom) / 2)

    relation_targets: dict[Any, list[Any]] = {}
    for node_id, data in graph.nodes(data=True):
        if data.get("relation") is None:
            continue
        argument_targets = [
            target
            for _source, target, edge in graph.out_edges(node_id, data=True)
            if edge.get("edge_type") == "argument"
        ]
        if not show_scene_nodes and any(target in hidden_scene_nodes for target in argument_targets):
            continue
        targets = [
            target
            for target in argument_targets
            if target in positions
        ]
        if targets:
            relation_targets[node_id] = targets
            positions[node_id] = (
                sum(positions[target][0] for target in targets) / len(targets),
                sum(positions[target][1] for target in targets) / len(targets),
            )

    def rgba(color: str, alpha: int = 255) -> tuple[int, int, int, int]:
        value = color.lstrip("#")
        return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4)) + (alpha,)  # type: ignore[return-value]

    def arrow(start: tuple[float, float], end: tuple[float, float], color: str) -> None:
        draw.line((start, end), fill=rgba(color, 210), width=line_width)
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        size = max(7, line_width * 3)
        wing_a = (end[0] - size * math.cos(angle - math.pi / 6), end[1] - size * math.sin(angle - math.pi / 6))
        wing_b = (end[0] - size * math.cos(angle + math.pi / 6), end[1] - size * math.sin(angle + math.pi / 6))
        draw.polygon((end, wing_a, wing_b), fill=rgba(color, 230))

    if show_relations:
        for relation_id, targets in relation_targets.items():
            relation_color = "#263238"
            anchor = positions[relation_id]
            for target in targets:
                arrow(anchor, positions[target], relation_color)

    for node_id, (box, whole_image) in boxes.items():
        if show_boxes and not whole_image:
            color = type_colors.get(str(graph.nodes[node_id].get("type", "entity")), "#ff5a5f")
            draw.rectangle(box, outline=rgba(color), width=line_width)

    occupied_labels: list[tuple[int, int, int, int]] = []
    padding_x, padding_y, label_gap = 5, 3, 6
    max_label_chars = max(12, min(34, (width - 32) // 7))

    def label_lines(label: str) -> str:
        return "\n".join(
            line
            for paragraph in label.splitlines()
            for line in (wrap(paragraph, width=max_label_chars) or [""])
        )

    def label_size(label: str) -> tuple[tuple[int, int, int, int], int, int]:
        bounds = draw.multiline_textbbox((0, 0), label, font=font, spacing=2, align="center")
        return bounds, bounds[2] - bounds[0], bounds[3] - bounds[1]

    def label_candidates(
        point: tuple[float, float], text_width: int, text_height: int, *, box: tuple[int, int, int, int] | None = None
    ) -> Iterable[tuple[float, float]]:
        x, y = point
        vertical_step = text_height + 2 * padding_y + label_gap
        if box is not None:
            left, top, right, bottom = box
            for row in range(max(1, height // max(1, vertical_step))):
                yield ((left + right) / 2, top - label_gap - text_height / 2 - row * vertical_step)
            for row in range(max(1, height // max(1, vertical_step))):
                yield ((left + right) / 2, bottom + label_gap + text_height / 2 + row * vertical_step)
            horizontal_offset = text_width / 2 + label_gap
            for row in range(max(1, height // max(1, vertical_step))):
                offset_y = row * vertical_step
                yield (left - horizontal_offset, (top + bottom) / 2 + offset_y)
                yield (right + horizontal_offset, (top + bottom) / 2 + offset_y)
                if row:
                    yield (left - horizontal_offset, (top + bottom) / 2 - offset_y)
                    yield (right + horizontal_offset, (top + bottom) / 2 - offset_y)
        else:
            yield (x, y)
            step_x = text_width + 2 * padding_x + label_gap
            rings = max(width // max(1, step_x), height // max(1, vertical_step))
            for ring in range(1, rings + 1):
                offsets = (
                    (0, -ring * vertical_step), (0, ring * vertical_step),
                    (-ring * step_x, 0), (ring * step_x, 0),
                    (-ring * step_x, -ring * vertical_step),
                    (ring * step_x, -ring * vertical_step),
                    (-ring * step_x, ring * vertical_step),
                    (ring * step_x, ring * vertical_step),
                )
                for offset_x, offset_y in offsets:
                    yield (x + offset_x, y + offset_y)

        # If nearby positions are full, scan the canvas for a clear slot so a
        # dense group of labels can spread out without covering one another.
        cell_width = text_width + 2 * padding_x + label_gap
        cell_height = text_height + 2 * padding_y + label_gap
        for row in range(max(1, height // max(1, cell_height))):
            for column in range(max(1, width // max(1, cell_width))):
                yield (
                    padding_x + text_width / 2 + column * cell_width,
                    padding_y + text_height / 2 + row * cell_height,
                )

    def pill(
        point: tuple[float, float],
        label: str,
        color: str,
        *,
        box: tuple[int, int, int, int] | None = None,
    ) -> bool:
        label = label_lines(label)
        bounds, text_width, text_height = label_size(label)
        rect_width = text_width + 2 * padding_x
        rect_height = text_height + 2 * padding_y
        if rect_width > width or rect_height > height:
            return False

        for center_x, center_y in label_candidates(point, text_width, text_height, box=box):
            left = max(0, min(width - rect_width, int(round(center_x - rect_width / 2))))
            top = max(0, min(height - rect_height, int(round(center_y - rect_height / 2))))
            rect = (left, top, left + rect_width, top + rect_height)
            if any(
                rect[0] < other[2] + label_gap
                and rect[2] + label_gap > other[0]
                and rect[1] < other[3] + label_gap
                and rect[3] + label_gap > other[1]
                for other in occupied_labels
            ):
                continue
            draw.rounded_rectangle(
                rect,
                radius=4,
                fill=rgba(color, 230),
                outline=(255, 255, 255, 230),
                width=1,
            )
            draw.multiline_text(
                (left + padding_x - bounds[0], top + padding_y - bounds[1]),
                label,
                font=font,
                spacing=2,
                align="center",
                fill=(255, 255, 255, 255),
            )
            occupied_labels.append(rect)
            return True
        return False

    entity_label_nodes = sorted(
        boxes,
        key=lambda node_id: (boxes[node_id][0][1], boxes[node_id][0][0], str(node_id)),
    )
    for node_id in entity_label_nodes:
        box, whole_image = boxes[node_id]
        data = graph.nodes[node_id]
        point = positions[node_id]
        color = type_colors.get(str(data.get("type", "entity")), "#ff5a5f")
        radius = max(5, line_width + 2)
        cx, cy = int(round(point[0])), int(round(point[1]))
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=rgba(color), outline=(255, 255, 255, 255), width=2)
        if show_labels:
            attributes = data.get("attributes") if isinstance(data.get("attributes"), Mapping) else {}
            mentions = attributes.get("mentions", data.get("mentions", []))
            mention = mentions[0] if isinstance(mentions, (list, tuple)) and mentions else str(node_id).split(":")[-1]
            label = f"{data.get('type', 'entity')}\n{mention}"
            if show_ids:
                label = f"{label}\n{node_id}"
            pill(point, label, color, box=None if whole_image else box)

    if show_relations:
        for relation_id in relation_targets:
            data = graph.nodes[relation_id]
            point = positions[relation_id]
            color = "#323b48"
            radius = max(6, line_width + 3)
            cx, cy = int(round(point[0])), int(round(point[1]))
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=rgba(color), outline=(255, 255, 255, 255), width=2)
            if show_labels:
                label = f"[{data.get('relation', 'relation')}]"
                if show_ids:
                    label = f"{label}\n{relation_id}"
                pill(point, label, color)

    try:
        from IPython.display import Image as NotebookImage
        from io import BytesIO

        buffer = BytesIO()
        canvas.convert("RGB").save(buffer, format="PNG")
        return NotebookImage(data=buffer.getvalue(), format="png")
    except ImportError:  # pragma: no cover - depends on optional notebook install
        return canvas.convert("RGB")

