"""Render neutral candidate labels on annotated keyframes."""

import hashlib
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from mathref.utils.paths import resolve_image_path

_COLORS = (
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#008080",
    "#9a6324",
    "#800000",
)

_LABEL_PADDING = 3
_LABEL_GAP = 2


def candidate_label_map(example: dict[str, Any]) -> list[dict[str, str]]:
    """Map neutral display labels to visual IDs in deterministic input order."""
    candidates = example.get("candidates", [])
    width = max(2, len(str(len(candidates))))
    return [
        {
            "label": f"C{index:0{width}d}",
            "visual_id": candidate["visual_id"],
        }
        for index, candidate in enumerate(candidates, start=1)
    ]


def rendered_image_name(example_id: str) -> str:
    """Return a stable filesystem-safe PNG name for an example ID."""
    digest = hashlib.sha256(example_id.encode("utf-8")).hexdigest()[:20]
    return f"{digest}.png"


def _rect_overlap_area(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> int:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    return max(0, right - left) * max(0, bottom - top)


def _label_rect(
    x: int,
    y: int,
    width: int,
    height: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    left = min(max(0, x), max(0, image_width - width))
    top = min(max(0, y), max(0, image_height - height))
    return (left, top, left + width, top + height)


def _place_label(
    box: tuple[int, int, int, int],
    label_size: tuple[int, int],
    occupied: list[tuple[int, int, int, int]],
    image_size: tuple[int, int],
    forbidden: list[tuple[int, int, int, int]] | None = None,
) -> tuple[int, int, int, int]:
    """Place a label near its box without covering labels or candidate regions."""
    left, top, right, bottom = box
    label_width, label_height = label_size
    image_width, image_height = image_size
    forbidden = forbidden or []
    gap = _LABEL_GAP
    local_candidates = [
        (left, top - label_height - gap),
        (right - label_width, top - label_height - gap),
        (left, bottom + gap),
        (right - label_width, bottom + gap),
        (left + gap, top + gap),
        (right - label_width - gap, top + gap),
        (left + gap, bottom - label_height - gap),
        (right - label_width - gap, bottom - label_height - gap),
    ]

    # Expand outward in small rings before falling back to a whole-image grid.
    step_x = max(8, label_width // 2)
    step_y = max(8, label_height // 2)
    for radius in range(1, 7):
        for dx in range(-radius, radius + 1):
            local_candidates.append(
                (left + dx * step_x, top - label_height - gap - radius * step_y)
            )
            local_candidates.append(
                (left + dx * step_x, bottom + gap + (radius - 1) * step_y)
            )
        for dy in range(-radius + 1, radius):
            local_candidates.append(
                (left - label_width - gap - (radius - 1) * step_x, top + dy * step_y)
            )
            local_candidates.append(
                (right + gap + (radius - 1) * step_x, top + dy * step_y)
            )

    seen: set[tuple[int, int, int, int]] = set()
    best_rect: tuple[int, int, int, int] | None = None
    best_score: tuple[int, int, int] | None = None
    anchor_x, anchor_y = left, top
    for x, y in local_candidates:
        rect = _label_rect(
            x, y, label_width, label_height, image_width, image_height
        )
        if rect in seen:
            continue
        seen.add(rect)
        forbidden_overlap = sum(
            _rect_overlap_area(rect, other) for other in forbidden
        )
        label_overlap = sum(_rect_overlap_area(rect, other) for other in occupied)
        distance = abs(rect[0] - anchor_x) + abs(rect[1] - anchor_y)
        score = (forbidden_overlap, label_overlap, distance)
        if best_score is None or score < best_score:
            best_rect, best_score = rect, score
            if forbidden_overlap == 0 and label_overlap == 0:
                return rect

    for y in range(0, max(1, image_height - label_height + 1), step_y):
        for x in range(0, max(1, image_width - label_width + 1), step_x):
            rect = _label_rect(
                x, y, label_width, label_height, image_width, image_height
            )
            forbidden_overlap = sum(
                _rect_overlap_area(rect, other) for other in forbidden
            )
            label_overlap = sum(
                _rect_overlap_area(rect, other) for other in occupied
            )
            distance = abs(rect[0] - anchor_x) + abs(rect[1] - anchor_y)
            score = (forbidden_overlap, label_overlap, distance)
            if best_score is None or score < best_score:
                best_rect, best_score = rect, score
                if forbidden_overlap == 0 and label_overlap == 0:
                    return rect

    assert best_rect is not None
    return best_rect


def render_candidate_labels(
    example: dict[str, Any],
    output_path: str | Path,
    images_root: str | Path | None = None,
) -> list[dict[str, str]]:
    """Draw every candidate box and its neutral label on the source keyframe."""
    label_map = candidate_label_map(example)
    if not label_map:
        raise ValueError(f"Example {example.get('example_id')!r} has no candidates.")

    with Image.open(resolve_image_path(example["image_path"], images_root)) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    font_size = max(14, min(image.size) // 40)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=font_size)
    except OSError:
        font = ImageFont.load_default()
    line_width = max(2, min(image.size) // 300)

    rendered_candidates = []
    for index, (candidate, mapping) in enumerate(
        zip(example["candidates"], label_map, strict=True)
    ):
        x, y, width, height = candidate["bbox"]
        left = max(0, round(x))
        top = max(0, round(y))
        right = min(image.width - 1, round(x + width))
        bottom = min(image.height - 1, round(y + height))
        color = _COLORS[index % len(_COLORS)]
        draw.rectangle(
            (left, top, right, bottom),
            outline=color,
            width=line_width,
        )
        rendered_candidates.append(((left, top, right, bottom), mapping, color))

    occupied: list[tuple[int, int, int, int]] = []
    candidate_boxes = [box for box, _, _ in rendered_candidates]
    for box, mapping, color in rendered_candidates:
        left, top, right, bottom = box
        text_bbox = draw.textbbox((0, 0), mapping["label"], font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        label_width = text_width + 2 * _LABEL_PADDING
        label_height = text_height + 2 * _LABEL_PADDING
        label_rect = _place_label(
            box,
            (label_width, label_height),
            occupied,
            image.size,
            forbidden=candidate_boxes,
        )
        label_left, label_top, label_right, label_bottom = label_rect
        occupied.append(label_rect)

        anchor_x = min(max(label_left, left), label_right)
        anchor_y = min(max(label_top, top), label_bottom)
        if not (label_left <= left <= label_right and label_top <= top <= label_bottom):
            draw.line(
                (anchor_x, anchor_y, left, top),
                fill=color,
                width=line_width,
            )
        draw.rectangle(
            label_rect,
            fill=color,
        )
        draw.text(
            (label_left + _LABEL_PADDING, label_top + _LABEL_PADDING),
            mapping["label"],
            fill="white",
            font=font,
        )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG")
    return label_map
