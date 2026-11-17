"""Spatially relaxed metrics for gold-candidate ranking predictions."""

import math
from collections.abc import Iterable, Sequence
from typing import Any

BBox = Sequence[float]


def center(box: BBox) -> tuple[float, float]:
    x, y, width, height = _validate_box(box)
    return x + width / 2.0, y + height / 2.0


def normalized_center_distance(
    predicted_box: BBox,
    gold_box: BBox,
    image_size: Sequence[float],
) -> float:
    width, height = _validate_image_size(image_size)
    predicted_x, predicted_y = center(predicted_box)
    gold_x, gold_y = center(gold_box)
    return math.hypot(predicted_x - gold_x, predicted_y - gold_y) / math.hypot(width, height)


def intersection_area(first_box: BBox, second_box: BBox) -> float:
    first_x, first_y, first_width, first_height = _validate_box(first_box)
    second_x, second_y, second_width, second_height = _validate_box(second_box)
    overlap_width = max(
        0.0,
        min(first_x + first_width, second_x + second_width) - max(first_x, second_x),
    )
    overlap_height = max(
        0.0,
        min(first_y + first_height, second_y + second_height) - max(first_y, second_y),
    )
    return overlap_width * overlap_height


def iou(first_box: BBox, second_box: BBox) -> float:
    intersection = intersection_area(first_box, second_box)
    first_area = _area(first_box)
    second_area = _area(second_box)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def point_inside(point: tuple[float, float], box: BBox) -> bool:
    x, y, width, height = _validate_box(box)
    point_x, point_y = point
    return x <= point_x <= x + width and y <= point_y <= y + height


def best_spatial_match(
    predicted_box: BBox,
    gold_boxes: Iterable[BBox],
    image_size: Sequence[float],
) -> dict[str, float]:
    matches = [_pair_metrics(predicted_box, gold_box, image_size) for gold_box in gold_boxes]
    if not matches:
        raise ValueError("At least one gold box is required.")
    return {
        "normalized_center_distance": min(
            match["normalized_center_distance"] for match in matches
        ),
        "iou": max(match["iou"] for match in matches),
        "prediction_center_in_gold": max(
            match["prediction_center_in_gold"] for match in matches
        ),
        "gold_center_in_prediction": max(
            match["gold_center_in_prediction"] for match in matches
        ),
        "prediction_overlap_fraction": max(
            match["prediction_overlap_fraction"] for match in matches
        ),
        "gold_overlap_fraction": max(match["gold_overlap_fraction"] for match in matches),
    }


def spatial_metrics_for_row(row: dict[str, Any]) -> dict[str, float] | None:
    ranked_ids = row.get("ranked_visual_ids") or []
    gold_ids = set(row.get("gold_visual_ids") or [])
    candidates = row.get("candidates") or []
    image_size = row.get("image_size")
    if not ranked_ids or not gold_ids or not candidates or not image_size:
        return None

    boxes_by_id = {
        candidate.get("visual_id"): candidate.get("bbox")
        for candidate in candidates
        if candidate.get("visual_id") and candidate.get("bbox")
    }
    predicted_box = boxes_by_id.get(ranked_ids[0])
    gold_boxes = [boxes_by_id[gold_id] for gold_id in gold_ids if gold_id in boxes_by_id]
    if predicted_box is None or not gold_boxes:
        return None

    try:
        metrics = best_spatial_match(predicted_box, gold_boxes, image_size)
    except (TypeError, ValueError):
        return None
    distance = metrics["normalized_center_distance"]
    metrics["center_distance_at_0_05"] = float(distance <= 0.05)
    metrics["center_distance_at_0_10"] = float(distance <= 0.10)
    return metrics


def _pair_metrics(
    predicted_box: BBox,
    gold_box: BBox,
    image_size: Sequence[float],
) -> dict[str, float]:
    intersection = intersection_area(predicted_box, gold_box)
    return {
        "normalized_center_distance": normalized_center_distance(
            predicted_box, gold_box, image_size
        ),
        "iou": iou(predicted_box, gold_box),
        "prediction_center_in_gold": float(point_inside(center(predicted_box), gold_box)),
        "gold_center_in_prediction": float(point_inside(center(gold_box), predicted_box)),
        "prediction_overlap_fraction": intersection / _area(predicted_box),
        "gold_overlap_fraction": intersection / _area(gold_box),
    }


def _area(box: BBox) -> float:
    _, _, width, height = _validate_box(box)
    return width * height


def _validate_box(box: BBox) -> tuple[float, float, float, float]:
    if len(box) != 4:
        raise ValueError("A bbox must contain [x, y, width, height].")
    x, y, width, height = (float(value) for value in box)
    if width <= 0 or height <= 0:
        raise ValueError("BBox width and height must be positive.")
    return x, y, width, height


def _validate_image_size(image_size: Sequence[float]) -> tuple[float, float]:
    if len(image_size) != 2:
        raise ValueError("Image size must contain [width, height].")
    width, height = (float(value) for value in image_size)
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive.")
    return width, height
