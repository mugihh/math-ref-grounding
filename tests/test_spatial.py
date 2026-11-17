import math

import pytest

from mathref.eval.spatial import (
    best_spatial_match,
    center,
    intersection_area,
    iou,
    normalized_center_distance,
    spatial_metrics_for_row,
)


def test_center_uses_xywh_bbox_format() -> None:
    assert center([10, 20, 30, 40]) == (25.0, 40.0)


def test_normalized_center_distance_uses_image_diagonal() -> None:
    distance = normalized_center_distance(
        [0, 0, 10, 10],
        [60, 80, 10, 10],
        [60, 80],
    )
    assert distance == pytest.approx(1.0)


def test_overlap_metrics_for_nested_boxes() -> None:
    outer = [0, 0, 100, 100]
    inner = [25, 25, 50, 50]
    assert intersection_area(outer, inner) == 2500
    assert iou(outer, inner) == 0.25

    metrics = best_spatial_match(inner, [outer], [200, 200])
    assert metrics["prediction_center_in_gold"] == 1.0
    assert metrics["gold_center_in_prediction"] == 1.0
    assert metrics["prediction_overlap_fraction"] == 1.0
    assert metrics["gold_overlap_fraction"] == 0.25


def test_best_spatial_match_uses_best_of_multiple_gold_boxes() -> None:
    metrics = best_spatial_match(
        [90, 90, 10, 10],
        [[0, 0, 10, 10], [90, 90, 10, 10]],
        [100, 100],
    )
    assert metrics["normalized_center_distance"] == 0.0
    assert metrics["iou"] == 1.0


def test_spatial_metrics_for_row_scores_top_ranked_candidate() -> None:
    row = {
        "ranked_visual_ids": ["prediction", "gold"],
        "gold_visual_ids": ["gold"],
        "image_size": [100, 100],
        "candidates": [
            {"visual_id": "prediction", "bbox": [0, 0, 10, 10]},
            {"visual_id": "gold", "bbox": [10, 0, 10, 10]},
        ],
    }
    metrics = spatial_metrics_for_row(row)
    assert metrics is not None
    assert metrics["normalized_center_distance"] == pytest.approx(
        10 / math.sqrt(20_000)
    )
    assert metrics["center_distance_at_0_05"] == 0.0
    assert metrics["center_distance_at_0_10"] == 1.0


def test_spatial_metrics_for_row_returns_none_without_metadata() -> None:
    assert spatial_metrics_for_row(
        {"ranked_visual_ids": ["a"], "gold_visual_ids": ["a"]}
    ) is None
