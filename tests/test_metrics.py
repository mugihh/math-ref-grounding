import json

import pytest

from mathref.eval.metrics import (
    _candidate_count_bucket,
    evaluate_predictions,
    main,
    recall_at_k,
    reciprocal_rank,
)
from mathref.utils.io import write_jsonl


def _prediction_row() -> dict:
    return {
        "ranked_visual_ids": ["a", "b"],
        "gold_visual_ids": ["a"],
        "gold_object_types": ["equation"],
        "image_size": [100, 100],
        "candidates": [
            {"visual_id": "a", "bbox": [0, 0, 10, 10]},
            {"visual_id": "b", "bbox": [0, 0, 10, 10]},
        ],
    }


def test_output_requires_method(tmp_path) -> None:
    preds = tmp_path / "preds.jsonl"
    write_jsonl(preds, [_prediction_row()])
    out = tmp_path / "eval.json"
    # Persisting without --method is rejected before any file is written.
    with pytest.raises(SystemExit):
        main(["--predictions", str(preds), "--output", str(out)])
    assert not out.exists()
    # With --method it writes a result carrying meta.
    main(["--predictions", str(preds), "--output", str(out), "--method", "random", "--seed", "0"])
    written = json.loads(out.read_text())
    assert written["meta"]["method"] == "random" and written["meta"]["seed"] == 0


def test_reciprocal_rank_returns_inverse_gold_position() -> None:
    assert reciprocal_rank(["a", "b", "c"], "b") == 0.5


def test_reciprocal_rank_accepts_multiple_gold_ids() -> None:
    assert reciprocal_rank(["a", "b", "c"], ["b", "c"]) == 0.5


def test_recall_at_k_checks_top_k_candidates() -> None:
    assert recall_at_k(["a", "b", "c"], "b", 2) == 1.0
    assert recall_at_k(["a", "b", "c"], "c", 2) == 0.0


def test_recall_at_k_accepts_multiple_gold_ids() -> None:
    assert recall_at_k(["a", "b", "c"], ["c", "d"], 3) == 1.0
    assert recall_at_k(["a", "b", "c"], ["c", "d"], 2) == 0.0


def test_evaluate_predictions_reports_spatial_coverage() -> None:
    rows = [
        {
            "ranked_visual_ids": ["a"],
            "gold_visual_ids": ["a"],
            "gold_object_types": ["term"],
            "image_size": [100, 100],
            "candidates": [{"visual_id": "a", "bbox": [0, 0, 10, 10]}],
        },
        {
            "ranked_visual_ids": ["b"],
            "gold_visual_ids": ["b"],
            "gold_object_types": ["term"],
        },
    ]
    result = evaluate_predictions(rows)
    assert result["recall_at_1"] == 1.0
    assert result["spatial"]["count"] == 1
    assert result["spatial"]["coverage"] == 0.5
    assert result["spatial"]["normalized_center_distance"] == 0.0
    assert result["by_gold_object_type"]["term"]["spatial"]["coverage"] == 0.5


def test_evaluate_predictions_reports_requested_breakdowns() -> None:
    rows = [
        {
            "mention": {"mention_type": "implicit"},
            "ranked_visual_ids": ["a", "b"],
            "gold_visual_ids": ["b"],
            "gold_object_types": ["term"],
            "candidates": [{"visual_id": str(index)} for index in range(5)],
        },
        {
            "mention": {"mention_type": "explicit"},
            "ranked_visual_ids": ["a", "b", "c"],
            "gold_visual_ids": ["a", "c"],
            "gold_object_types": ["equation"],
            "candidates": [{"visual_id": str(index)} for index in range(16)],
        },
    ]

    result = evaluate_predictions(rows)

    assert result["by_mention_type"]["implicit"]["count"] == 1
    assert result["by_mention_type"]["implicit"]["recall_at_1"] == 0.0
    assert result["by_mention_type"]["explicit"]["recall_at_1"] == 1.0
    assert result["by_candidate_count"]["1-5"]["count"] == 1
    assert result["by_candidate_count"]["16+"]["count"] == 1
    assert result["by_gold_count"]["single_gold"]["count"] == 1
    assert result["by_gold_count"]["multi_gold"]["count"] == 1


def test_evaluate_predictions_uses_unknown_for_missing_mention_type() -> None:
    result = evaluate_predictions(
        [
            {
                "mention": {"text": ""},
                "ranked_visual_ids": ["a"],
                "gold_visual_ids": ["a"],
                "candidates": [{"visual_id": "a"}],
            }
        ]
    )

    assert result["by_mention_type"]["unknown"]["count"] == 1


def test_candidate_count_bucket_boundaries() -> None:
    assert _candidate_count_bucket(1) == "1-5"
    assert _candidate_count_bucket(5) == "1-5"
    assert _candidate_count_bucket(6) == "6-10"
    assert _candidate_count_bucket(10) == "6-10"
    assert _candidate_count_bucket(11) == "11-15"
    assert _candidate_count_bucket(15) == "11-15"
    assert _candidate_count_bucket(16) == "16+"
