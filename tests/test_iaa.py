import json
from pathlib import Path

import pytest

from mathref.eval.iaa import _parse_frame, compare_frames, compute_iaa


def _write_frame(
    root: Path,
    caption: str,
    mentions: list[tuple[int, int, str]],
    boxes: list[tuple[str, str, list[int]]],
) -> None:
    path = root / "en" / "probability" / "video1" / "000.jpg.json"
    path.parent.mkdir(parents=True)
    data = {
        "image_id": "000",
        "caption": {"id": "000", "text": caption},
        "bounding_boxes": [
            {"id": visual_id, "class_name": object_type, "bb": bbox}
            for visual_id, object_type, bbox in boxes
        ],
        "i2t_relations": [
            {
                "id": str(index),
                "visual_id": visual_id,
                "caption_id": "000",
                "caption_region": [start, length],
                "relation_name": "equation",
            }
            for index, (start, length, visual_id) in enumerate(mentions)
        ],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_parser_maps_tagged_mentions_to_plain_canonical_offsets(tmp_path: Path) -> None:
    root = tmp_path / "a"
    caption = "Now divide <<<this expression_00>>> by x."
    start = caption.index("this expression_00")
    _write_frame(root, caption, [(start, len("this expression_00"), "eq_a")], [("eq_a", "equation", [0, 0, 10, 10])])

    frame = _parse_frame(next(root.rglob("*.jpg.json")))

    assert frame["caption"] == "Now divide this expression by x."
    assert frame["mentions"] == [{"span": [11, 26], "text": "this expression", "visual_ids": ["eq_a"]}]


def test_relaxed_mention_box_class_and_link_metrics() -> None:
    frame_a = {
        "caption": "Now divide this expression by x.",
        "mentions": [{"span": [11, 26], "text": "this expression", "visual_ids": ["eq_a"]}],
        "boxes": [{"visual_id": "eq_a", "object_type": "equation", "bbox": [100, 50, 200, 60]}],
    }
    frame_b = {
        "caption": frame_a["caption"],
        "mentions": [{"span": [16, 26], "text": "expression", "visual_ids": ["box_b"]}],
        "boxes": [{"visual_id": "box_b", "object_type": "span", "bbox": [105, 52, 195, 58]}],
    }

    metrics, disagreements = compare_frames(frame_a, frame_b)

    assert metrics["mention_exact"]["f1"] == 0
    assert metrics["mention_relaxed"]["f1"] == 1
    assert metrics["box"]["f1"] == 1
    assert metrics["box_class_agreement"] == 0
    assert metrics["box_class_kappa"] == 0
    assert metrics["link"]["exact_agreement"] == 1
    assert len(disagreements["class_disagreements"]) == 1


def test_directory_comparison_writes_pairwise_frame_result(tmp_path: Path) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    caption_a = "Choose <<<this term_00>>>."
    caption_b = "Choose this <<<term>>>."
    start_a = caption_a.index("this term_00")
    start_b = caption_b.index("term")
    _write_frame(root_a, caption_a, [(start_a, len("this term_00"), "term_a")], [("term_a", "term", [0, 0, 20, 20])])
    _write_frame(root_b, caption_b, [(start_b, len("term"), "term_b")], [("term_b", "term", [1, 1, 20, 20])])

    metrics, disagreements = compute_iaa(root_a, root_b)

    assert metrics["frame_count"] == 1
    assert metrics["summary"]["mention_relaxed"]["f1"] == 1
    assert metrics["summary"]["box_class_agreement"] == 1
    assert metrics["summary"]["box_class_kappa"] == 1
    assert metrics["summary"]["link"]["exact_agreement"] == 1
    assert metrics["frames"][0]["mention_relaxed"]["f1"] == 1
    assert metrics["frames"][0]["link"]["exact_agreement"] == 1
    assert disagreements == []


def test_canonical_transcript_mismatch_is_rejected() -> None:
    frame = {"caption": "same", "mentions": [], "boxes": []}
    with pytest.raises(ValueError, match="canonical plain transcript"):
        compare_frames(frame, {**frame, "caption": "different"})
