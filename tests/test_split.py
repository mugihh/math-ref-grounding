import json

import pytest

from mathref.data.split import (
    build_split_report,
    create_video_splits,
    write_split_outputs,
)


RATIOS = {"train": 0.5, "dev": 0.25, "test": 0.25}


def _examples(video_count: int = 6, examples_per_video: int = 2) -> list[dict]:
    rows = []
    for video_index in range(video_count):
        for example_index in range(examples_per_video):
            rows.append(
                {
                    "example_id": f"v{video_index}-e{example_index}",
                    "frame_id": f"jp/topic/video{video_index}/000",
                    "language": "jp" if video_index % 2 == 0 else "en",
                    "topic": "calculus" if video_index % 3 else "linear_algebra",
                    "video": f"video{video_index}",
                    "gold_object_types": [
                        "term" if example_index % 2 == 0 else "span"
                    ],
                    "gold_visual_ids": ["term_0"],
                    "candidates": [],
                }
            )
    return rows


def test_video_split_is_reproducible_and_has_no_leakage() -> None:
    examples = _examples()
    first, first_score = create_video_splits(examples, RATIOS, seed=7, trials=100)
    second, second_score = create_video_splits(examples, RATIOS, seed=7, trials=100)

    assert {
        name: [row["example_id"] for row in rows] for name, rows in first.items()
    } == {
        name: [row["example_id"] for row in rows] for name, rows in second.items()
    }
    assert first_score == second_score

    video_splits = {}
    for split_name, rows in first.items():
        for row in rows:
            key = (row["language"], row["topic"], row["video"])
            assert key not in video_splits or video_splits[key] == split_name
            video_splits[key] = split_name
    assert set(video_splits.values()) == {"train", "dev", "test"}
    assert {row["language"] for row in first["train"]} == {"en", "jp"}
    assert {row["topic"] for row in first["train"]} == {
        "calculus",
        "linear_algebra",
    }


def test_split_preserves_every_example_once() -> None:
    examples = _examples()
    splits, _ = create_video_splits(examples, RATIOS, seed=13, trials=100)
    output_ids = [
        row["example_id"] for rows in splits.values() for row in rows
    ]
    assert sorted(output_ids) == sorted(row["example_id"] for row in examples)
    assert len(output_ids) == len(set(output_ids))


def test_write_outputs_creates_ids_assignments_and_report(tmp_path) -> None:
    splits, objective = create_video_splits(
        _examples(), RATIOS, seed=13, trials=100
    )
    report = build_split_report(
        splits, RATIOS, seed=13, trials=100, objective=objective
    )
    write_split_outputs(tmp_path, splits, report)

    assert (tmp_path / "train.txt").read_text(encoding="utf-8")
    assert len(
        (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ) == len(splits["train"])
    assignments = json.loads(
        (tmp_path / "assignments.json").read_text(encoding="utf-8")
    )
    saved_report = json.loads(
        (tmp_path / "report.json").read_text(encoding="utf-8")
    )
    assert len(assignments) == 12
    assert saved_report["total_examples"] == 12
    assert set(saved_report["splits"]) == {"train", "dev", "test"}


def test_split_requires_at_least_three_videos() -> None:
    with pytest.raises(ValueError, match="At least 3 videos"):
        create_video_splits(_examples(video_count=2), RATIOS)
