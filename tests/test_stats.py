from mathref.data.stats import summarize_dataset


def test_summarize_dataset_adds_per_language_coverage() -> None:
    frames = [
        {
            "frame_id": "jp/calculus/video1/000",
            "language": "jp",
            "topic": "calculus",
            "video": "video1",
            "mentions": [{"mention_id": "m0"}],
        },
        {
            "frame_id": "jp/calculus/video1/001",
            "language": "jp",
            "topic": "calculus",
            "video": "video1",
            "mentions": [],
        },
    ]
    examples = [
        {
            "frame_id": "jp/calculus/video1/000",
            "language": "jp",
            "topic": "calculus",
            "video": "video1",
            "mention": {"mention_type": "explicit"},
            "candidates": [{"visual_id": "term_00"}],
            "gold_visual_ids": ["term_00"],
            "gold_object_types": ["term"],
        }
    ]

    stats = summarize_dataset(examples, frames)

    assert stats["by_language"]["jp"] == {
        "videos": 1,
        "annotated_frames": 2,
        "usable_frames": 1,
        "empty_frames": 1,
        "examples": 1,
        "topics": {"calculus": 1},
        "candidate_count": {"min": 1, "max": 1, "mean": 1, "median": 1},
        "mention_type_distribution": {
            "explicit": {"count": 1, "percentage": 1.0}
        },
    }
