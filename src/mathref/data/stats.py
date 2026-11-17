"""Dataset statistics for normalized frames and ranking examples."""

from collections import Counter
from collections.abc import Iterable
from statistics import mean, median
from typing import Any


def summarize_examples(examples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(examples)
    videos = {(row["language"], row["topic"], row["video"]) for row in rows}
    frame_ids = {row["frame_id"] for row in rows}
    topics = Counter(row["topic"] for row in rows)
    languages = Counter(row["language"] for row in rows)
    object_types: Counter[str] = Counter()
    candidate_counts: list[int] = []
    gold_counts: list[int] = []

    for row in rows:
        candidate_counts.append(len(row.get("candidates", [])))
        gold_counts.append(len(row.get("gold_visual_ids", [])))
        for object_type in row.get("gold_object_types", []):
            object_types[object_type] += 1

    return {
        "examples": len(rows),
        "videos": len(videos),
        "frames": len(frame_ids),
        "languages": dict(sorted(languages.items())),
        "topics": dict(sorted(topics.items())),
        "gold_object_type_distribution": dict(sorted(object_types.items())),
        "candidate_count": _numeric_summary(candidate_counts),
        "gold_count_per_mention": _numeric_summary(gold_counts),
        "multi_gold_mentions": sum(count > 1 for count in gold_counts),
    }


def summarize_dataset(
    examples: Iterable[dict[str, Any]],
    frames: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize ranking examples with per-language raw-frame coverage."""
    example_rows = list(examples)
    frame_rows = list(frames)
    stats = summarize_examples(example_rows)
    languages = sorted(
        {row["language"] for row in example_rows}
        | {row["language"] for row in frame_rows}
    )
    stats["by_language"] = {
        language: _summarize_language(
            [row for row in example_rows if row["language"] == language],
            [row for row in frame_rows if row["language"] == language],
        )
        for language in languages
    }
    return stats


def _summarize_language(
    examples: list[dict[str, Any]],
    frames: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate_counts = [len(row.get("candidates", [])) for row in examples]
    mention_types = Counter(
        (row.get("mention") or {}).get("mention_type") or "unknown"
        for row in examples
    )
    example_count = len(examples)
    usable_frame_ids = {row["frame_id"] for row in examples}
    topics = Counter(row["topic"] for row in examples)
    videos = {(row["topic"], row["video"]) for row in frames}

    return {
        "videos": len(videos),
        "annotated_frames": len(frames),
        "usable_frames": len(usable_frame_ids),
        "empty_frames": sum(not row.get("mentions") for row in frames),
        "examples": example_count,
        "topics": dict(sorted(topics.items())),
        "candidate_count": _numeric_summary(candidate_counts),
        "mention_type_distribution": {
            mention_type: {
                "count": count,
                "percentage": count / example_count if example_count else 0.0,
            }
            for mention_type, count in sorted(mention_types.items())
        },
    }


def summarize_frames(frames: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(frames)
    videos = {(row["language"], row["topic"], row["video"]) for row in rows}
    languages = Counter(row["language"] for row in rows)
    topics = Counter(row["topic"] for row in rows)
    box_types: Counter[str] = Counter()
    mention_counts: list[int] = []
    box_counts: list[int] = []

    for row in rows:
        mentions = row.get("mentions", [])
        boxes = row.get("boxes", [])
        mention_counts.append(len(mentions))
        box_counts.append(len(boxes))
        for box in boxes:
            box_types[box["object_type"]] += 1

    return {
        "frames": len(rows),
        "videos": len(videos),
        "languages": dict(sorted(languages.items())),
        "topics": dict(sorted(topics.items())),
        "boxes": sum(box_counts),
        "mentions": sum(mention_counts),
        "box_type_distribution": dict(sorted(box_types.items())),
        "boxes_per_frame": _numeric_summary(box_counts),
        "mentions_per_frame": _numeric_summary(mention_counts),
        "empty_frames": sum(count == 0 for count in mention_counts),
    }


def _numeric_summary(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    return {
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "median": median(values),
    }
