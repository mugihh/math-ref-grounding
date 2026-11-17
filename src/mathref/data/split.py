"""Deterministic video-level train/dev/test split generation."""

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from mathref.data.stats import summarize_examples
from mathref.utils.io import read_jsonl, write_json, write_jsonl

SPLIT_NAMES = ("train", "dev", "test")
FEATURE_WEIGHTS = {
    "examples": 100.0,
    "languages": 1.0,
    "topics": 1.0,
    "object_types": 1.0,
}
VideoKey = tuple[str, str, str]


def create_video_splits(
    examples: list[dict[str, Any]],
    ratios: dict[str, float],
    seed: int = 13,
    trials: int = 5000,
) -> tuple[dict[str, list[dict[str, Any]]], float]:
    _validate_inputs(examples, ratios, trials)
    grouped = _group_by_video(examples)
    video_keys = sorted(grouped)
    rng = random.Random(seed)
    best_assignment: dict[str, list[VideoKey]] | None = None
    best_score = math.inf

    for _ in range(trials):
        assignment = _random_assignment(video_keys, ratios, rng)
        score = _imbalance_score(assignment, grouped, ratios)
        if score < best_score:
            best_assignment = {
                split_name: list(keys) for split_name, keys in assignment.items()
            }
            best_score = score

    if best_assignment is None:
        raise ValueError(
            "No split found with train coverage for every language, topic, "
            "and gold object type. Increase --trials or revise the split ratios."
        )
    splits = {
        split_name: [
            example
            for key in sorted(best_assignment[split_name])
            for example in grouped[key]
        ]
        for split_name in SPLIT_NAMES
    }
    return splits, best_score


def build_split_report(
    splits: dict[str, list[dict[str, Any]]],
    ratios: dict[str, float],
    seed: int,
    trials: int,
    objective: float,
) -> dict[str, Any]:
    total_examples = sum(len(rows) for rows in splits.values())
    return {
        "strategy": "video_grouped_random_search",
        "seed": seed,
        "trials": trials,
        "ratios": ratios,
        "balance_feature_weights": FEATURE_WEIGHTS,
        "objective": objective,
        "total_examples": total_examples,
        "splits": {
            split_name: {
                "target_ratio": ratios[split_name],
                "actual_ratio": len(splits[split_name]) / total_examples,
                "video_ids": sorted(
                    {
                        _format_video_key(_video_key(row))
                        for row in splits[split_name]
                    }
                ),
                **summarize_examples(splits[split_name]),
            }
            for split_name in SPLIT_NAMES
        },
    }


def write_split_outputs(
    output_dir: str | Path,
    splits: dict[str, list[dict[str, Any]]],
    report: dict[str, Any],
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    assignments: dict[str, str] = {}

    for split_name in SPLIT_NAMES:
        example_ids = sorted(row["example_id"] for row in splits[split_name])
        (destination / f"{split_name}.txt").write_text(
            "".join(f"{example_id}\n" for example_id in example_ids),
            encoding="utf-8",
        )
        write_jsonl(destination / f"{split_name}.jsonl", splits[split_name])
        for row in splits[split_name]:
            assignments[row["example_id"]] = split_name

    write_json(destination / "assignments.json", dict(sorted(assignments.items())))
    write_json(destination / "report.json", report)


def _validate_inputs(
    examples: list[dict[str, Any]],
    ratios: dict[str, float],
    trials: int,
) -> None:
    if not examples:
        raise ValueError("At least one ranking example is required.")
    if trials <= 0:
        raise ValueError("Trials must be positive.")
    if set(ratios) != set(SPLIT_NAMES):
        raise ValueError(f"Ratios must define exactly: {', '.join(SPLIT_NAMES)}.")
    if any(value <= 0 for value in ratios.values()):
        raise ValueError("All split ratios must be positive.")
    if not math.isclose(sum(ratios.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Split ratios must sum to 1.")

    example_ids = [row.get("example_id") for row in examples]
    if any(not example_id for example_id in example_ids):
        raise ValueError("Every example must have a non-empty example_id.")
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("Example IDs must be unique.")

    video_count = len({_video_key(row) for row in examples})
    if video_count < len(SPLIT_NAMES):
        raise ValueError(
            f"At least {len(SPLIT_NAMES)} videos are required for non-empty splits."
        )


def _group_by_video(
    examples: list[dict[str, Any]],
) -> dict[VideoKey, list[dict[str, Any]]]:
    grouped: dict[VideoKey, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        grouped[_video_key(example)].append(example)
    return dict(grouped)


def _video_key(example: dict[str, Any]) -> VideoKey:
    try:
        return (
            str(example["language"]),
            str(example["topic"]),
            str(example["video"]),
        )
    except KeyError as error:
        raise ValueError(f"Example is missing video grouping field: {error.args[0]}") from error


def _random_assignment(
    video_keys: list[VideoKey],
    ratios: dict[str, float],
    rng: random.Random,
) -> dict[str, list[VideoKey]]:
    shuffled = list(video_keys)
    rng.shuffle(shuffled)
    assignment = {
        split_name: [shuffled[index]]
        for index, split_name in enumerate(SPLIT_NAMES)
    }
    weights = [ratios[name] for name in SPLIT_NAMES]
    for video_key in shuffled[len(SPLIT_NAMES) :]:
        split_name = rng.choices(SPLIT_NAMES, weights=weights, k=1)[0]
        assignment[split_name].append(video_key)
    return assignment


def _imbalance_score(
    assignment: dict[str, list[VideoKey]],
    grouped: dict[VideoKey, list[dict[str, Any]]],
    ratios: dict[str, float],
) -> float:
    all_examples = [example for rows in grouped.values() for example in rows]
    global_features = _feature_counts(all_examples)
    train_examples = [
        example
        for video_key in assignment["train"]
        for example in grouped[video_key]
    ]
    train_features = _feature_counts(train_examples)
    for feature_group in ("languages", "topics", "object_types"):
        if set(train_features[feature_group]) != set(global_features[feature_group]):
            return math.inf

    score = 0.0

    for split_name in SPLIT_NAMES:
        split_examples = [
            example
            for video_key in assignment[split_name]
            for example in grouped[video_key]
        ]
        split_features = _feature_counts(split_examples)
        target_ratio = ratios[split_name]
        for feature_group, global_counts in global_features.items():
            group_errors = []
            for feature_name, global_count in global_counts.items():
                target = global_count * target_ratio
                actual = split_features[feature_group][feature_name]
                group_errors.append(((actual - target) / max(target, 1.0)) ** 2)
            if group_errors:
                score += (
                    FEATURE_WEIGHTS[feature_group]
                    * sum(group_errors)
                    / len(group_errors)
                )
    return score


def _feature_counts(
    examples: list[dict[str, Any]],
) -> dict[str, Counter[str]]:
    features: dict[str, Counter[str]] = {
        "examples": Counter(),
        "languages": Counter(),
        "topics": Counter(),
        "object_types": Counter(),
    }
    for example in examples:
        features["examples"]["all"] += 1
        features["languages"][str(example["language"])] += 1
        features["topics"][str(example["topic"])] += 1
        for object_type in example.get("gold_object_types", []):
            features["object_types"][str(object_type)] += 1
    return features


def _format_video_key(key: VideoKey) -> str:
    return "/".join(key)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", required=True, help="Ranking examples JSONL.")
    parser.add_argument("--output-dir", required=True, help="Split output directory.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--dev-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--trials", type=int, default=5000)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    ratios = {
        "train": args.train_ratio,
        "dev": args.dev_ratio,
        "test": args.test_ratio,
    }
    examples = list(read_jsonl(args.examples))
    splits, objective = create_video_splits(
        examples,
        ratios,
        seed=args.seed,
        trials=args.trials,
    )
    report = build_split_report(
        splits,
        ratios,
        seed=args.seed,
        trials=args.trials,
        objective=objective,
    )
    write_split_outputs(args.output_dir, splits, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
