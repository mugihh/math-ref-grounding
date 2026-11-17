"""Random ranking baseline."""

import argparse
import random
from typing import Any

from mathref.utils.io import read_jsonl, write_jsonl


def rank_candidates(candidate_ids: list[str], seed: int | None = None) -> list[str]:
    rng = random.Random(seed)
    ranked = list(candidate_ids)
    rng.shuffle(ranked)
    return ranked


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", required=True, help="Path to ranking examples JSONL.")
    parser.add_argument("--output", required=True, help="Path to prediction output JSONL.")
    parser.add_argument("--seed", type=int, default=13)
    return parser


def predict_examples(examples: list[dict[str, Any]], seed: int = 13) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    for example in examples:
        ranked = [candidate["visual_id"] for candidate in example["candidates"]]
        rng.shuffle(ranked)
        rows.append(_prediction_row(example, ranked, "random"))
    return rows


def _prediction_row(example: dict[str, Any], ranked: list[str], baseline: str) -> dict[str, Any]:
    return {
        "example_id": example["example_id"],
        "frame_id": example["frame_id"],
        "mention": example["mention"],
        "gold_visual_ids": example["gold_visual_ids"],
        "gold_object_types": example.get("gold_object_types", []),
        "image_size": example.get("image_size"),
        "candidates": example.get("candidates", []),
        "ranked_visual_ids": ranked,
        "scores": {},
        "baseline": baseline,
    }


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    main_with_args(args.examples, args.output, seed=args.seed)


def main_with_args(examples_path: str, output_path: str, seed: int = 13) -> None:
    examples = list(read_jsonl(examples_path))
    write_jsonl(output_path, predict_examples(examples, seed=seed))
