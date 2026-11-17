"""Class-prior ranking with uniform selection within each candidate class."""

import argparse
import random
from collections import Counter
from typing import Any

from mathref.baselines.random import _prediction_row
from mathref.utils.io import read_jsonl, write_jsonl


def learn_class_prior(examples: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for example in examples:
        for object_type in example.get("gold_object_types", []):
            counts[object_type] += 1
    return counts


def predict_examples(
    examples: list[dict[str, Any]],
    prior: Counter[str],
    seed: int = 13,
) -> list[dict[str, Any]]:
    if sum(prior.values()) <= 0:
        raise ValueError("Class prior must contain at least one gold object type.")

    rng = random.Random(seed)
    rows = []
    for example in examples:
        candidate_class_counts = Counter(
            candidate["object_type"] for candidate in example["candidates"]
        )
        present_prior_count = sum(prior[name] for name in candidate_class_counts)
        if present_prior_count:
            scores = {
                candidate["visual_id"]: (
                    prior[candidate["object_type"]]
                    / present_prior_count
                    / candidate_class_counts[candidate["object_type"]]
                )
                for candidate in example["candidates"]
            }
        else:
            uniform_score = 1.0 / len(example["candidates"])
            scores = {
                candidate["visual_id"]: uniform_score
                for candidate in example["candidates"]
            }
        tie_breakers = {
            candidate["visual_id"]: rng.random() for candidate in example["candidates"]
        }
        ranked = sorted(
            scores,
            key=lambda visual_id: (-scores[visual_id], tie_breakers[visual_id]),
        )
        row = _prediction_row(example, ranked, "class_prior_uniform")
        row["scores"] = scores
        rows.append(row)
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", required=True, help="Path to ranking examples JSONL.")
    parser.add_argument(
        "--prior-examples",
        required=True,
        help="Training examples JSONL used only to estimate the class prior.",
    )
    parser.add_argument("--output", required=True, help="Path to prediction output JSONL.")
    parser.add_argument("--seed", type=int, default=13)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    main_with_args(
        args.examples,
        args.prior_examples,
        args.output,
        seed=args.seed,
    )


def main_with_args(
    examples_path: str,
    prior_examples_path: str,
    output_path: str,
    seed: int = 13,
) -> None:
    examples = list(read_jsonl(examples_path))
    prior_examples = list(read_jsonl(prior_examples_path))
    prior = learn_class_prior(prior_examples)
    write_jsonl(output_path, predict_examples(examples, prior, seed=seed))
