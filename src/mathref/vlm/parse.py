"""Parse frozen-VLM output into complete prediction rankings."""

import argparse
import hashlib
import json
import random
from collections.abc import Iterable
from typing import Any

from mathref.utils.io import read_jsonl, write_jsonl


def parse_ranking_response(response: Any) -> list[str]:
    """Extract a candidate-label ranking from JSON-like model output."""
    payload = response
    if isinstance(payload, str):
        text = payload.strip()
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1]).strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("Model response does not contain a JSON object.")
            try:
                payload = json.loads(text[start : end + 1])
            except json.JSONDecodeError as error:
                raise ValueError("Model response contains invalid JSON.") from error
    if isinstance(payload, dict):
        payload = payload.get("ranking")
    if not isinstance(payload, list) or not all(
        isinstance(value, str) for value in payload
    ):
        raise ValueError("Model response ranking must be a list of strings.")
    return payload


def complete_candidate_ranking(
    ranked_labels: list[str],
    candidate_labels: list[dict[str, str]],
) -> list[str]:
    """Validate labels, deduplicate them, and append omissions deterministically."""
    allowed = [candidate["label"] for candidate in candidate_labels]
    allowed_set = set(allowed)
    unknown = sorted(set(ranked_labels) - allowed_set)
    if unknown:
        raise ValueError(f"Unknown candidate labels: {', '.join(unknown)}")

    complete_labels = []
    seen: set[str] = set()
    for label in ranked_labels:
        if label not in seen:
            complete_labels.append(label)
            seen.add(label)
    complete_labels.extend(label for label in allowed if label not in seen)
    visual_id_by_label = {
        candidate["label"]: candidate["visual_id"] for candidate in candidate_labels
    }
    return [visual_id_by_label[label] for label in complete_labels]


def seeded_random_candidate_ranking(
    example_id: str,
    candidate_labels: list[dict[str, str]],
    seed: int = 0,
) -> list[str]:
    """Return a stable answer-independent fallback permutation."""
    digest = hashlib.sha256(f"{seed}:{example_id}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest, byteorder="big"))
    visual_ids = [candidate["visual_id"] for candidate in candidate_labels]
    rng.shuffle(visual_ids)
    return visual_ids


def build_vlm_predictions(
    examples: Iterable[dict[str, Any]],
    vlm_inputs: Iterable[dict[str, Any]],
    model_outputs: Iterable[dict[str, Any]],
    invalid_output_policy: str = "error",
    fallback_seed: int = 0,
) -> list[dict[str, Any]]:
    """Join examples, manifests, and raw outputs into prediction rows."""
    if invalid_output_policy not in {"error", "seeded-random"}:
        raise ValueError(
            "invalid_output_policy must be 'error' or 'seeded-random'."
        )
    example_by_id = _index_unique(examples, "ranking examples")
    input_by_id = _index_unique(vlm_inputs, "VLM inputs")
    output_by_id = _index_unique(model_outputs, "model outputs")
    expected_ids = set(input_by_id)
    if set(output_by_id) != expected_ids:
        missing = sorted(expected_ids - set(output_by_id))
        extra = sorted(set(output_by_id) - expected_ids)
        raise ValueError(f"Model-output ID mismatch; missing={missing}, extra={extra}")
    if not expected_ids <= set(example_by_id):
        missing = sorted(expected_ids - set(example_by_id))
        raise ValueError(f"Missing ranking examples for: {missing}")

    predictions = []
    for example_id, vlm_input in input_by_id.items():
        example = example_by_id[example_id]
        candidate_labels = vlm_input["candidate_labels"]
        _validate_candidate_map(example, candidate_labels)
        raw_output = output_by_id[example_id]
        response = raw_output.get("response", raw_output.get("ranking"))
        try:
            ranked_labels = parse_ranking_response(response)
            ranked_visual_ids = complete_candidate_ranking(
                ranked_labels,
                candidate_labels,
            )
        except ValueError:
            if invalid_output_policy == "error":
                raise
            ranked_visual_ids = seeded_random_candidate_ranking(
                example_id,
                candidate_labels,
                seed=fallback_seed,
            )
        predictions.append(
            {
                "example_id": example_id,
                "frame_id": example["frame_id"],
                "mention": example["mention"],
                "gold_visual_ids": example["gold_visual_ids"],
                "gold_object_types": example.get("gold_object_types", []),
                "image_size": example.get("image_size"),
                "candidates": example["candidates"],
                "ranked_visual_ids": ranked_visual_ids,
                "scores": {},
                "baseline": (
                    f"frozen_vlm:{vlm_input['prompt_version']}:"
                    f"{vlm_input['condition']}"
                ),
            }
        )
    return predictions


def _index_unique(
    rows: Iterable[dict[str, Any]],
    source_name: str,
) -> dict[str, dict[str, Any]]:
    indexed = {}
    for row in rows:
        example_id = row["example_id"]
        if example_id in indexed:
            raise ValueError(f"Duplicate example_id in {source_name}: {example_id}")
        indexed[example_id] = row
    return indexed


def _validate_candidate_map(
    example: dict[str, Any],
    candidate_labels: list[dict[str, str]],
) -> None:
    mapped_ids = [candidate["visual_id"] for candidate in candidate_labels]
    expected_ids = [candidate["visual_id"] for candidate in example["candidates"]]
    if mapped_ids != expected_ids:
        raise ValueError(
            f"Candidate map does not match ranking example {example['example_id']}."
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", required=True, help="Ranking examples JSONL.")
    parser.add_argument("--vlm-inputs", required=True, help="VLM input manifest JSONL.")
    parser.add_argument(
        "--model-outputs", required=True, help="Raw model output JSONL."
    )
    parser.add_argument("--output", required=True, help="Prediction JSONL.")
    parser.add_argument(
        "--invalid-output-policy",
        choices=("error", "seeded-random"),
        default="error",
        help=(
            "Policy for malformed responses or unknown labels. The seeded-random "
            "fallback replaces the entire row with an answer-independent stable "
            "permutation."
        ),
    )
    parser.add_argument(
        "--fallback-seed",
        type=int,
        default=0,
        help="Seed used by --invalid-output-policy seeded-random.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    predictions = build_vlm_predictions(
        read_jsonl(args.examples),
        read_jsonl(args.vlm_inputs),
        read_jsonl(args.model_outputs),
        invalid_output_policy=args.invalid_output_policy,
        fallback_seed=args.fallback_seed,
    )
    write_jsonl(args.output, predictions)
