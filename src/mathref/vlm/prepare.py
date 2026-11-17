"""Build model-ready frozen-VLM input manifests."""

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from mathref.utils.io import read_jsonl, write_jsonl
from mathref.vlm.prompts import CONDITION_PROMPTS, render_prompt
from mathref.vlm.render import render_candidate_labels, rendered_image_name


def build_vlm_input(
    example: dict[str, Any],
    labeled_image_path: str | Path,
    candidate_labels: list[dict[str, str]],
    prompt_version: str,
    condition: str = "vlm",
    candidate_latex: list[str | None] | None = None,
) -> dict[str, Any]:
    """Build one manifest row without copying gold-answer fields."""
    labels = [candidate["label"] for candidate in candidate_labels]
    return {
        "example_id": example["example_id"],
        "frame_id": example["frame_id"],
        "labeled_image_path": str(labeled_image_path),
        "prompt_version": prompt_version,
        "condition": condition,
        "candidate_labels": candidate_labels,
        "prompt": render_prompt(example, labels, prompt_version, candidate_latex=candidate_latex),
    }


def prepare_vlm_inputs(
    examples: Iterable[dict[str, Any]],
    rendered_dir: str | Path,
    condition: str = "vlm",
    prompt_version: str | None = None,
    images_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Render labeled images and return their model-input manifest rows.

    `condition` selects the ablation: "vlm" or "vlm+latex". The prompt version
    defaults to the one registered for the condition unless overridden.
    """
    if condition not in CONDITION_PROMPTS:
        raise ValueError(f"Unknown condition {condition!r}; choose from {sorted(CONDITION_PROMPTS)}.")
    version = prompt_version or CONDITION_PROMPTS[condition]
    output_dir = Path(rendered_dir)
    rows = []
    seen_ids: set[str] = set()
    for example in examples:
        example_id = example["example_id"]
        if example_id in seen_ids:
            raise ValueError(f"Duplicate example_id: {example_id}")
        seen_ids.add(example_id)
        image_path = output_dir / rendered_image_name(example_id)
        candidate_labels = render_candidate_labels(example, image_path, images_root=images_root)
        candidate_latex: list[str | None] | None = None
        if condition == "vlm+latex":
            candidate_latex = [candidate.get("latex") for candidate in example["candidates"]]
            for label, latex in zip(candidate_labels, candidate_latex, strict=True):
                label["latex"] = latex
        rows.append(
            build_vlm_input(
                example,
                image_path,
                candidate_labels,
                version,
                condition=condition,
                candidate_latex=candidate_latex,
            )
        )
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", required=True, help="Ranking examples JSONL.")
    parser.add_argument(
        "--rendered-dir", required=True, help="Labeled-image directory."
    )
    parser.add_argument("--output", required=True, help="VLM input manifest JSONL.")
    parser.add_argument("--condition", choices=sorted(CONDITION_PROMPTS), default="vlm")
    parser.add_argument("--prompt-version", default=None, help="Override the condition's default prompt.")
    parser.add_argument("--images-root", default=None, help="Root prepended to relative image paths.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = prepare_vlm_inputs(
        read_jsonl(args.examples),
        args.rendered_dir,
        condition=args.condition,
        prompt_version=args.prompt_version,
        images_root=args.images_root,
    )
    write_jsonl(args.output, rows)
