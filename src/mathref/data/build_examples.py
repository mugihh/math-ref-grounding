"""Build mention-to-box ranking examples."""

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from mathref.data.mention_type import classify_mention_type
from mathref.data.parse_annotation import parse_annotation_dir
from mathref.utils.io import read_jsonl, write_jsonl


def build_examples(frames: Iterable[dict[str, Any]], include_no_positive: bool = False) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for frame in frames:
        candidates = [
            {
                "visual_id": box["visual_id"],
                "object_type": box["object_type"],
                "bbox": box["bbox"],
                "source": box.get("source", "gold"),
            }
            for box in frame.get("boxes", [])
        ]
        candidate_ids = {candidate["visual_id"] for candidate in candidates}
        for mention in frame.get("mentions", []):
            gold_ids = [
                visual_id
                for visual_id in mention.get("gold_visual_ids", [])
                if visual_id in candidate_ids
            ]
            if not gold_ids and not include_no_positive:
                continue
            examples.append(
                {
                    "example_id": mention["mention_id"],
                    "frame_id": frame["frame_id"],
                    "language": frame["language"],
                    "topic": frame["topic"],
                    "video": frame["video"],
                    "image_path": frame["image_path"],
                    "image_size": frame.get("image_size"),
                    "caption": frame["caption"],
                    "mention": {
                        "mention_id": mention["mention_id"],
                        "text": mention["text"],
                        "span": mention["span"],
                        "mention_type": classify_mention_type(
                            mention["text"], frame["language"]
                        ),
                    },
                    "gold_visual_ids": gold_ids,
                    "gold_object_types": mention.get("gold_object_types", []),
                    "candidates": candidates,
                }
            )
    return examples


def load_frames(input_path: str | Path, input_format: str) -> list[dict[str, Any]]:
    if input_format == "frames-jsonl":
        return list(read_jsonl(input_path))
    if input_format == "annotation-dir":
        return parse_annotation_dir(input_path)
    raise ValueError(f"Unsupported input format: {input_format}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to source annotation directory or frames JSONL.")
    parser.add_argument("--output", required=True, help="Path to output examples JSONL.")
    parser.add_argument(
        "--input-format",
        choices=["annotation-dir", "frames-jsonl"],
        default="annotation-dir",
    )
    parser.add_argument("--include-no-positive", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    frames = load_frames(args.input, args.input_format)
    write_jsonl(args.output, build_examples(frames, include_no_positive=args.include_no_positive))
