"""Generate provisional dataset statistics."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mathref.data.build_examples import load_frames
from mathref.data.stats import summarize_examples, summarize_frames
from mathref.utils.io import read_jsonl, write_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Annotation directory, frames JSONL, or examples JSONL.")
    parser.add_argument(
        "--input-format",
        choices=["annotation-dir", "frames-jsonl", "examples-jsonl"],
        default="annotation-dir",
    )
    parser.add_argument("--output", help="Optional path to write stats JSON.")
    parser.add_argument(
        "--annotations",
        help=(
            "Optional raw annotation directory. With --input-format examples-jsonl, "
            "adds per-language annotated/usable/empty frame coverage."
        ),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.input_format == "examples-jsonl":
        examples = read_jsonl(args.input)
        if args.annotations:
            from mathref.data.stats import summarize_dataset

            stats = summarize_dataset(
                examples,
                load_frames(args.annotations, "annotation-dir"),
            )
        else:
            stats = summarize_examples(examples)
    else:
        stats = summarize_frames(load_frames(args.input, args.input_format))
    if args.output:
        write_json(args.output, stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
