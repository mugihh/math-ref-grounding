"""Parse annotation JSON files into normalized frame JSONL."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mathref.data.parse_annotation import write_frames_jsonl


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Annotation directory or one .jpg.json file.")
    parser.add_argument("--output", required=True, help="Path to normalized frame JSONL.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    write_frames_jsonl(args.input, args.output)


if __name__ == "__main__":
    main()
