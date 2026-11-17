"""Validate annotation JSON files and report likely data issues."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mathref.data.validate import validate_annotation_dir
from mathref.utils.io import write_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Annotation directory or one .jpg.json file.")
    parser.add_argument("--output", help="Optional path to write validation report JSON.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = validate_annotation_dir(args.input)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
