"""Run a non-learned mention-to-box ranking baseline."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mathref.baselines import class_prior, random


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=["random", "class-prior"], default="random")
    parser.add_argument("--examples", required=True, help="Path to ranking examples JSONL.")
    parser.add_argument(
        "--prior-examples",
        help="Training examples used to estimate class prior; required for class-prior.",
    )
    parser.add_argument("--output", required=True, help="Path to prediction output JSONL.")
    parser.add_argument("--seed", type=int, default=13)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.baseline == "random":
        random.main_with_args(args.examples, args.output, seed=args.seed)
        return
    if not args.prior_examples:
        raise SystemExit("--prior-examples is required for the class-prior baseline.")
    class_prior.main_with_args(
        args.examples,
        args.prior_examples,
        args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
