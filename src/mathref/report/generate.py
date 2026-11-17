"""Generate paper-ready tables (Markdown + LaTeX) from evaluation JSON files."""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

from mathref.report.aggregate import load_results, unmatched_methods
from mathref.report.render import render_latex, render_markdown
from mathref.report.tables import all_tables


def generate(
    eval_paths: list[str | Path],
    out_dir: str | Path,
    percent: bool = False,
    decimals: int | None = None,
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = load_results(eval_paths)
    dropped = unmatched_methods(results)
    if dropped:
        pairs = ", ".join(f"{method}/{condition}" for method, condition in dropped)
        print(
            f"WARNING: {len(dropped)} (method, condition) not in METHOD_ROWS, omitted from tables: {pairs}",
            file=sys.stderr,
        )
    tables = all_tables(results)

    written: dict[str, Path] = {}
    combined = ["# Results\n"]
    for slug, table in tables.items():
        md = render_markdown(table, percent=percent, decimals=decimals)
        tex = render_latex(table, percent=percent, decimals=decimals)
        (out / f"{slug}.md").write_text(md, encoding="utf-8")
        (out / f"{slug}.tex").write_text(tex, encoding="utf-8")
        written[slug] = out / f"{slug}.tex"
        combined.append(md)
    (out / "report.md").write_text("\n".join(combined), encoding="utf-8")
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", help="Directory of evaluation JSON files (globs *.json).")
    parser.add_argument("--eval", nargs="*", default=[], help="Explicit evaluation JSON paths.")
    parser.add_argument("--out", required=True, help="Output directory for tables.")
    parser.add_argument("--percent", action="store_true", help="Show metrics as percentages (52.0 vs 0.520).")
    parser.add_argument("--decimals", type=int, default=None, help="Decimal places (default 3, or 1 with --percent).")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    paths: list[str | Path] = list(args.eval)
    if args.eval_dir:
        paths += sorted(str(p) for p in Path(args.eval_dir).glob("*.json"))
    if not paths:
        raise SystemExit("No evaluation files: pass --eval-dir or --eval.")
    written = generate(paths, args.out, percent=args.percent, decimals=args.decimals)
    print(f"Wrote {len(written)} tables (.md + .tex) to {args.out}")


if __name__ == "__main__":
    main()
