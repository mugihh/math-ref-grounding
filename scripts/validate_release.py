"""Validate the public benchmark and its release-level invariants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_EXAMPLES = 1842
EXPECTED_FRAMES = 206


def _jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def _validate_jsonl(
    path: Path, validator: Draft202012Validator
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, row in _jsonl(path):
        errors = sorted(validator.iter_errors(row), key=lambda error: list(error.path))
        if errors:
            detail = "; ".join(error.message for error in errors[:3])
            raise ValueError(f"{path}:{line_number}: {detail}")
        rows.append(row)
    return rows


def validate_release(root: Path = ROOT) -> dict[str, int]:
    schema = json.loads(
        (root / "schemas/ranking_example.schema.json").read_text(encoding="utf-8")
    )
    validator = Draft202012Validator(schema)
    plain = _validate_jsonl(root / "data/benchmark/examples.jsonl", validator)
    latex = _validate_jsonl(root / "data/benchmark/examples_latex.jsonl", validator)

    if len(plain) != EXPECTED_EXAMPLES or len(latex) != EXPECTED_EXAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_EXAMPLES} rows per condition; "
            f"found {len(plain)} plain and {len(latex)} LaTeX rows"
        )

    plain_ids = [row["example_id"] for row in plain]
    latex_ids = [row["example_id"] for row in latex]
    if len(set(plain_ids)) != len(plain_ids):
        raise ValueError("Duplicate example_id in examples.jsonl")
    if plain_ids != latex_ids:
        raise ValueError("Plain and LaTeX files are not aligned by example_id")

    for index, (plain_row, latex_row) in enumerate(zip(plain, latex), start=1):
        for field in plain_row:
            if field != "candidates" and plain_row[field] != latex_row[field]:
                raise ValueError(
                    f"Row {index} differs outside candidates: field {field!r}"
                )
        plain_candidates = plain_row["candidates"]
        latex_candidates = latex_row["candidates"]
        if len(plain_candidates) != len(latex_candidates):
            raise ValueError(f"Row {index} candidate counts differ")
        for plain_candidate, latex_candidate in zip(
            plain_candidates, latex_candidates
        ):
            stripped = {
                key: value
                for key, value in latex_candidate.items()
                if key != "latex"
            }
            if plain_candidate != stripped:
                raise ValueError(
                    f"Row {index} candidate metadata differs outside LaTeX"
                )

    manifest_paths = {
        row["image_path"]
        for _, row in _jsonl(root / "data/keyframes/keyframes.jsonl")
    }
    benchmark_paths = {row["image_path"] for row in plain}
    missing = benchmark_paths - manifest_paths
    if missing:
        raise ValueError(f"Image paths absent from keyframe manifest: {sorted(missing)}")
    if len(benchmark_paths) != EXPECTED_FRAMES:
        raise ValueError(
            f"Expected {EXPECTED_FRAMES} benchmark frames; "
            f"found {len(benchmark_paths)}"
        )

    return {
        "examples": len(plain),
        "frames": len(benchmark_paths),
        "manifest_frames": len(manifest_paths),
    }


def main() -> None:
    print(json.dumps(validate_release(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

