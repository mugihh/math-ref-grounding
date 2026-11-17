"""Audit raw frozen-VLM rankings before parser completion hides omissions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mathref.utils.io import read_jsonl
from mathref.vlm.parse import parse_ranking_response


def audit_vlm_outputs(
    vlm_inputs: list[dict[str, Any]],
    model_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    inputs_by_id = {row["example_id"]: row for row in vlm_inputs}
    outputs_by_id = {row["example_id"]: row for row in model_outputs}
    details = []
    counts = {
        "total": len(inputs_by_id),
        "missing_output": 0,
        "parse_failure": 0,
        "unknown_labels": 0,
        "duplicate_labels": 0,
        "incomplete_rankings": 0,
        "complete_rankings": 0,
    }

    for example_id, row in inputs_by_id.items():
        allowed = [candidate["label"] for candidate in row["candidate_labels"]]
        output = outputs_by_id.get(example_id)
        issues: list[str] = []
        returned: list[str] = []
        if output is None:
            counts["missing_output"] += 1
            issues.append("missing_output")
        else:
            try:
                returned = parse_ranking_response(
                    output.get("response", output.get("ranking"))
                )
            except ValueError:
                counts["parse_failure"] += 1
                issues.append("parse_failure")

        if output is not None and "parse_failure" not in issues:
            unknown = sorted(set(returned) - set(allowed))
            if unknown:
                counts["unknown_labels"] += 1
                issues.append("unknown_labels")
            if len(returned) != len(set(returned)):
                counts["duplicate_labels"] += 1
                issues.append("duplicate_labels")
            if set(returned) != set(allowed) or len(returned) != len(allowed):
                counts["incomplete_rankings"] += 1
                issues.append("incomplete_ranking")
            elif not issues:
                counts["complete_rankings"] += 1

        if issues:
            details.append(
                {
                    "example_id": example_id,
                    "candidate_count": len(allowed),
                    "returned_count": len(returned),
                    "issues": issues,
                }
            )

    total = counts["total"]
    counts["incomplete_rate"] = (
        counts["incomplete_rankings"] / total if total else 0.0
    )
    extra_ids = sorted(set(outputs_by_id) - set(inputs_by_id))
    return {**counts, "extra_output_ids": extra_ids, "details": details}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlm-inputs", required=True)
    parser.add_argument("--model-outputs", required=True)
    parser.add_argument("--out", default=None, help="Optional JSON report path.")
    parser.add_argument("--max-incomplete-rate", type=float, default=0.10)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    report = audit_vlm_outputs(
        list(read_jsonl(args.vlm_inputs)),
        list(read_jsonl(args.model_outputs)),
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)

    failed = (
        report["missing_output"] > 0
        or report["parse_failure"] > 0
        or report["unknown_labels"] > 0
        or report["incomplete_rate"] > args.max_incomplete_rate
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
