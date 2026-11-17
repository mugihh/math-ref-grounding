"""Load evaluation results and aggregate metrics over seeds."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from mathref.report.render import MISSING, Num
from mathref.utils.io import read_json

# Canonical paper rows: (method, condition, display name). Condition None matches any.
METHOD_ROWS: list[tuple[str, str | None, str]] = [
    ("random", None, "Random"),
    ("class_prior", None, "Class-prior"),
    ("clip_dual_encoder", "plain", "CLIP dual-encoder"),
    ("clip_dual_encoder", "latex", "CLIP dual-encoder + LaTeX"),
]

VLM_CONDITIONS = (("vlm", ""), ("vlm+latex", " + LaTeX"))


def _vlm_display(model: str | None) -> str:
    if not model:
        return "Frozen VLM"
    normalized = model.lower()
    if "gpt-5.4" in normalized:
        return "GPT-5.4"
    if "qwen2.5-vl-7b" in normalized:
        return "Qwen2.5-VL-7B"
    if "sarashina" in normalized:
        return "Sarashina2-Vision"
    return model


def load_results(paths: list[str | Path]) -> list[dict[str, Any]]:
    """Load evaluation JSON files; every file must carry a `meta.method` block."""
    results = []
    for path in paths:
        result = read_json(path)
        meta = result.get("meta")
        if not meta or not meta.get("method"):
            raise ValueError(
                f"{path} has no meta.method; rerun eval with --method (and --seed/--condition)."
            )
        results.append(result)
    return results


def _matches(meta: dict[str, Any], method: str, condition: str | None) -> bool:
    if meta.get("method") != method:
        return False
    return condition is None or meta.get("condition") == condition


def group_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group loaded results into the canonical method rows, keeping seed lists."""
    rows = []
    for method, condition, display in METHOD_ROWS:
        matched = [r for r in results if _matches(r.get("meta", {}), method, condition)]
        if matched:
            rows.append({"display": display, "results": matched})
    vlm_models = sorted(
        {r.get("meta", {}).get("model") for r in results if r.get("meta", {}).get("method") == "frozen_vlm"},
        key=lambda model: (_vlm_display(model), model or ""),
    )
    for model in vlm_models:
        for condition, suffix in VLM_CONDITIONS:
            matched = [
                r
                for r in results
                if _matches(r.get("meta", {}), "frozen_vlm", condition)
                and r.get("meta", {}).get("model") == model
            ]
            if matched:
                rows.append({"display": f"{_vlm_display(model)}{suffix}", "results": matched})
    return rows


def unmatched_methods(results: list[dict[str, Any]]) -> list[tuple[str, str | None]]:
    """(method, condition) pairs in the eval files that no canonical row covers.

    These would otherwise vanish from the tables silently; the reporter warns.
    """
    seen = {(r.get("meta", {}).get("method"), r.get("meta", {}).get("condition")) for r in results}
    covered = {
        (method, condition)
        for method, condition in seen
        if (
            any(_matches({"method": method, "condition": condition}, m, c) for m, c, _ in METHOD_ROWS)
            or (method == "frozen_vlm" and condition in {item[0] for item in VLM_CONDITIONS})
        )
    }
    return sorted(seen - covered, key=lambda pair: (pair[0] or "", pair[1] or ""))


def get_value(result: dict[str, Any], path: tuple[str, ...]) -> float | None:
    """Walk a nested key path; return None if any key is absent."""
    node: Any = result
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node if isinstance(node, (int, float)) else None


def aggregate_cell(results: list[dict[str, Any]], path: tuple[str, ...]) -> Num | str:
    """Mean (and sample std over >1 run) of a metric across a row's results."""
    values = [v for r in results if (v := get_value(r, path)) is not None]
    if not values:
        return MISSING
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else None
    return Num(mean, std)


def seed_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {row["display"]: len(row["results"]) for row in rows}


def seed_caption(rows: list[dict[str, Any]]) -> str:
    counts = sorted({len(row["results"]) for row in rows})
    multi = [c for c in counts if c > 1]
    if not multi:
        return "Single run per method."
    span = f"{min(multi)}" if len(multi) == 1 else f"{min(multi)}–{max(multi)}"
    return f"± is the standard deviation over {span} seeds."
