"""Ranking metrics for mention-to-box prediction."""

import argparse
import json
import statistics
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from mathref.utils.io import read_jsonl, write_json
from mathref.eval.spatial import spatial_metrics_for_row


def normalize_gold_ids(gold_ids: str | Iterable[str]) -> set[str]:
    if isinstance(gold_ids, str):
        return {gold_ids}
    return set(gold_ids)


def reciprocal_rank(ranked_ids: list[str], gold_id: str | Iterable[str]) -> float:
    gold_ids = normalize_gold_ids(gold_id)
    for index, candidate_id in enumerate(ranked_ids, start=1):
        if candidate_id in gold_ids:
            return 1.0 / index
    return 0.0


def recall_at_k(ranked_ids: list[str], gold_id: str | Iterable[str], k: int) -> float:
    gold_ids = normalize_gold_ids(gold_id)
    return float(any(candidate_id in gold_ids for candidate_id in ranked_ids[:k]))


def evaluate_predictions(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, float] = defaultdict(float)
    breakdowns: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    spatial_totals: dict[str, list[float]] = defaultdict(list)
    spatial_breakdowns: dict[
        str, dict[str, dict[str, list[float]]]
    ] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        ranked_ids = row["ranked_visual_ids"]
        gold_ids = row["gold_visual_ids"]
        metrics = {
            "recall_at_1": recall_at_k(ranked_ids, gold_ids, 1),
            "recall_at_3": recall_at_k(ranked_ids, gold_ids, 3),
            "mrr": reciprocal_rank(ranked_ids, gold_ids),
        }
        totals["count"] += 1
        for name, value in metrics.items():
            totals[name] += value
        spatial_metrics = spatial_metrics_for_row(row)
        if spatial_metrics is not None:
            for name, value in spatial_metrics.items():
                spatial_totals[name].append(value)
        mention_type = row.get("mention", {}).get("mention_type", "unknown")
        if mention_type not in {"implicit", "explicit", "unknown"}:
            mention_type = "unknown"
        candidate_count = len(row.get("candidates", []))
        bucket_labels = {
            "by_gold_object_type": row.get("gold_object_types") or ["unknown"],
            "by_mention_type": [mention_type],
            "by_candidate_count": [_candidate_count_bucket(candidate_count)],
            "by_gold_count": [
                "single_gold" if len(gold_ids) == 1 else "multi_gold"
            ],
        }
        for breakdown_name, labels in bucket_labels.items():
            for label in labels:
                bucket = breakdowns[breakdown_name][label]
                bucket["count"] += 1
                for name, value in metrics.items():
                    bucket[name] += value
                if spatial_metrics is not None:
                    for name, value in spatial_metrics.items():
                        spatial_breakdowns[breakdown_name][label][name].append(value)

    result = _average_bucket(totals)
    result["spatial"] = _average_spatial_bucket(spatial_totals, int(totals["count"]))
    for breakdown_name in (
        "by_gold_object_type",
        "by_mention_type",
        "by_candidate_count",
        "by_gold_count",
    ):
        result[breakdown_name] = {
            label: {
                **_average_bucket(bucket),
                "spatial": _average_spatial_bucket(
                    spatial_breakdowns[breakdown_name][label],
                    int(bucket["count"]),
                ),
            }
            for label, bucket in sorted(breakdowns[breakdown_name].items())
        }
    return result


def _candidate_count_bucket(candidate_count: int) -> str:
    if candidate_count <= 5:
        return "1-5"
    if candidate_count <= 10:
        return "6-10"
    if candidate_count <= 15:
        return "11-15"
    return "16+"


def _average_bucket(bucket: dict[str, float]) -> dict[str, float]:
    count = int(bucket.get("count", 0))
    if count == 0:
        return {"count": 0, "recall_at_1": 0.0, "recall_at_3": 0.0, "mrr": 0.0}
    return {
        "count": count,
        "recall_at_1": bucket["recall_at_1"] / count,
        "recall_at_3": bucket["recall_at_3"] / count,
        "mrr": bucket["mrr"] / count,
    }


def _average_spatial_bucket(
    bucket: dict[str, list[float]], total_count: int
) -> dict[str, float | int]:
    distances = bucket.get("normalized_center_distance", [])
    spatial_count = len(distances)
    result: dict[str, float | int] = {
        "count": spatial_count,
        "coverage": spatial_count / total_count if total_count else 0.0,
    }
    if not spatial_count:
        return result
    for name, values in sorted(bucket.items()):
        result[name] = sum(values) / len(values)
    result["median_normalized_center_distance"] = statistics.median(distances)
    return result


def build_meta(args: argparse.Namespace) -> dict[str, Any] | None:
    """Provenance block for the reporting layer; attached only if --method is set."""
    if not args.method:
        return None
    return {
        "method": args.method,
        "model": args.model,
        "prompt_version": args.prompt_version,
        "seed": args.seed,
        "split": args.split,
        "condition": args.condition,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Path to ranked predictions JSONL.")
    parser.add_argument("--output", help="Optional path to write evaluation JSON.")
    parser.add_argument("--method", help="Method id for reporting, e.g. random, class_prior, frozen_vlm.")
    parser.add_argument("--model", default=None, help="Model name (frozen-model methods).")
    parser.add_argument("--prompt-version", default=None, help="Prompt version (VLM methods).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for this run.")
    parser.add_argument("--split", default=None, help="Split id, e.g. test@v1.")
    parser.add_argument("--condition", default=None, help="Ablation condition, e.g. vlm or vlm+latex.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    # Persisted results feed the reporting layer, so they must carry provenance.
    if args.output and not args.method:
        parser.error("--output requires --method (and ideally --seed/--split/--condition) for reporting.")
    result = evaluate_predictions(read_jsonl(args.predictions))
    meta = build_meta(args)
    if meta is not None:
        result = {**result, "meta": meta}
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
