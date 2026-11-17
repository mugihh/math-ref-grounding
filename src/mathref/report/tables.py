"""Assemble the specific paper tables from grouped evaluation results."""

from __future__ import annotations

from typing import Any

from mathref.report.aggregate import aggregate_cell, group_rows, seed_caption
from mathref.report.render import Table

# (column header, metric key)
MAIN_METRICS = [("R@1", "recall_at_1"), ("R@3", "recall_at_3"), ("MRR", "mrr")]
SPATIAL_METRICS = [
    ("Coverage", "coverage"),
    ("IoU", "iou"),
    ("Med. dist", "median_normalized_center_distance"),
    ("dist@0.05", "center_distance_at_0_05"),
    ("dist@0.10", "center_distance_at_0_10"),
]
# breakdown dimension -> (result key, ordered buckets)
BREAKDOWNS = {
    "object_type": ("by_gold_object_type", ["equation", "span", "term", "figure", "figure_part"]),
    "mention_type": ("by_mention_type", ["implicit", "explicit", "unknown"]),
    "candidate_count": ("by_candidate_count", ["1-5", "6-10", "11-15", "16+"]),
}
BREAKDOWN_METRICS = [("recall_at_1", "R@1"), ("mrr", "MRR")]


def main_table(rows: list[dict[str, Any]]) -> Table:
    headers = ["Method"] + [h for h, _ in MAIN_METRICS]
    table_rows = [
        [row["display"]] + [aggregate_cell(row["results"], (key,)) for _, key in MAIN_METRICS]
        for row in rows
    ]
    return Table(
        title="Main results",
        headers=headers,
        rows=table_rows,
        caption=f"Mention-to-box ranking. {seed_caption(rows)}",
        label="tab:main",
    )


def spatial_table(rows: list[dict[str, Any]]) -> Table:
    headers = ["Method"] + [h for h, _ in SPATIAL_METRICS]
    table_rows = [
        [row["display"]] + [aggregate_cell(row["results"], ("spatial", key)) for _, key in SPATIAL_METRICS]
        for row in rows
    ]
    return Table(
        title="Auxiliary spatial evaluation",
        headers=headers,
        rows=table_rows,
        caption=(
            "Auxiliary spatial agreement between the selected gold candidate and the gold box; "
            f"this is not detection evaluation. {seed_caption(rows)}"
        ),
        label="tab:spatial",
    )


def breakdown_table(rows: list[dict[str, Any]], dimension: str, metric: str, metric_label: str) -> Table:
    result_key, buckets = BREAKDOWNS[dimension]
    headers = ["Method"] + buckets
    table_rows = [
        [row["display"]]
        + [aggregate_cell(row["results"], (result_key, bucket, metric)) for bucket in buckets]
        for row in rows
    ]
    return Table(
        title=f"Breakdown by {dimension.replace('_', ' ')} ({metric_label})",
        headers=headers,
        rows=table_rows,
        caption=f"{metric_label} per {dimension.replace('_', ' ')} bucket. {seed_caption(rows)}",
        label=f"tab:bd_{dimension}_{metric_label.lower().replace('@', '')}",
    )


def gold_count_table(rows: list[dict[str, Any]]) -> Table:
    buckets = ["single_gold", "multi_gold"]
    headers = ["Method"]
    for bucket in buckets:
        prefix = "Single" if bucket == "single_gold" else "Multi"
        headers += [f"{prefix} {h}" for h, _ in MAIN_METRICS]
    table_rows = []
    for row in rows:
        cells: list[Any] = [row["display"]]
        for bucket in buckets:
            for _, key in MAIN_METRICS:
                cells.append(aggregate_cell(row["results"], ("by_gold_count", bucket, key)))
        table_rows.append(cells)
    return Table(
        title="Single-gold vs multi-gold",
        headers=headers,
        rows=table_rows,
        caption=(
            "Recall@k accepts any gold id; MRR uses the highest-ranked gold. "
            f"{seed_caption(rows)}"
        ),
        label="tab:gold_count",
    )


def all_tables(results: list[dict[str, Any]]) -> dict[str, Table]:
    """Build every paper table keyed by a slug (used for output filenames)."""
    rows = group_rows(results)
    tables: dict[str, Table] = {
        "main": main_table(rows),
        "spatial": spatial_table(rows),
        "gold_count": gold_count_table(rows),
    }
    for dimension in BREAKDOWNS:
        for metric, label in BREAKDOWN_METRICS:
            slug = f"breakdown_{dimension}_{label.lower().replace('@', '')}"
            tables[slug] = breakdown_table(rows, dimension, metric, label)
    return tables
