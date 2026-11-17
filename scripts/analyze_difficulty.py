#!/usr/bin/env python3
"""Post-hoc difficulty, ambiguity, interaction, and clustered-bootstrap analysis."""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-plain", required=True, type=Path)
    parser.add_argument("--predictions-latex", required=True, type=Path)
    parser.add_argument("--reference-logs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalized_mention(text: str) -> str:
    text = text.replace("<", "").replace(">", "").strip().lower()
    text = re.sub(r"_(?:\d+)$", "", text)
    return re.sub(r"\s+", " ", text)


def reference_targets(root: Path) -> dict[str, list[str]]:
    targets: dict[str, list[str]] = defaultdict(list)
    pattern = str(root / "*" / "*" / "*" / "video_reference_log.txt")
    for raw_path in glob.glob(pattern):
        path = Path(raw_path)
        language, topic, video = path.parts[-4:-1]
        prefix = f"{language}/{topic}/{video}"
        text = path.read_text(errors="replace")
        for block in re.split(r"\[video_ref\]", text)[1:]:
            keyframe = re.search(r"keyframe\s*=\s*(\d{3})\.jpg", block)
            if not keyframe:
                continue
            frame_id = f"{prefix}/{keyframe.group(1)}"
            for target in re.findall(r'target\s*=\s*["“](.*?)["”]', block):
                targets[frame_id].append(normalized_mention(target))
    return targets


def area(box: list[float]) -> float:
    return max(0.0, box[2]) * max(0.0, box[3])


def intersection(a: list[float], b: list[float]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[0] + a[2], b[0] + b[2])
    bottom = min(a[1] + a[3], b[1] + b[3])
    return max(0.0, right - left) * max(0.0, bottom - top)


def iou(a: list[float], b: list[float]) -> float:
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union else 0.0


def center(box: list[float]) -> tuple[float, float]:
    return box[0] + box[2] / 2, box[1] + box[3] / 2


def point_inside(point: tuple[float, float], box: list[float]) -> bool:
    x, y = point
    return box[0] <= x <= box[0] + box[2] and box[1] <= y <= box[1] + box[3]


def latex_key(value: str | None) -> str:
    return re.sub(r"\s+", "", (value or "").lower())


def best_gold(row: dict) -> dict:
    candidate_map = {candidate["visual_id"]: candidate for candidate in row["candidates"]}
    return candidate_map[row["gold_visual_ids"][0]]


def feature_row(row: dict, logged_targets: dict[str, list[str]]) -> dict:
    gold = best_gold(row)
    others = [
        candidate
        for candidate in row["candidates"]
        if candidate["visual_id"] not in row["gold_visual_ids"]
    ]
    gold_box = gold["bbox"]
    width, height = row["image_size"]
    image_area = width * height
    gold_latex = latex_key(gold.get("latex"))
    overlap_count = sum(intersection(gold_box, candidate["bbox"]) > 0 for candidate in others)
    containing_count = sum(
        point_inside(center(gold_box), candidate["bbox"]) for candidate in others
    )
    contained_count = sum(
        point_inside(center(candidate["bbox"]), gold_box) for candidate in others
    )
    same_type_count = sum(candidate["object_type"] == gold["object_type"] for candidate in others)
    duplicate_latex_count = (
        sum(latex_key(candidate.get("latex")) == gold_latex for candidate in others)
        if gold_latex
        else 0
    )
    mention = normalized_mention(row["mention"]["text"])
    correct = row["ranked_visual_ids"][0] in row["gold_visual_ids"]
    return {
        "example_id": row["example_id"],
        "video_id": "/".join(row["frame_id"].split("/")[:3]),
        "language": row["frame_id"].split("/")[0],
        "topic": row["frame_id"].split("/")[1],
        "mention_type": row["mention"]["mention_type"],
        "implicit": int(row["mention"]["mention_type"] == "implicit"),
        "logged_target": int(mention in logged_targets.get(row["frame_id"], [])),
        "gold_type": gold["object_type"],
        "candidate_count": len(row["candidates"]),
        "gold_area_fraction": area(gold_box) / image_area if image_area else 0.0,
        "overlap_count": overlap_count,
        "containing_count": containing_count,
        "contained_count": contained_count,
        "same_type_count": same_type_count,
        "duplicate_latex_count": duplicate_latex_count,
        "max_non_gold_iou": max((iou(gold_box, c["bbox"]) for c in others), default=0.0),
        "correct": int(correct),
    }


def zscore(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        mean = frame[column].mean()
        std = frame[column].std(ddof=0)
        frame[f"z_{column}"] = (frame[column] - mean) / std if std else 0.0
    return frame


def logistic_analysis(frame: pd.DataFrame) -> dict:
    continuous = [
        "candidate_count",
        "gold_area_fraction",
        "overlap_count",
        "same_type_count",
        "duplicate_latex_count",
    ]
    model_frame = zscore(frame, continuous)
    formula = (
        "correct ~ implicit + logged_target + z_candidate_count + "
        "z_gold_area_fraction + z_overlap_count + z_same_type_count + "
        "z_duplicate_latex_count + C(gold_type, Treatment(reference='span')) + "
        "C(language, Treatment(reference='en')) + C(topic, Treatment(reference='calculus'))"
    )
    robust = smf.logit(formula, data=model_frame).fit(
        disp=False,
        cov_type="cluster",
        cov_kwds={"groups": model_frame["video_id"]},
    )
    names = robust.model.exog_names
    coefficients = []
    for name, coefficient, standard_error, pvalue in zip(
        names, robust.params, robust.bse, robust.pvalues
    ):
        low = coefficient - 1.96 * standard_error
        high = coefficient + 1.96 * standard_error
        coefficients.append(
            {
                "term": name,
                "log_odds": round(float(coefficient), 6),
                "odds_ratio": round(float(math.exp(coefficient)), 6),
                "cluster_robust_se": round(float(standard_error), 6),
                "odds_ratio_ci_95": [
                    round(float(math.exp(low)), 6),
                    round(float(math.exp(high)), 6),
                ],
                "p_value": round(float(pvalue), 6),
            }
        )
    return {
        "outcome": "GPT-5.4+LaTeX exact R@1 correctness",
        "formula": formula,
        "standardization": "Continuous predictors are z-scored over all 1,842 examples.",
        "cluster_robust_se": "video_id (21 clusters)",
        "observations": len(model_frame),
        "coefficients": coefficients,
        "caveats": [
            "This is a post-hoc associational model, not a causal analysis.",
            "Only 21 video clusters are available, so cluster-robust intervals are approximate.",
            "Logging, language, topic, source style, and transcript provenance remain partially confounded.",
        ],
    }


def metric(rows: pd.DataFrame, column: str = "correct") -> float:
    return float(rows[column].mean()) if len(rows) else float("nan")


def bootstrap_stat(sample: pd.DataFrame) -> dict:
    return {
        "r1": metric(sample),
        "explicit_minus_implicit": metric(sample[sample["implicit"] == 0])
        - metric(sample[sample["implicit"] == 1]),
        "logged_minus_other": metric(sample[sample["logged_target"] == 1])
        - metric(sample[sample["logged_target"] == 0]),
        "en_minus_jp": metric(sample[sample["language"] == "en"])
        - metric(sample[sample["language"] == "jp"]),
        "equation_minus_term": metric(sample[sample["gold_type"] == "equation"])
        - metric(sample[sample["gold_type"] == "term"]),
        "small_minus_dense": metric(sample[sample["candidate_count"] <= 5])
        - metric(sample[sample["candidate_count"] >= 16]),
    }


def clustered_bootstrap(
    latex_frame: pd.DataFrame,
    plain_frame: pd.DataFrame,
    samples: int,
    seed: int,
) -> dict:
    joined = latex_frame.merge(
        plain_frame[["example_id", "correct"]].rename(columns={"correct": "plain_correct"}),
        on="example_id",
        validate="one_to_one",
    )
    videos = sorted(joined["video_id"].unique())
    rng = np.random.default_rng(seed)
    observed = bootstrap_stat(joined)
    observed["latex_gain"] = metric(joined) - metric(joined, "plain_correct")
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(samples):
        chosen = rng.choice(videos, size=len(videos), replace=True)
        sample = pd.concat(
            [joined[joined["video_id"] == video] for video in chosen],
            ignore_index=True,
        )
        stats = {
            **bootstrap_stat(sample),
            "latex_gain": metric(sample) - metric(sample, "plain_correct"),
        }
        for key, value in stats.items():
            if not math.isnan(value):
                draws[key].append(value)
    output = {}
    for key, values in draws.items():
        output[key] = {
            "estimate": round(float(observed[key]), 6),
            "percentile_ci_95": [
                round(float(np.quantile(values, 0.025)), 6),
                round(float(np.quantile(values, 0.975)), 6),
            ],
        }
    return {
        "unit": "source video",
        "clusters": len(videos),
        "samples": samples,
        "seed": seed,
        "method": "Percentile cluster bootstrap; sample 21 videos with replacement and retain all examples from each sampled video.",
        "results": output,
    }


def binned_summary(frame: pd.DataFrame, column: str, labels: list[str]) -> list[dict]:
    ranked = frame[column].rank(method="first")
    bins = pd.qcut(ranked, q=len(labels), labels=labels)
    output = []
    for label in labels:
        subset = frame[bins == label]
        output.append(
            {
                "bin": label,
                "count": len(subset),
                "mean_feature_value": round(float(subset[column].mean()), 6),
                "recall_at_1": round(metric(subset), 6),
                "error_rate": round(1 - metric(subset), 6),
            }
        )
    return output


def categorical_bins(
    frame: pd.DataFrame,
    column: str,
    specifications: list[tuple[str, pd.Series]],
) -> list[dict]:
    output = []
    for label, mask in specifications:
        subset = frame[mask]
        output.append(
            {
                "bin": label,
                "count": len(subset),
                "mean_feature_value": (
                    round(float(subset[column].mean()), 6) if len(subset) else None
                ),
                "recall_at_1": round(metric(subset), 6) if len(subset) else None,
                "error_rate": round(1 - metric(subset), 6) if len(subset) else None,
            }
        )
    return output


def ambiguity_analysis(frame: pd.DataFrame) -> dict:
    return {
        "candidate_count_bins": categorical_bins(
            frame,
            "candidate_count",
            [
                ("1-5", frame["candidate_count"] <= 5),
                ("6-10", frame["candidate_count"].between(6, 10)),
                ("11-15", frame["candidate_count"].between(11, 15)),
                ("16+", frame["candidate_count"] >= 16),
            ],
        ),
        "overlap_count_bins": categorical_bins(
            frame,
            "overlap_count",
            [
                ("0", frame["overlap_count"] == 0),
                ("1", frame["overlap_count"] == 1),
                ("2-3", frame["overlap_count"].between(2, 3)),
                ("4+", frame["overlap_count"] >= 4),
            ],
        ),
        "same_type_count_bins": categorical_bins(
            frame,
            "same_type_count",
            [
                ("0-1", frame["same_type_count"] <= 1),
                ("2-3", frame["same_type_count"].between(2, 3)),
                ("4-6", frame["same_type_count"].between(4, 6)),
                ("7+", frame["same_type_count"] >= 7),
            ],
        ),
        "max_non_gold_iou_bins": categorical_bins(
            frame,
            "max_non_gold_iou",
            [
                ("0", frame["max_non_gold_iou"] == 0),
                ("0-.10", (frame["max_non_gold_iou"] > 0) & (frame["max_non_gold_iou"] <= 0.10)),
                (".10-.30", (frame["max_non_gold_iou"] > 0.10) & (frame["max_non_gold_iou"] <= 0.30)),
                (".30+", frame["max_non_gold_iou"] > 0.30),
            ],
        ),
        "gold_area_quartiles": binned_summary(
            frame, "gold_area_fraction", ["Q1 small", "Q2", "Q3", "Q4 large"]
        ),
        "duplicate_latex": [
            {
                "group": "no exact duplicate",
                "count": int((frame["duplicate_latex_count"] == 0).sum()),
                "recall_at_1": round(metric(frame[frame["duplicate_latex_count"] == 0]), 6),
            },
            {
                "group": "one or more exact duplicates",
                "count": int((frame["duplicate_latex_count"] > 0).sum()),
                "recall_at_1": round(metric(frame[frame["duplicate_latex_count"] > 0]), 6),
            },
        ],
        "definitions": {
            "overlap_count": "Number of non-gold candidates with positive intersection area with the first gold box.",
            "same_type_count": "Number of non-gold candidates sharing the first gold candidate's object type.",
            "max_non_gold_iou": "Maximum IoU between the first gold box and any non-gold candidate.",
            "duplicate_latex_count": "Number of non-gold candidates with the same whitespace-insensitive reviewed LaTeX string as the first gold candidate.",
        },
    }


def subgroup(frame: pd.DataFrame, mask: pd.Series) -> dict:
    subset = frame[mask]
    return {
        "count": len(subset),
        "recall_at_1": round(metric(subset), 6) if len(subset) else None,
    }


def interactions(frame: pd.DataFrame) -> dict:
    return {
        "mention_type_by_candidate_density": {
            mention: {
                bucket: subgroup(
                    frame,
                    (frame["mention_type"] == mention)
                    & (
                        (frame["candidate_count"] <= 5)
                        if bucket == "1-5"
                        else (frame["candidate_count"] >= 16)
                    ),
                )
                for bucket in ["1-5", "16+"]
            }
            for mention in ["explicit", "implicit"]
        },
        "mention_type_by_language": {
            language: {
                mention: subgroup(
                    frame,
                    (frame["language"] == language)
                    & (frame["mention_type"] == mention),
                )
                for mention in ["explicit", "implicit"]
            }
            for language in ["en", "jp"]
        },
        "reference_log_by_mention_type": {
            mention: {
                group: subgroup(
                    frame,
                    (frame["mention_type"] == mention)
                    & (frame["logged_target"] == int(group == "logged")),
                )
                for group in ["logged", "other"]
            }
            for mention in ["explicit", "implicit"]
        },
        "object_type_by_candidate_density": {
            object_type: {
                bucket: subgroup(
                    frame,
                    (frame["gold_type"] == object_type)
                    & (
                        (frame["candidate_count"] <= 5)
                        if bucket == "1-5"
                        else (frame["candidate_count"] >= 16)
                    ),
                )
                for bucket in ["1-5", "16+"]
            }
            for object_type in ["equation", "span", "term", "figure", "figure_part"]
        },
    }


def paired_gain_interactions(latex_frame: pd.DataFrame, plain_frame: pd.DataFrame) -> dict:
    frame = latex_frame.merge(
        plain_frame[["example_id", "correct"]].rename(columns={"correct": "plain_correct"}),
        on="example_id",
        validate="one_to_one",
    )

    def gain(mask: pd.Series) -> dict:
        subset = frame[mask]
        return {
            "count": len(subset),
            "plain_recall_at_1": round(metric(subset, "plain_correct"), 6),
            "latex_recall_at_1": round(metric(subset, "correct"), 6),
            "latex_gain": round(
                metric(subset, "correct") - metric(subset, "plain_correct"), 6
            ),
        }

    return {
        "mention_type": {
            label: gain(frame["mention_type"] == label)
            for label in ["explicit", "implicit"]
        },
        "language": {
            label: gain(frame["language"] == label) for label in ["en", "jp"]
        },
        "reference_log": {
            label: gain(frame["logged_target"] == int(label == "logged"))
            for label in ["logged", "other"]
        },
        "candidate_count": {
            "1-5": gain(frame["candidate_count"] <= 5),
            "6-10": gain(frame["candidate_count"].between(6, 10)),
            "11-15": gain(frame["candidate_count"].between(11, 15)),
            "16+": gain(frame["candidate_count"] >= 16),
        },
        "gold_type": {
            label: gain(frame["gold_type"] == label)
            for label in ["equation", "span", "term", "figure", "figure_part"]
        },
    }


def main() -> None:
    args = parse_args()
    plain_rows = read_jsonl(args.predictions_plain)
    latex_rows = read_jsonl(args.predictions_latex)
    if [row["example_id"] for row in plain_rows] != [row["example_id"] for row in latex_rows]:
        raise ValueError("Plain and LaTeX prediction rows are not aligned.")
    targets = reference_targets(args.reference_logs)
    plain_frame = pd.DataFrame(feature_row(row, targets) for row in plain_rows)
    latex_frame = pd.DataFrame(feature_row(row, targets) for row in latex_rows)
    report = {
        "dataset_version": "arr_aug_v1",
        "analysis_status": "post_hoc_exploratory",
        "model": "gpt-5.4-2026-03-05",
        "condition": "vlm-ranking-latex-v4",
        "multivariable_logistic": logistic_analysis(latex_frame),
        "video_clustered_bootstrap": clustered_bootstrap(
            latex_frame, plain_frame, args.bootstrap_samples, args.seed
        ),
        "candidate_ambiguity": ambiguity_analysis(latex_frame),
        "interactions": interactions(latex_frame),
        "latex_gain_interactions": paired_gain_interactions(latex_frame, plain_frame),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
