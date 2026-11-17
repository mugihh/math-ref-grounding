"""Inter-annotator agreement for paired raw annotation directories."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from mathref.utils.io import read_json


TAG_RE = re.compile(r"<<<(.*?)>>>", re.DOTALL)
REPEAT_SUFFIX_RE = re.compile(r"_(\d{2})$")


def _iter_annotation_files(path: Path) -> list[Path]:
    return [path] if path.is_file() else sorted(path.rglob("*.jpg.json"))


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _prf(matches: int, count_a: int, count_b: int) -> dict[str, float | int]:
    precision = _safe_div(matches, count_b)
    recall = _safe_div(matches, count_a)
    return {
        "count_a": count_a,
        "count_b": count_b,
        "matches": matches,
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2 * precision * recall, precision + recall),
    }


def _plain_caption_and_markers(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Remove annotation markup and map tagged mentions to plain-text spans."""
    pieces: list[str] = []
    markers: list[dict[str, Any]] = []
    source_cursor = 0
    plain_cursor = 0
    for match in TAG_RE.finditer(text):
        prefix = text[source_cursor : match.start()]
        pieces.append(prefix)
        plain_cursor += len(prefix)

        tagged = match.group(1)
        surface = REPEAT_SUFFIX_RE.sub("", tagged)
        start = plain_cursor
        pieces.append(surface)
        plain_cursor += len(surface)
        markers.append(
            {
                "source_span": [match.start(1), match.end(1)],
                "plain_span": [start, plain_cursor],
                "text": surface,
            }
        )
        source_cursor = match.end()

    suffix = text[source_cursor:]
    pieces.append(suffix)
    return "".join(pieces), markers


def _parse_frame(path: Path) -> dict[str, Any]:
    data = read_json(path)
    caption = data.get("caption") or {}
    caption_text = str(caption.get("text") or "")
    plain_text, markers = _plain_caption_and_markers(caption_text)

    mentions: dict[tuple[int, int], dict[str, Any]] = {}
    for relation in data.get("i2t_relations", []):
        region = relation.get("caption_region") or [0, 0]
        source_start = int(region[0])
        source_end = source_start + int(region[1])
        marker = next(
            (
                item
                for item in markers
                if source_start < item["source_span"][1]
                and source_end > item["source_span"][0]
            ),
            None,
        )
        if marker is None:
            raise ValueError(
                f"Relation span [{source_start}, {source_end}) in {path} is not inside a <<<...>>> tag"
            )
        key = tuple(marker["plain_span"])
        mention = mentions.setdefault(
            key,
            {
                "span": list(key),
                "text": marker["text"],
                "visual_ids": [],
            },
        )
        visual_id = str(relation.get("visual_id") or "")
        if visual_id and visual_id not in mention["visual_ids"]:
            mention["visual_ids"].append(visual_id)

    boxes = [
        {
            "visual_id": str(box.get("id") or ""),
            "object_type": str(box.get("class_name") or ""),
            "bbox": [float(value) for value in box.get("bb", [])],
        }
        for box in data.get("bounding_boxes", [])
    ]
    return {
        "caption": plain_text,
        "mentions": list(mentions.values()),
        "boxes": boxes,
    }


def _span_overlap_shorter(span_a: list[int], span_b: list[int]) -> float:
    overlap = max(0, min(span_a[1], span_b[1]) - max(span_a[0], span_b[0]))
    shorter = min(span_a[1] - span_a[0], span_b[1] - span_b[0])
    return _safe_div(overlap, shorter)


def _greedy_pairs(
    items_a: list[dict[str, Any]],
    items_b: list[dict[str, Any]],
    score_fn: Any,
    threshold: float,
) -> list[tuple[int, int, float]]:
    candidates = [
        (score_fn(item_a, item_b), index_a, index_b)
        for index_a, item_a in enumerate(items_a)
        for index_b, item_b in enumerate(items_b)
    ]
    used_a: set[int] = set()
    used_b: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for score, index_a, index_b in sorted(candidates, reverse=True):
        if score < threshold:
            break
        if index_a in used_a or index_b in used_b:
            continue
        used_a.add(index_a)
        used_b.add(index_b)
        pairs.append((index_a, index_b, score))
    return pairs


def _iou(box_a: dict[str, Any], box_b: dict[str, Any]) -> float:
    ax, ay, aw, ah = box_a["bbox"]
    bx, by, bw, bh = box_b["bbox"]
    intersection_w = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    intersection_h = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    intersection = intersection_w * intersection_h
    union = aw * ah + bw * bh - intersection
    return _safe_div(intersection, union)


def _cohen_kappa(labels_a: list[bool], labels_b: list[bool]) -> float | None:
    if not labels_a:
        return None
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / len(labels_a)
    positive_a = sum(labels_a) / len(labels_a)
    positive_b = sum(labels_b) / len(labels_b)
    expected = positive_a * positive_b + (1 - positive_a) * (1 - positive_b)
    if expected == 1:
        return 1.0 if observed == 1 else None
    return (observed - expected) / (1 - expected)


def _multiclass_kappa(labels_a: list[str], labels_b: list[str]) -> float | None:
    if not labels_a:
        return None
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / len(labels_a)
    classes = set(labels_a) | set(labels_b)
    expected = sum(
        (labels_a.count(label) / len(labels_a)) * (labels_b.count(label) / len(labels_b))
        for label in classes
    )
    if expected == 1:
        return 1.0 if observed == 1 else None
    return (observed - expected) / (1 - expected)


def compare_frames(
    frame_a: dict[str, Any],
    frame_b: dict[str, Any],
    mention_overlap_threshold: float = 0.5,
    box_iou_threshold: float = 0.5,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if frame_a["caption"] != frame_b["caption"]:
        raise ValueError("Annotators do not share the same canonical plain transcript")

    mentions_a = frame_a["mentions"]
    mentions_b = frame_b["mentions"]
    boxes_a = frame_a["boxes"]
    boxes_b = frame_b["boxes"]

    exact_pairs = _greedy_pairs(
        mentions_a,
        mentions_b,
        lambda a, b: 1.0 if a["span"] == b["span"] else 0.0,
        1.0,
    )
    relaxed_pairs = _greedy_pairs(
        mentions_a,
        mentions_b,
        lambda a, b: _span_overlap_shorter(a["span"], b["span"]),
        mention_overlap_threshold,
    )
    box_pairs = _greedy_pairs(boxes_a, boxes_b, _iou, box_iou_threshold)
    box_a_to_b = {
        boxes_a[index_a]["visual_id"]: boxes_b[index_b]["visual_id"]
        for index_a, index_b, _ in box_pairs
    }

    class_matches = sum(
        boxes_a[index_a]["object_type"] == boxes_b[index_b]["object_type"]
        for index_a, index_b, _ in box_pairs
    )
    class_labels_a = [boxes_a[index_a]["object_type"] for index_a, _, _ in box_pairs]
    class_labels_b = [boxes_b[index_b]["object_type"] for _, index_b, _ in box_pairs]
    class_confusion: dict[str, int] = {}
    for label_a, label_b in zip(class_labels_a, class_labels_b):
        key = f"{label_a}|||{label_b}"
        class_confusion[key] = class_confusion.get(key, 0) + 1
    exact_link_matches = 0
    link_tp = link_fp = link_fn = 0
    binary_a: list[bool] = []
    binary_b: list[bool] = []
    link_disagreements: list[dict[str, Any]] = []

    for mention_index_a, mention_index_b, overlap in relaxed_pairs:
        mention_a = mentions_a[mention_index_a]
        mention_b = mentions_b[mention_index_b]
        mapped_a = {
            box_a_to_b[visual_id]
            for visual_id in mention_a["visual_ids"]
            if visual_id in box_a_to_b
        }
        links_b = set(mention_b["visual_ids"])
        all_a_mapped = len(mapped_a) == len(mention_a["visual_ids"])
        exact = all_a_mapped and mapped_a == links_b
        exact_link_matches += int(exact)
        link_tp += len(mapped_a & links_b)
        link_fp += len(links_b - mapped_a)
        link_fn += len(mapped_a - links_b) + (len(mention_a["visual_ids"]) - len(mapped_a))

        for box_index_a, box_index_b, _ in box_pairs:
            binary_a.append(boxes_a[box_index_a]["visual_id"] in mention_a["visual_ids"])
            binary_b.append(boxes_b[box_index_b]["visual_id"] in mention_b["visual_ids"])
        if not exact:
            link_disagreements.append(
                {
                    "mention_a": mention_a,
                    "mention_b": mention_b,
                    "overlap": overlap,
                    "mapped_links_a": sorted(mapped_a),
                    "links_b": sorted(links_b),
                }
            )

    link_precision = _safe_div(link_tp, link_tp + link_fp)
    link_recall = _safe_div(link_tp, link_tp + link_fn)
    metrics = {
        "mention_exact": _prf(len(exact_pairs), len(mentions_a), len(mentions_b)),
        "mention_relaxed": _prf(len(relaxed_pairs), len(mentions_a), len(mentions_b)),
        "box": _prf(len(box_pairs), len(boxes_a), len(boxes_b)),
        "box_class_agreement": _safe_div(class_matches, len(box_pairs)),
        "box_class_kappa": _multiclass_kappa(class_labels_a, class_labels_b),
        "box_class_matches": class_matches,
        "box_class_confusion": class_confusion,
        "link": {
            "matched_mentions": len(relaxed_pairs),
            "exact_matches": exact_link_matches,
            "exact_agreement": _safe_div(exact_link_matches, len(relaxed_pairs)),
            "set_tp": link_tp,
            "set_fp": link_fp,
            "set_fn": link_fn,
            "set_precision": link_precision,
            "set_recall": link_recall,
            "set_f1": _safe_div(2 * link_precision * link_recall, link_precision + link_recall),
            "binary_decisions": len(binary_a),
            "binary_both_positive": sum(a and b for a, b in zip(binary_a, binary_b)),
            "binary_a_positive_b_negative": sum(a and not b for a, b in zip(binary_a, binary_b)),
            "binary_a_negative_b_positive": sum(not a and b for a, b in zip(binary_a, binary_b)),
            "binary_both_negative": sum(not a and not b for a, b in zip(binary_a, binary_b)),
            "cohen_kappa": _cohen_kappa(binary_a, binary_b),
        },
    }
    disagreements = {
        "unmatched_mentions_a": [
            mention for index, mention in enumerate(mentions_a)
            if index not in {pair[0] for pair in relaxed_pairs}
        ],
        "unmatched_mentions_b": [
            mention for index, mention in enumerate(mentions_b)
            if index not in {pair[1] for pair in relaxed_pairs}
        ],
        "unmatched_boxes_a": [
            box for index, box in enumerate(boxes_a)
            if index not in {pair[0] for pair in box_pairs}
        ],
        "unmatched_boxes_b": [
            box for index, box in enumerate(boxes_b)
            if index not in {pair[1] for pair in box_pairs}
        ],
        "class_disagreements": [
            {"box_a": boxes_a[index_a], "box_b": boxes_b[index_b], "iou": score}
            for index_a, index_b, score in box_pairs
            if boxes_a[index_a]["object_type"] != boxes_b[index_b]["object_type"]
        ],
        "link_disagreements": link_disagreements,
    }
    return metrics, disagreements


def _pooled_summary(frame_results: list[dict[str, Any]]) -> dict[str, Any]:
    def pooled_prf(key: str) -> dict[str, float | int]:
        return _prf(
            sum(frame[key]["matches"] for frame in frame_results),
            sum(frame[key]["count_a"] for frame in frame_results),
            sum(frame[key]["count_b"] for frame in frame_results),
        )

    matched_boxes = sum(frame["box"]["matches"] for frame in frame_results)
    class_matches = sum(frame["box_class_matches"] for frame in frame_results)
    class_labels_a: list[str] = []
    class_labels_b: list[str] = []
    for frame in frame_results:
        for key, count in frame["box_class_confusion"].items():
            label_a, label_b = key.split("|||", 1)
            class_labels_a.extend([label_a] * count)
            class_labels_b.extend([label_b] * count)
    matched_mentions = sum(frame["link"]["matched_mentions"] for frame in frame_results)
    exact_links = sum(frame["link"]["exact_matches"] for frame in frame_results)
    link_tp = sum(frame["link"]["set_tp"] for frame in frame_results)
    link_fp = sum(frame["link"]["set_fp"] for frame in frame_results)
    link_fn = sum(frame["link"]["set_fn"] for frame in frame_results)
    link_precision = _safe_div(link_tp, link_tp + link_fp)
    link_recall = _safe_div(link_tp, link_tp + link_fn)

    both_positive = sum(frame["link"]["binary_both_positive"] for frame in frame_results)
    a_positive = both_positive + sum(
        frame["link"]["binary_a_positive_b_negative"] for frame in frame_results
    )
    b_positive = both_positive + sum(
        frame["link"]["binary_a_negative_b_positive"] for frame in frame_results
    )
    both_negative = sum(frame["link"]["binary_both_negative"] for frame in frame_results)
    total_binary = sum(frame["link"]["binary_decisions"] for frame in frame_results)
    labels_a = [True] * a_positive + [False] * (total_binary - a_positive)
    labels_b = (
        [True] * both_positive
        + [False] * (a_positive - both_positive)
        + [True] * (b_positive - both_positive)
        + [False] * both_negative
    )

    return {
        "mention_exact": pooled_prf("mention_exact"),
        "mention_relaxed": pooled_prf("mention_relaxed"),
        "box": pooled_prf("box"),
        "box_class_agreement": _safe_div(class_matches, matched_boxes),
        "box_class_kappa": _multiclass_kappa(class_labels_a, class_labels_b),
        "link": {
            "matched_mentions": matched_mentions,
            "exact_agreement": _safe_div(exact_links, matched_mentions),
            "set_precision": link_precision,
            "set_recall": link_recall,
            "set_f1": _safe_div(2 * link_precision * link_recall, link_precision + link_recall),
            "binary_decisions": total_binary,
            "cohen_kappa": _cohen_kappa(labels_a, labels_b),
        },
    }


def compute_iaa(
    directory_a: str | Path,
    directory_b: str | Path,
    mention_overlap_threshold: float = 0.5,
    box_iou_threshold: float = 0.5,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root_a = Path(directory_a)
    root_b = Path(directory_b)
    files_a = {path.relative_to(root_a): path for path in _iter_annotation_files(root_a)}
    files_b = {path.relative_to(root_b): path for path in _iter_annotation_files(root_b)}
    if files_a.keys() != files_b.keys():
        missing_a = sorted(str(path) for path in files_b.keys() - files_a.keys())
        missing_b = sorted(str(path) for path in files_a.keys() - files_b.keys())
        raise ValueError(f"Frame sets differ; missing from A={missing_a}, missing from B={missing_b}")

    frame_results: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    for relative_path in sorted(files_a):
        metrics, frame_disagreements = compare_frames(
            _parse_frame(files_a[relative_path]),
            _parse_frame(files_b[relative_path]),
            mention_overlap_threshold,
            box_iou_threshold,
        )
        frame_results.append({"frame": str(relative_path), **metrics})
        if any(frame_disagreements.values()):
            disagreements.append({"frame": str(relative_path), **frame_disagreements})

    return {
        "config": {
            "mention_overlap_threshold": mention_overlap_threshold,
            "box_iou_threshold": box_iou_threshold,
        },
        "frame_count": len(frame_results),
        "summary": _pooled_summary(frame_results),
        "frames": frame_results,
    }, disagreements


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator-a", required=True)
    parser.add_argument("--annotator-b", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--disagreements", required=True)
    parser.add_argument("--mention-overlap-threshold", type=float, default=0.5)
    parser.add_argument("--box-iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    metrics, disagreements = compute_iaa(
        args.annotator_a,
        args.annotator_b,
        args.mention_overlap_threshold,
        args.box_iou_threshold,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    disagreement_path = Path(args.disagreements)
    disagreement_path.parent.mkdir(parents=True, exist_ok=True)
    with disagreement_path.open("w", encoding="utf-8") as handle:
        for row in disagreements:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
