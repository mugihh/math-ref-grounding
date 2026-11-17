"""Validation checks for annotated keyframe JSON files."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from mathref.data.parse_annotation import iter_annotation_files
from mathref.utils.io import read_json

KNOWN_CLASSES = {"equation", "span", "term", "figure", "figure_part"}


def validate_annotation_dir(input_path: str | Path) -> dict[str, Any]:
    root = Path(input_path)
    if root.is_file():
        root = root.parent.parent.parent.parent
    json_files = iter_annotation_files(input_path)
    issues: list[dict[str, Any]] = []

    for json_path in json_files:
        issues.extend(validate_annotation_file(json_path, root=root))

    summary = _summarize_issues(issues)
    summary["files"] = len(json_files)
    return {"summary": summary, "issues": issues}


def validate_annotation_file(path: str | Path, root: str | Path | None = None) -> list[dict[str, Any]]:
    json_path = Path(path)
    root_path = Path(root).resolve() if root is not None else json_path.parent.parent.parent
    rel_path = _relative_path(json_path, root_path)
    issues: list[dict[str, Any]] = []

    if not json_path.with_suffix("").exists():
        issues.append(_issue("error", "missing_image", rel_path, "Annotation JSON has no matching image file."))

    data = read_json(json_path)
    caption_text = (data.get("caption") or {}).get("text") or ""
    boxes = data.get("bounding_boxes") or []
    relations = data.get("i2t_relations") or []

    if not boxes:
        issues.append(_issue("warning", "empty_boxes", rel_path, "Frame has no bounding boxes."))
    if not relations:
        issues.append(_issue("warning", "empty_relations", rel_path, "Frame has no mention-box relations."))

    box_ids: list[str] = []
    box_types: dict[str, str] = {}
    for index, box in enumerate(boxes):
        visual_id = str(box.get("id") or "")
        box_ids.append(visual_id)
        box_types[visual_id] = str(box.get("class_name") or "")
        issues.extend(_validate_box(box, index, rel_path))

    for visual_id, count in Counter(box_ids).items():
        if visual_id and count > 1:
            issues.append(
                _issue(
                    "error",
                    "duplicate_visual_id",
                    rel_path,
                    f"Visual id {visual_id!r} appears {count} times in one frame.",
                    visual_id=visual_id,
                )
            )

    box_id_set = set(box_ids)
    mention_links: dict[tuple[str, int, int], list[str]] = defaultdict(list)
    for index, relation in enumerate(relations):
        issues.extend(_validate_relation(relation, index, rel_path, caption_text, box_id_set, box_types))
        region = relation.get("caption_region") or [0, 0]
        if len(region) == 2:
            key = (str(relation.get("caption_id") or ""), int(region[0]), int(region[1]))
            visual_id = str(relation.get("visual_id") or "")
            if visual_id and visual_id not in mention_links[key]:
                mention_links[key].append(visual_id)

    for (caption_id, start, length), visual_ids in mention_links.items():
        if len(visual_ids) > 1:
            issues.append(
                _issue(
                    "info",
                    "multi_gold_mention",
                    rel_path,
                    f"One mention links to {len(visual_ids)} visual ids.",
                    caption_id=caption_id,
                    span=[start, start + length],
                    mention_text=caption_text[start : start + length],
                    visual_ids=visual_ids,
                )
            )

    if len(boxes) >= 20:
        issues.append(
            _issue(
                "info",
                "candidate_count_outlier",
                rel_path,
                f"Frame has {len(boxes)} candidate boxes.",
                count=len(boxes),
            )
        )

    return issues


def _validate_box(box: dict[str, Any], index: int, rel_path: str) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    visual_id = str(box.get("id") or "")
    class_name = str(box.get("class_name") or "")
    bbox = box.get("bb")

    if not visual_id:
        issues.append(_issue("error", "missing_visual_id", rel_path, f"Box {index} has no id.", box_index=index))
    if class_name not in KNOWN_CLASSES:
        issues.append(
            _issue(
                "error",
                "unknown_class",
                rel_path,
                f"Box {visual_id or index!r} has unknown class {class_name!r}.",
                visual_id=visual_id,
                class_name=class_name,
            )
        )
    if not isinstance(bbox, list) or len(bbox) != 4:
        issues.append(
            _issue(
                "error",
                "invalid_bbox",
                rel_path,
                f"Box {visual_id or index!r} bbox must be a 4-item list.",
                visual_id=visual_id,
                bbox=bbox,
            )
        )
        return issues
    try:
        x, y, width, height = [float(value) for value in bbox]
    except (TypeError, ValueError):
        issues.append(
            _issue(
                "error",
                "invalid_bbox",
                rel_path,
                f"Box {visual_id or index!r} bbox contains non-numeric values.",
                visual_id=visual_id,
                bbox=bbox,
            )
        )
        return issues
    if width <= 0 or height <= 0:
        issues.append(
            _issue(
                "error",
                "invalid_bbox_size",
                rel_path,
                f"Box {visual_id or index!r} has non-positive width/height.",
                visual_id=visual_id,
                bbox=[x, y, width, height],
            )
        )
    if x < 0 or y < 0:
        issues.append(
            _issue(
                "warning",
                "negative_bbox_coordinate",
                rel_path,
                f"Box {visual_id or index!r} has a negative x/y coordinate.",
                visual_id=visual_id,
                bbox=[x, y, width, height],
            )
        )
    return issues


def _validate_relation(
    relation: dict[str, Any],
    index: int,
    rel_path: str,
    caption_text: str,
    box_ids: set[str],
    box_types: dict[str, str],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    visual_id = str(relation.get("visual_id") or "")
    relation_name = str(relation.get("relation_name") or "")
    region = relation.get("caption_region")

    if not visual_id:
        issues.append(_issue("error", "missing_relation_visual_id", rel_path, f"Relation {index} has no visual_id."))
    elif visual_id not in box_ids:
        issues.append(
            _issue(
                "error",
                "dangling_visual_id",
                rel_path,
                f"Relation points to {visual_id!r}, but it is not in bounding_boxes.",
                visual_id=visual_id,
            )
        )

    if visual_id in box_types and relation_name and relation_name != box_types[visual_id]:
        issues.append(
            _issue(
                "warning",
                "relation_class_mismatch",
                rel_path,
                f"Relation class {relation_name!r} differs from box class {box_types[visual_id]!r}.",
                visual_id=visual_id,
                relation_name=relation_name,
                box_class=box_types[visual_id],
            )
        )

    if not isinstance(region, list) or len(region) != 2:
        issues.append(
            _issue(
                "error",
                "invalid_caption_region",
                rel_path,
                f"Relation {index} caption_region must be a 2-item list.",
                visual_id=visual_id,
                caption_region=region,
            )
        )
        return issues
    start, length = int(region[0]), int(region[1])
    if start < 0 or length <= 0 or start + length > len(caption_text):
        issues.append(
            _issue(
                "error",
                "caption_region_out_of_range",
                rel_path,
                f"Relation {index} caption_region is outside caption text.",
                visual_id=visual_id,
                caption_region=region,
                caption_length=len(caption_text),
            )
        )
    return issues


def _summarize_issues(issues: list[dict[str, Any]]) -> dict[str, Any]:
    by_severity = Counter(issue["severity"] for issue in issues)
    by_type = Counter(issue["type"] for issue in issues)
    return {
        "issues": len(issues),
        "errors": by_severity.get("error", 0),
        "warnings": by_severity.get("warning", 0),
        "info": by_severity.get("info", 0),
        "by_type": dict(sorted(by_type.items())),
    }


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _issue(severity: str, issue_type: str, file_path: str, message: str, **details: Any) -> dict[str, Any]:
    return {
        "severity": severity,
        "type": issue_type,
        "file": file_path,
        "message": message,
        **details,
    }
