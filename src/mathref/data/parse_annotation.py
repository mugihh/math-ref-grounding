"""Annotation parsing utilities for annotated keyframe JSON files."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from mathref.utils.io import read_json, write_jsonl


@dataclass(frozen=True)
class VisualObject:
    visual_id: str
    object_type: str
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class Mention:
    mention_id: str
    text: str
    gold_visual_ids: tuple[str, ...]


def iter_annotation_files(input_path: str | Path) -> list[Path]:
    """Return annotation JSON files under a file or directory path."""
    path = Path(input_path)
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.jpg.json"))


def parse_annotation_file(path: str | Path, root: str | Path | None = None) -> dict[str, Any]:
    """Parse one annotated keyframe JSON file into a normalized frame record."""
    json_path = Path(path)
    root_path = Path(root).resolve() if root is not None else json_path.parent.parent.parent
    data = read_json(json_path)
    image_file = json_path.with_suffix("")  # real local file; used only to read the size
    image_size = _read_image_size(image_file)
    frame_stem = image_file.stem
    relative_json_path = json_path.resolve().relative_to(root_path)
    # Logical, root-free image id stored in artifacts: "<lang>/<topic>/<video>/<stem>.jpg".
    relative_image_path = relative_json_path.with_suffix("").as_posix()
    path_parts = relative_json_path.parts
    language, topic, video = _infer_dataset_parts(path_parts)
    frame_id = f"{language}/{topic}/{video}/{frame_stem}"
    caption = data.get("caption") or {}
    caption_text = caption.get("text") or ""
    boxes = [_normalize_box(box) for box in data.get("bounding_boxes", [])]
    box_ids = {box["visual_id"] for box in boxes}
    mentions_by_key: dict[tuple[str, int, int], dict[str, Any]] = {}

    for relation in data.get("i2t_relations", []):
        region = relation.get("caption_region") or [0, 0]
        start, length = int(region[0]), int(region[1])
        caption_id = str(relation.get("caption_id") or caption.get("id") or "")
        key = (caption_id, start, length)
        mention = mentions_by_key.setdefault(
            key,
            {
                "mention_id": f"{frame_id}#m{len(mentions_by_key):04d}",
                "caption_id": caption_id,
                "text": caption_text[start : start + length],
                "span": [start, start + length],
                "gold_visual_ids": [],
                "gold_object_types": [],
                "relation_ids": [],
            },
        )
        visual_id = str(relation.get("visual_id"))
        if visual_id not in mention["gold_visual_ids"]:
            mention["gold_visual_ids"].append(visual_id)
        relation_name = relation.get("relation_name")
        if relation_name and relation_name not in mention["gold_object_types"]:
            mention["gold_object_types"].append(relation_name)
        mention["relation_ids"].append(str(relation.get("id") or ""))

    dangling = sorted(
        {
            visual_id
            for mention in mentions_by_key.values()
            for visual_id in mention["gold_visual_ids"]
            if visual_id not in box_ids
        }
    )

    return {
        "frame_id": frame_id,
        "language": language,
        "topic": topic,
        "video": video,
        "image_id": str(data.get("image_id") or frame_stem),
        "image_path": relative_image_path,
        "image_size": image_size,
        "annotation_path": str(json_path),
        "caption": {
            "caption_id": str(caption.get("id") or ""),
            "text": caption_text,
        },
        "boxes": boxes,
        "mentions": list(mentions_by_key.values()),
        "warnings": {"dangling_visual_ids": dangling},
    }


def parse_annotation_dir(input_path: str | Path) -> list[dict[str, Any]]:
    """Parse all annotation files under a file or directory path."""
    root = Path(input_path)
    if root.is_file():
        root = root.parent.parent.parent.parent
    return [parse_annotation_file(path, root=root) for path in iter_annotation_files(input_path)]


def write_frames_jsonl(input_path: str | Path, output_path: str | Path) -> None:
    write_jsonl(output_path, parse_annotation_dir(input_path))


def _infer_dataset_parts(path_parts: tuple[str, ...]) -> tuple[str, str, str]:
    if len(path_parts) >= 4:
        return path_parts[-4], path_parts[-3], path_parts[-2]
    return "unknown", "unknown", "unknown"


def _normalize_box(box: dict[str, Any]) -> dict[str, Any]:
    return {
        "visual_id": str(box["id"]),
        "object_type": str(box.get("class_name") or ""),
        "bbox": [float(value) for value in box.get("bb", [])],
        "source": "gold",
        "auto": bool(box.get("auto_flg", False)),
    }


def _read_image_size(image_path: Path) -> list[int] | None:
    try:
        with Image.open(image_path) as image:
            return [image.width, image.height]
    except (FileNotFoundError, OSError):
        return None
