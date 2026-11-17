"""Export gold candidate crops from ranking examples.

The same visual box appears in many ranking examples, so candidates are
deduplicated on the composite key ``(image_path, visual_id)`` before cropping.
For each candidate we save a tight crop and a context crop (the box expanded by
a fraction of its larger side, so isolated handwritten symbols keep surrounding
context). Small crops are upscaled with LANCZOS so a reviewer can read them.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from PIL import Image

from mathref.utils.io import read_jsonl, write_jsonl
from mathref.utils.paths import resolve_image_path


def candidate_key(image_path: str, visual_id: str) -> str:
    """Stable composite key; ``visual_id`` is only unique within one image."""
    return f"{image_path}::{visual_id}"


def iter_unique_candidates(examples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Yield one record per unique (image_path, visual_id) across all examples."""
    seen: dict[str, dict[str, Any]] = {}
    for example in examples:
        image_path = example["image_path"]
        for candidate in example["candidates"]:
            key = candidate_key(image_path, candidate["visual_id"])
            if key in seen:
                continue
            seen[key] = {
                "key": key,
                "image_path": image_path,
                "visual_id": candidate["visual_id"],
                "object_type": candidate["object_type"],
                "bbox": candidate["bbox"],
            }
    return list(seen.values())


def _clamp_box(x: float, y: float, w: float, h: float, size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    x0 = max(0, int(round(x)))
    y0 = max(0, int(round(y)))
    x1 = min(width, int(round(x + w)))
    y1 = min(height, int(round(y + h)))
    return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)


def crop_tight(image: Image.Image, bbox: list[float]) -> Image.Image:
    x, y, w, h = bbox
    box = _clamp_box(x, y, w, h, image.size)
    return image.crop(box)


def crop_context(image: Image.Image, bbox: list[float], pad_frac: float) -> Image.Image:
    x, y, w, h = bbox
    pad = pad_frac * max(w, h)
    box = _clamp_box(x - pad, y - pad, w + 2 * pad, h + 2 * pad, image.size)
    return image.crop(box)


def _maybe_upscale(crop: Image.Image, min_side: int, max_scale: float) -> Image.Image:
    short = min(crop.size)
    if short >= min_side:
        return crop
    scale = min(max_scale, min_side / short)
    return crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.LANCZOS)




def export_crops(
    examples_path: str | Path,
    out_dir: str | Path,
    images_root: str | Path | None = None,
    context_pad_frac: float = 0.6,
    min_side: int = 96,
    max_scale: float = 4.0,
) -> list[dict[str, Any]]:
    """Crop every unique candidate; return the crop manifest records."""
    out_dir = Path(out_dir)
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    root = Path(images_root) if images_root is not None else None

    candidates = iter_unique_candidates(read_jsonl(examples_path))
    image_cache: dict[str, Image.Image] = {}
    manifest: list[dict[str, Any]] = []

    for cand in candidates:
        resolved = resolve_image_path(cand["image_path"], root)
        if str(resolved) not in image_cache:
            image_cache[str(resolved)] = Image.open(resolved).convert("RGB")
        image = image_cache[str(resolved)]

        safe_id = cand["visual_id"].replace("/", "_")
        key_digest = hashlib.sha256(cand["key"].encode("utf-8")).hexdigest()[:12]
        stem = f"{Path(cand['image_path']).stem}__{safe_id}__{key_digest}"

        tight = _maybe_upscale(crop_tight(image, cand["bbox"]), min_side, max_scale)
        context = _maybe_upscale(crop_context(image, cand["bbox"], context_pad_frac), min_side, max_scale)
        tight_name = f"{stem}__tight.png"
        context_name = f"{stem}__context.png"
        tight.save(crops_dir / tight_name)
        context.save(crops_dir / context_name)

        manifest.append(
            {
                "key": cand["key"],
                "image_path": cand["image_path"],
                "visual_id": cand["visual_id"],
                "object_type": cand["object_type"],
                "bbox": cand["bbox"],
                "crop_tight": f"crops/{tight_name}",
                "crop_context": f"crops/{context_name}",
            }
        )

    write_jsonl(out_dir / "crop_manifest.jsonl", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export gold candidate crops for LaTeX review.")
    parser.add_argument("--examples", required=True, help="Ranking examples JSONL.")
    parser.add_argument("--out", required=True, help="Output directory for crops and manifest.")
    parser.add_argument("--images-root", default=None, help="Root prepended to relative image paths.")
    parser.add_argument("--context-pad-frac", type=float, default=0.6, help="Context padding as fraction of the larger box side.")
    parser.add_argument("--min-side", type=int, default=96, help="Upscale crops whose shorter side is below this.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    manifest = export_crops(
        args.examples,
        args.out,
        images_root=args.images_root,
        context_pad_frac=args.context_pad_frac,
        min_side=args.min_side,
    )
    print(f"Exported {len(manifest)} unique candidate crops to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
