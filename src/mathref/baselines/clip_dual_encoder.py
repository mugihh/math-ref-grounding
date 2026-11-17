"""Frozen dual-encoder baseline: rank candidates by mention-box similarity.

A zero-training baseline that realizes the dual-encoder diagram in zero-shot form:
encode the mention text and each candidate's visual crop with a frozen encoder,
then rank candidates by cosine similarity. This is the representation-only rung
between the class-prior baseline (no content) and the generative frozen VLM, and
the zero-shot precursor of a future trained scoring model. There is no decoder:
the ranking is simply the argsort of the similarities.

`--use-latex` fuses each candidate's LaTeX text into its box vector, giving a
clean with/without-LaTeX ablation. Encoders are pluggable: StubEncoder runs the
whole chain with no torch/GPU; OpenClipEncoder loads a frozen CLIP model.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from mathref.latex.crop import crop_tight
from mathref.utils.io import read_jsonl, write_jsonl
from mathref.utils.paths import resolve_image_path

Vector = list[float]


class Encoder(Protocol):
    name: str
    model: str

    def load(self) -> None:
        """Load the model once before the first encode (no-op if N/A)."""

    def encode_text(self, texts: list[str]) -> list[Vector]:
        ...

    def encode_image(self, images: list[Image.Image]) -> list[Vector]:
        ...


def _hash_vector(key: str, dim: int = 32) -> Vector:
    """Deterministic pseudo-embedding from a string (stub use only)."""
    return [
        (int(hashlib.sha256(f"{key}:{i}".encode()).hexdigest()[:8], 16) % 2000) / 1000.0 - 1.0
        for i in range(dim)
    ]


class StubEncoder:
    """No-torch deterministic encoder so the whole chain can be tested offline."""

    name = "stub"
    model = "stub"

    def load(self) -> None:  # pragma: no cover - trivial
        return None

    def encode_text(self, texts: list[str]) -> list[Vector]:
        return [_hash_vector(f"t:{text}") for text in texts]

    def encode_image(self, images: list[Image.Image]) -> list[Vector]:
        return [_hash_vector("i:" + str(image.size) + ":" + image.resize((4, 4)).tobytes().hex()) for image in images]


class OpenClipEncoder:
    """Frozen CLIP via open_clip. Weights load lazily; needs torch + open_clip."""

    name = "open_clip"

    def __init__(self, model: str = "ViT-B-32-quickgelu", pretrained: str = "openai", device: str | None = None) -> None:
        self.model = model
        self.pretrained = pretrained
        self.device = device or "cpu"
        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import open_clip
            import torch
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "open_clip/torch not installed. Run `pip install open_clip_torch torch`."
            ) from exc
        self._torch = torch
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.model, pretrained=self.pretrained
        )
        self._tokenizer = open_clip.get_tokenizer(self.model)
        self._model.eval().to(self.device)

    def encode_text(self, texts: list[str]) -> list[Vector]:  # pragma: no cover - needs torch
        tokens = self._tokenizer(texts).to(self.device)
        with self._torch.no_grad():
            features = self._model.encode_text(tokens)
        return features.float().cpu().tolist()

    def encode_image(self, images: list[Image.Image]) -> list[Vector]:  # pragma: no cover - needs torch
        batch = self._torch.stack([self._preprocess(image) for image in images]).to(self.device)
        with self._torch.no_grad():
            features = self._model.encode_image(batch)
        return features.float().cpu().tolist()


ENCODERS = {"stub": StubEncoder, "open_clip": OpenClipEncoder}


def get_encoder(name: str, **kwargs: Any) -> Encoder:
    if name not in ENCODERS:
        raise ValueError(f"Unknown encoder {name!r}; choose from {sorted(ENCODERS)}.")
    return ENCODERS[name](**kwargs)


def _l2norm(vector: Vector) -> Vector:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def _cosine(a: Vector, b: Vector) -> float:
    na, nb = _l2norm(a), _l2norm(b)
    return sum(x * y for x, y in zip(na, nb))


def _fuse(image_vec: Vector, latex_vec: Vector, has_latex: bool) -> Vector:
    """Average the L2-normalized image and LaTeX vectors when LaTeX is present."""
    if not has_latex:
        return image_vec
    ni, nl = _l2norm(image_vec), _l2norm(latex_vec)
    return [(x + y) / 2.0 for x, y in zip(ni, nl)]


def _prediction_row(
    example: dict[str, Any], ranked: list[str], scores: dict[str, float], baseline: str
) -> dict[str, Any]:
    return {
        "example_id": example["example_id"],
        "frame_id": example["frame_id"],
        "mention": example["mention"],
        "gold_visual_ids": example["gold_visual_ids"],
        "gold_object_types": example.get("gold_object_types", []),
        "image_size": example.get("image_size"),
        "candidates": example.get("candidates", []),
        "ranked_visual_ids": ranked,
        "scores": scores,
        "baseline": baseline,
    }


def predict_examples(
    examples: list[dict[str, Any]],
    encoder: Encoder,
    images_root: str | Path | None = None,
    use_latex: bool = False,
) -> list[dict[str, Any]]:
    encoder.load()
    root = Path(images_root) if images_root is not None else None
    image_cache: dict[str, Image.Image] = {}
    baseline = f"clip_dual_encoder:{encoder.model}:{'latex' if use_latex else 'plain'}"
    rows = []

    for example in examples:
        resolved = resolve_image_path(example["image_path"], root)
        if str(resolved) not in image_cache:
            image_cache[str(resolved)] = Image.open(resolved).convert("RGB")
        image = image_cache[str(resolved)]
        candidates = example["candidates"]

        mention_vec = encoder.encode_text([example["mention"]["text"]])[0]
        box_vecs = encoder.encode_image([crop_tight(image, c["bbox"]) for c in candidates])
        if use_latex:
            latex_vecs = encoder.encode_text([c.get("latex") or "" for c in candidates])
            box_vecs = [
                _fuse(box, latex, bool(c.get("latex")))
                for box, latex, c in zip(box_vecs, latex_vecs, candidates)
            ]

        scores = [_cosine(mention_vec, box) for box in box_vecs]
        order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
        ranked = [candidates[i]["visual_id"] for i in order]
        score_map = {candidates[i]["visual_id"]: round(scores[i], 6) for i in range(len(candidates))}
        rows.append(_prediction_row(example, ranked, score_map, baseline))
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen dual-encoder (CLIP) ranking baseline.")
    parser.add_argument("--examples", required=True, help="Ranking examples JSONL.")
    parser.add_argument("--output", required=True, help="Prediction output JSONL.")
    parser.add_argument("--encoder", choices=sorted(ENCODERS), default="stub")
    parser.add_argument("--model", default=None, help="CLIP architecture (default ViT-B-32-quickgelu, which matches the openai weights).")
    parser.add_argument("--pretrained", default=None, help="CLIP weights tag, e.g. openai or laion2b_s34b_b79k.")
    parser.add_argument("--images-root", default=None, help="Root prepended to relative image paths.")
    parser.add_argument("--use-latex", action="store_true", help="Fuse candidate LaTeX into the box vector.")
    parser.add_argument("--device", default=None, help="Device for open_clip, e.g. cuda:0.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    kwargs: dict[str, Any] = {}
    if args.encoder == "open_clip":
        if args.model:
            kwargs["model"] = args.model
        if args.pretrained:
            kwargs["pretrained"] = args.pretrained
        kwargs["device"] = args.device
    encoder = get_encoder(args.encoder, **kwargs)
    rows = predict_examples(read_jsonl(args.examples), encoder, images_root=args.images_root, use_latex=args.use_latex)
    write_jsonl(args.output, rows)
    print(f"CLIP dual-encoder ({encoder.name}:{encoder.model}, latex={args.use_latex}): wrote {len(rows)} -> {args.output}")


if __name__ == "__main__":
    main()
