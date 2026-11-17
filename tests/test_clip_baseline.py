import json
from pathlib import Path

from jsonschema import Draft202012Validator
from PIL import Image

from mathref.baselines.clip_dual_encoder import StubEncoder, predict_examples
from mathref.eval.metrics import evaluate_predictions

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"


def _schema(name: str) -> dict:
    with (SCHEMA_DIR / name).open(encoding="utf-8") as stream:
        return json.load(stream)


def _toy_example(tmp_path: Path, latex=None) -> dict:
    image_path = tmp_path / "000.png"
    Image.new("RGB", (300, 200), "white").save(image_path)
    candidates = [
        {"visual_id": "equation_00", "object_type": "equation", "bbox": [10, 10, 120, 50], "source": "gold"},
        {"visual_id": "term_00", "object_type": "term", "bbox": [20, 120, 18, 20], "source": "gold"},
        {"visual_id": "span_00", "object_type": "span", "bbox": [150, 60, 90, 40], "source": "gold"},
    ]
    if latex:
        for candidate, value in zip(candidates, latex):
            candidate["latex"] = value
    return {
        "example_id": "ex0",
        "frame_id": "en/algebra/video1/000",
        "image_path": str(image_path),
        "image_size": [300, 200],
        "mention": {"mention_id": "m0", "text": "this denominator", "span": [0, 4]},
        "gold_visual_ids": ["term_00"],
        "gold_object_types": ["term"],
        "candidates": candidates,
    }


def test_stub_ranks_all_candidates_and_validates(tmp_path):
    example = _toy_example(tmp_path)
    rows = predict_examples([example], StubEncoder())
    assert len(rows) == 1
    row = rows[0]
    # every candidate ranked exactly once, nothing dropped.
    assert sorted(row["ranked_visual_ids"]) == ["equation_00", "span_00", "term_00"]
    assert set(row["scores"]) == {"equation_00", "span_00", "term_00"}
    assert row["baseline"] == "clip_dual_encoder:stub:plain"
    Draft202012Validator(_schema("prediction.schema.json")).validate(row)
    evaluate_predictions(rows)  # runs without error


def test_stub_is_deterministic(tmp_path):
    example = _toy_example(tmp_path)
    a = predict_examples([example], StubEncoder())[0]["ranked_visual_ids"]
    b = predict_examples([example], StubEncoder())[0]["ranked_visual_ids"]
    assert a == b


def test_latex_fusion_changes_scores_and_marks_baseline(tmp_path):
    example = _toy_example(tmp_path, latex=["x^2", None, "\\frac{a}{b}"])
    plain = predict_examples([example], StubEncoder(), use_latex=False)[0]
    fused = predict_examples([example], StubEncoder(), use_latex=True)[0]
    assert fused["baseline"] == "clip_dual_encoder:stub:latex"
    # candidate with LaTeX gets a fused box vector -> a different score; the
    # null-LaTeX candidate keeps its image-only score.
    assert plain["scores"]["equation_00"] != fused["scores"]["equation_00"]
    assert plain["scores"]["term_00"] == fused["scores"]["term_00"]
    Draft202012Validator(_schema("prediction.schema.json")).validate(fused)
