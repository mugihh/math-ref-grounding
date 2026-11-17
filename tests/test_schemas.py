import json
from pathlib import Path

from jsonschema import Draft202012Validator

from mathref.baselines.random import predict_examples
from mathref.data.build_examples import build_examples
from mathref.eval.metrics import evaluate_predictions


SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"


def _load_schema(name: str) -> dict:
    with (SCHEMA_DIR / name).open(encoding="utf-8") as stream:
        return json.load(stream)


def _frame() -> dict:
    return {
        "frame_id": "jp/calculus/video1/000",
        "language": "jp",
        "topic": "calculus",
        "video": "video1",
        "image_id": "1",
        "image_path": "/data/000.jpg",
        "image_size": [1280, 720],
        "annotation_path": "/data/000.jpg.json",
        "caption": {"caption_id": "1", "text": "this term"},
        "boxes": [
            {
                "visual_id": "term_00",
                "object_type": "term",
                "bbox": [10.0, 20.0, 30.0, 40.0],
                "source": "gold",
                "auto": False,
            }
        ],
        "mentions": [
            {
                "mention_id": "jp/calculus/video1/000#m0000",
                "caption_id": "1",
                "text": "this",
                "span": [0, 4],
                "gold_visual_ids": ["term_00"],
                "gold_object_types": ["term"],
                "relation_ids": ["1"],
            }
        ],
        "warnings": {"dangling_visual_ids": []},
    }


def test_all_json_schemas_are_valid_draft_2020_12() -> None:
    for path in sorted(SCHEMA_DIR.glob("*.schema.json")):
        Draft202012Validator.check_schema(_load_schema(path.name))


def test_pipeline_rows_validate_against_contracts() -> None:
    frame = _frame()
    example = build_examples([frame])[0]
    prediction = predict_examples([example], seed=13)[0]
    evaluation = evaluate_predictions([prediction])

    Draft202012Validator(
        _load_schema("normalized_frame.schema.json")
    ).validate(frame)
    Draft202012Validator(
        _load_schema("ranking_example.schema.json")
    ).validate(example)
    Draft202012Validator(
        _load_schema("prediction.schema.json")
    ).validate(prediction)
    Draft202012Validator(
        _load_schema("evaluation.schema.json")
    ).validate(evaluation)
