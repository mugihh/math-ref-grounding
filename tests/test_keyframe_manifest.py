import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[1]


def test_keyframe_manifest_is_complete_and_valid() -> None:
    schema = json.loads(
        (ROOT / "schemas/keyframe_manifest.schema.json").read_text(encoding="utf-8")
    )
    rows = [
        json.loads(line)
        for line in (ROOT / "data/keyframes/keyframes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    for row in rows:
        validator.validate(row)
        assert row["frame_id"] == row["image_path"].removesuffix(".jpg")

    assert len(rows) == 208
    assert len({row["image_path"] for row in rows}) == 208
    assert len({row["sha256"] for row in rows}) == 208
    assert len(
        {(row["language"], row["topic"], row["video"]) for row in rows}
    ) == 21
    assert {row["image_verification"] for row in rows} == {
        "byte_identical_to_phase1",
        "pixel_equivalent_reencode",
    }
