from pathlib import Path

from scripts.validate_release import validate_release


def test_public_release_artifacts_validate() -> None:
    root = Path(__file__).resolve().parents[1]
    assert validate_release(root) == {
        "examples": 1842,
        "frames": 206,
        "manifest_frames": 208,
    }

