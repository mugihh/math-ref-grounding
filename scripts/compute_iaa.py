"""Compute mention, box, and link agreement between two annotators."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mathref.eval.iaa import main


if __name__ == "__main__":
    main()
