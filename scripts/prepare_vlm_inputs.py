"""Render candidate labels and build a frozen-VLM input manifest."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mathref.vlm.prepare import main


if __name__ == "__main__":
    main()
