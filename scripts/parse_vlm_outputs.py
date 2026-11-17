"""Convert raw frozen-VLM outputs into complete prediction rows."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mathref.vlm.parse import main


if __name__ == "__main__":
    main()
