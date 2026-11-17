"""CLI wrapper for mathref.vlm.audit."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mathref.vlm.audit import main


if __name__ == "__main__":
    main()
