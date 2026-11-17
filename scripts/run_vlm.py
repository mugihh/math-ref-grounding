"""CLI wrapper: see mathref.vlm.run.

Default backend is `mock` (no API/GPU). Use `--backend openai --model <snapshot>`
with OPENAI_API_KEY set, or `--backend qwen --device cuda:0` on a GPU host.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mathref.vlm.run import main


if __name__ == "__main__":
    main()
