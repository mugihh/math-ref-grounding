"""CLI wrapper: see mathref.baselines.clip_dual_encoder.

Default encoder is `stub` (no torch/GPU). Use `--encoder open_clip --model ViT-B-32
--pretrained openai` once open_clip/torch are installed; add `--use-latex` for the
LaTeX-fusion ablation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mathref.baselines.clip_dual_encoder import main


if __name__ == "__main__":
    main()
