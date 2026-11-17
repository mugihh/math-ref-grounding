#!/usr/bin/env python3
"""Extract benchmark keyframes from locally available source clips."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sources-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--full-source",
        action="store_true",
        help="Read full.mp4 and use the visually verified full-source timestamp.",
    )
    args = parser.parse_args()

    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        source_name = "full.mp4" if args.full_source else "clip_001.mp4"
        source = (
            args.sources_root
            / row["language"]
            / row["topic"]
            / row["video"]
            / source_name
        )
        output = args.output_root / row["image_path"]
        if output.exists() and not args.overwrite:
            continue
        if not source.is_file():
            raise FileNotFoundError(source)
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                (
                    row["source_timestamp_verified"]
                    if args.full_source
                    else row["clip_timestamp"]
                ),
                "-i",
                str(source),
                "-frames:v",
                "1",
                "-update",
                "1",
                "-q:v",
                "2",
                str(output),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
