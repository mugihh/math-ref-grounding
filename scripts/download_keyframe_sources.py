#!/usr/bin/env python3
"""Download the unique source clips required by a keyframe manifest."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--full-video",
        action="store_true",
        help="Download complete source videos instead of the recorded sections.",
    )
    parser.add_argument(
        "--video-only",
        action="store_true",
        help="Download video-only streams up to 480p (sufficient for timestamp matching).",
    )
    args = parser.parse_args()

    sources = {}
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = (row["language"], row["topic"], row["video"])
        sources[key] = (row["source_url"], row["download_section"])

    for (language, topic, video), (url, section) in sorted(sources.items()):
        output_dir = args.output_root / language / topic / video
        output_dir.mkdir(parents=True, exist_ok=True)
        output_name = "full.%(ext)s" if args.full_video else "clip_001.%(ext)s"
        video_format = (
            "134/18/best[ext=mp4][height<=480]/best[height<=480]"
            if args.video_only
            else (
                "bv*[ext=mp4][vcodec^=avc1]+ba[ext=m4a]/"
                "b[ext=mp4][vcodec^=avc1]/bv*[ext=mp4]+ba[ext=m4a]/"
                "b[ext=mp4]/best"
            )
        )
        command = [
            "yt-dlp",
            url,
            "--no-playlist",
            "-f",
            video_format,
            "--merge-output-format",
            "mp4",
            "-o",
            str(output_dir / output_name),
        ]
        if not args.full_video and section != "all":
            command[2:2] = ["--download-sections", section]
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
