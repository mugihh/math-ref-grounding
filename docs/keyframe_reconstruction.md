# Keyframe reconstruction

The benchmark uses 208 keyframes from 21 publicly accessible instructional
videos. Source videos and extracted images are not committed to this
repository. The release instead provides a reconstruction record for every
logical `image_path` in [`data/keyframes/keyframes.jsonl`](../data/keyframes/keyframes.jsonl).

## Time fields

- `download_section`: the padded section originally requested from the source
  video. `all` means the complete source.
- `clip_timestamp`: the authoritative timestamp passed to FFmpeg after the
  section was downloaded.
- `source_timestamp_nominal`: `clip_timestamp` plus the requested section
  start. This is useful for navigating the original video but is labeled
  nominal because section downloads can begin on a nearby codec keyframe.
- `source_timestamp_verified`: timestamp in a freshly downloaded complete
  source video, verified against the released keyframe at 10 frames per second.
- `source_timestamp_similarity`: normalized grayscale visual-match score used
  during verification. Scores range from -1 to 1; all released records scored
  at least 0.914.
- `sha256`: checksum of the final keyframe used for annotation and experiments.
  Different downloader/FFmpeg versions may reconstruct a visually equivalent
  frame with different JPEG bytes.

All 208 final keyframes were checked against the Phase 1 extraction set: 207
are byte-identical and one is a pixel-equivalent JPEG re-encoding. They were
also matched against freshly downloaded complete source videos. This corrected
one stale Phase 1 timestamp for `en/probability/video4/007`; the image itself
was unchanged.

## Reconstruct

Install `yt-dlp` and FFmpeg, then download the complete source videos:

```bash
python scripts/download_keyframe_sources.py \
  --manifest data/keyframes/keyframes.jsonl \
  --output-root local/keyframe_sources \
  --full-video \
  --video-only
```

Extract the frames:

```bash
python scripts/extract_keyframes.py \
  --manifest data/keyframes/keyframes.jsonl \
  --sources-root local/keyframe_sources \
  --output-root local/keyframes \
  --full-source
```

Both output roots should remain outside version control. Source availability
and the right to download or use a video remain governed by the source
platform and copyright holder; users are responsible for compliance.
