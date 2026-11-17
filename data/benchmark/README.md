# Benchmark release

This directory contains the public, image-free release of the benchmark used
in *Grounding Referring Expressions to Handwritten Mathematics in
Instructional Conversation*.

## Files

- `examples.jsonl`: 1,842 mention-to-box ranking examples with transcript
  context, mention spans, candidate boxes, object classes, and gold links.
- `examples_latex.jsonl`: the same examples with reviewed candidate-level
  LaTeX metadata. The row order and `example_id` values are identical.

Both files conform to `schemas/ranking_example.schema.json`. Bounding boxes use
`[x, y, width, height]` in source-image pixels. Candidate `visual_id` values
are frame-local and should not be interpreted as persistent object identities
across keyframes.

The benchmark is evaluated as one complete collection for the frozen-model
experiments in the paper. It has no official train/dev/test split. Any future
trained-model study should split or cross-validate by
`(language, topic, video)` so that a video never crosses folds.

## Images

Source videos and extracted keyframes are not redistributed. The relative
`image_path` in each example can be reconstructed using
`data/keyframes/keyframes.jsonl` and `docs/keyframe_reconstruction.md`. Users
are responsible for complying with the terms of the original video providers.

## Integrity checks

```bash
python3 scripts/validate_release.py
```

The release validator checks both JSONL files against the schema, verifies
their one-to-one example alignment, confirms the expected benchmark counts,
and checks that every image path is represented in the keyframe manifest.

