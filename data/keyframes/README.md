# Keyframe reconstruction metadata

`keyframes.jsonl` contains one record for each of the benchmark's 208
keyframes. It publishes source-video identifiers, the original clip-relative
timestamp, a visually verified timestamp in the complete source video, the
original padded download section, and the SHA-256 of the final annotated image.

Use `source_timestamp_verified` with a freshly downloaded complete source
video. The older `source_timestamp_nominal` fields retain the arithmetic
estimate from the Phase 1 section-download metadata for provenance only.
`clip_timestamp` plus `download_section` retains the original reconstruction
recipe.

The images themselves are not stored here. See
[`docs/keyframe_reconstruction.md`](../../docs/keyframe_reconstruction.md).
