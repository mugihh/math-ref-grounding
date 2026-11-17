# Annotation Format

Expected source annotations should provide:

- keyframe or image identifier
- transcript text with marked mention spans
- visual objects with IDs, labels, and bounding boxes
- mention-to-visual-object links

The dataset builder will convert those annotations into ranking examples.

Bounding boxes use `[x, y, width, height]` in image pixel coordinates. Parsed
frames and ranking examples include `image_size: [width, height]` when the
keyframe image is available.

The fixed normalized-frame and ranking-example contracts are documented in
[`data_contracts.md`](data_contracts.md), with machine-readable schemas under
[`schemas/`](../schemas/).
