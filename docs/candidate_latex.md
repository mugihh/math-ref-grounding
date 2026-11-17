# Reviewed candidate LaTeX

The publication package includes the final human-reviewed symbolic metadata
in `data/latex/candidate_latex.jsonl` and the aligned ranking examples in
`data/benchmark/examples_latex.jsonl`. No OCR or review step is required to
use these files. See the [reproduction guide](reproduction.md).

During dataset construction, Chandra OCR produced drafts from candidate
crops. Human reviewers corrected or rejected them; missing recognition was
preserved as `null`, without dropping candidates. Non-symbolic figure and
figure-part candidates have null LaTeX by policy.

Candidate identity uses `(image_path, visual_id)` because visual IDs are only
unique within a frame. This key keeps repeated IDs across frames distinct.

The released condition evaluates reviewed metadata, not a fully automatic OCR
pipeline. Aggregate review statistics are in
[`reports/paper/latex_review_stats.json`](../reports/paper/latex_review_stats.json).
Private OCR/review-workbook preparation tools and workbooks are maintained in
the research repository and excluded from this publication package.

`mathref.latex.crop` remains because the frozen CLIP baseline uses its tight
candidate-crop function.
