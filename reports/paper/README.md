# Paper result snapshot

This directory contains compact, version-controlled statistics cited by the
paper. Large or sensitive intermediate artifacts remain outside
Git, including source media, raw OCR output, review workbooks, candidate crops,
rendered previews, model responses, and prediction files.

## Dataset statistics

`dataset_stats.json` summarizes the frozen 1,842-example benchmark. The core
statistics were generated with:

```bash
python scripts/dataset_stats.py \
  --input data/processed/examples.jsonl \
  --input-format examples-jsonl \
  --annotations data/raw/annotations \
  --output outputs/paper_stats/dataset_stats_final.json
```

Unique candidate-object counts were reconciled against
`data/latex/candidate_latex.jsonl`.

## Candidate-LaTeX review statistics

`latex_review_stats.json` compares the raw Chandra output recorded in the four
final human-review workbooks with the released
`data/latex/candidate_latex.jsonl`. A correction is counted when the trimmed
reviewed string differs exactly from the trimmed raw string; this includes
filling an initially missing recognition. The correction rate is calculated
over symbolic candidates (`equation`, `span`, and `term`) only.

The source workbooks are retained privately for audit because they contain
large embedded review images and are not release artifacts. Their 1,344
non-overlapping rows reconcile with the public 1,348-candidate artifact: the
remaining four candidates are one `figure` and three `figure_part` objects
that were never submitted to OCR. All 57 non-symbolic candidates have null
LaTeX by policy.

Before publication, record the release commit and create a Git tag for the
paper dataset and reports.

## Inter-annotator agreement

`iaa/primary_en_calculus_video3.json` contains the reported 12-frame IAA
results comparing the researcher and the hired annotator. The exploratory
category-enriched subset is stored separately in
`iaa/exploratory_en_calculus_video2_figure_subset.json` and is not pooled into
the primary result. The JSONs contain aggregate and per-frame metrics but no
annotator identities or raw disagreement text.

The analysis used `scripts/compute_iaa.py` after canonical transcript-boundary
normalization with a private delivery-normalization tool. Raw deliveries,
normalization audits, and disagreement-level JSONL remain private.

## Official evaluation snapshot

`evaluations/` contains the compact evaluator JSONs for the official
full-benchmark runs: five seeds each for random and class-prior, deterministic
CLIP and CLIP+LaTeX runs, and GPT-5.4 and Qwen2.5-VL under both VLM conditions.
These files contain metrics, breakdowns, and run metadata, but no raw model
responses or per-example predictions.

`result_summary.json` records the final prompt-v4 headline metrics and
per-language slices. Prompt v4 corrects terminal annotation-suffix removal for
parenthesized mentions; 1,697 byte-for-byte unchanged model requests per
condition reuse their audited prompt-v3 outputs, while the affected 145
requests per condition were rerun.
`vlm_compliance.json` records raw output-compliance counts before deterministic
ranking completion. `openai_cost.json` records aggregate token use and estimated
API cost at the rates captured for the official run. `tables/` is generated
from only the official full-benchmark evaluator JSONs:

```bash
python scripts/generate_tables.py \
  --eval reports/paper/evaluations/*.json \
  --out reports/paper/tables
```

`gpt54_latex_paired_analysis.json` records the paired GPT-5.4 rank transitions
used in the manuscript error analysis, including counts of examples improved,
harmed, and unchanged by reviewed candidate LaTeX and the audited
representative-case identifiers. `docs/error_analysis.md` documents the
reporting decisions and qualitative interpretation.

`source_provenance.json` reconciles the 21 source videos, 208 keyframes,
language/topic counts, transcript provenance, first-to-last selected-keyframe
spans, and the final source-link check. These durations are source-timeline
spans rather than continuously extracted clip durations.

`spatial_error_analysis.json` records the exact-error-only spatial diagnostic
for GPT-5.4 + LaTeX. `docs/spatial_analysis.md` gives the full-model spatial
table, wrong-only slices, directional containment analysis, object-type
confusions, and the reporting recommendation. Spatial measures compare
human-annotated candidates and are auxiliary to exact visual-ID evaluation;
they are not detection metrics.

`reference_log_difficulty_analysis.json` records the post-hoc comparison
between mentions whose normalized text matches a target in a per-video
reference log and all other mentions. It includes overall and
explicit/implicit-stratified R@1, together with exact-error-only spatial
diagnostics. Reference logs are treated as a difficulty proxy, not as a
prospectively designed annotator-confusion label.

`difficulty_multivariable_analysis.json` adds an associational logistic model
with video-clustered standard errors, a 5,000-sample video-clustered
bootstrap, candidate ambiguity and nesting summaries, and selected
interactions. It is reproducible with `scripts/analyze_difficulty.py`.

`analysis_findings.json` is the consolidated interpretation snapshot. It
links each numerical finding to its evidence, conservative interpretation,
claim strength, and intended manuscript placement so that later revisions do
not need to reconstruct conclusions from chat or multiple evaluator files.

## Exploratory diagnostics

`exploratory/` contains the post-hoc Sarashina2-Vision evaluation JSONs and
`sarashina_summary.json`. The predeclared Sarashina trigger did not fire, so
these are not official baseline rows. They are retained for transparency
because output-format failure was itself informative.

## Excluded artifacts

Raw model responses, per-example predictions, labeled keyframes, API
checkpoints, OCR workbooks, source media, model caches, and Slurm logs remain
outside Git. They are retained privately for error analysis and release
preparation where applicable.
