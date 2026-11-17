# Evaluation

The main evaluation is exact visual object ranking.

Primary metrics:

- Accuracy / Recall@1
- Recall@3
- Mean Reciprocal Rank

Recommended breakdowns:

- gold object type: equation, span, term, figure, figure_part
- mention type: implicit, explicit, unknown
- candidate set size: 1-5, 6-10, 11-15, 16+
- gold count: single_gold, multi_gold

Each breakdown bucket reports its example count together with Accuracy /
Recall@1, Recall@3, MRR, and spatial coverage and diagnostics when candidate
geometry is available.

## Mention-Type Policy

Mention type is assigned deterministically while building ranking examples.
The current rules are provisional until the full dataset distribution and a
manual sample have been reviewed.

- `implicit`: the surface form requires discourse or visual context to identify
  the referent. This includes English pronouns and demonstratives such as
  `it`, `this system`, and `here`, and Japanese deictic/anaphoric forms such as
  `これ`, `この式`, `ここ`, and `こういう形`.
- `explicit`: the surface form supplies a mathematical name, expression,
  ordinal, position, property, or other descriptive identifier, such as
  `g(t)`, `the first row`, `行列Aの要素`, or `左辺`.
- `unknown`: the text is empty or unusable, or its language is unsupported.
  Ordinary linguistic ambiguity alone is not assigned to `unknown`.

## Candidate-Count and Multi-Gold Policy

Candidate count is the number of annotated candidates in the ranking example.
The fixed buckets are `1-5`, `6-10`, `11-15`, and `16+`.

Gold count separates examples with exactly one acceptable gold ID
(`single_gold`) from examples with more than one (`multi_gold`). This reveals
whether results differ when a mention has multiple valid referents. Exact
metrics continue to count any gold ID as correct, and MRR continues to use the
highest-ranked gold.

Relaxed or hierarchy-aware evaluation should be reported separately from exact-match metrics.

## Spatially Relaxed Candidate Evaluation

Spatial metrics compare the top-ranked annotated candidate box with the annotated
gold box. They do not evaluate box detection and do not require a detector.

Reported auxiliary metrics:

- normalized center distance, divided by the image diagonal
- proportion with normalized center distance at most 0.05 and 0.10
- prediction center inside a gold box
- gold center inside the predicted box
- prediction and gold overlap fractions
- intersection over union

For multi-gold mentions, distance uses the nearest gold box and overlap metrics
use the best gold box. Spatial coverage is reported because older prediction
files may not contain candidate boxes or image dimensions.

Prediction and evaluation output contracts are documented in
[`data_contracts.md`](data_contracts.md).
