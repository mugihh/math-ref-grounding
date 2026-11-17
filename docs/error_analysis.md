# Error Analysis

This analysis uses the final prompt-v4 GPT-5.4 predictions on all 1,842
examples. It compares the image-and-transcript condition with the otherwise
matched condition that adds reviewed candidate-level LaTeX. The rank of an
example is the best one-based rank assigned to any of its gold visual IDs.

## Main finding

Reviewed symbolic metadata helps primarily by separating fine-grained
mathematical candidates when the candidate inventory is large. It is not a
replacement for discourse grounding: it can hurt when several nested or
nearby regions have mathematically plausible transcriptions and the target
depends on deictic context or annotation granularity.

Across paired examples, LaTeX improves the gold rank for 710/1,842 (38.5%),
harms it for 177 (9.6%), and leaves it unchanged for 955 (51.8%). There are
506 wrong-to-correct R@1 transitions and 103 correct-to-wrong transitions,
giving the observed net R@1 gain of 21.9 percentage points.

## Quantitative slices

The candidate-count breakdown is the most useful body table. GPT-5.4 MRR
without/with LaTeX is 0.864/0.921 for 1--5 candidates, 0.647/0.841 for 6--10,
0.531/0.750 for 11--15, and 0.412/0.676 for 16 or more. The paired mean
reciprocal-rank gain therefore grows from 0.058 in the smallest bucket to
0.264 in the largest.

The gain is similar for lexically explicit and implicit mentions: MRR rises
from 0.694 to 0.869 for explicit mentions and from 0.535 to 0.705 for
implicit mentions. This indicates that reviewed symbols help candidate
discrimination even though implicit references still require transcript and
visual context.

By object type, the largest GPT-5.4 MRR gain is for equations
(0.680 to 0.915), followed by spans (0.655 to 0.832) and terms
(0.639 to 0.807). Figures and figure parts receive null LaTeX by policy and
show only negligible changes; their counts are also small (13 and 72
examples), so these slices are descriptive rather than strong comparative
evidence.

The absolute gain is larger in English (MRR 0.604 to 0.820) than Japanese
(0.711 to 0.818), but the final LaTeX-assisted performance is similar across
languages. The experiment does not isolate language from source, topic,
candidate density, or transcript provenance.

## Representative paired cases

- `en/probability/video4/007#m0009`, “Cov(X,Y) + Cov(X,Z)”: the gold span
  moves from rank 16 to rank 1 because its reviewed transcription directly
  matches the spoken expression.
- `jp/linear_algebra/video2/008#m0015`, “こういう式”: the gold equation moves
  from rank 17 to rank 1. Here transcript context identifies the equation
  under discussion, while LaTeX distinguishes the full equation from its
  nested spans.
- `en/calculus/video3/004#m0008`, “two derivatives”: the gold transformed
  span moves from rank 1 to rank 5. The added transcriptions favor the nearby
  differential equation and derivative-bearing regions, which are
  mathematically plausible but do not match the annotated referent
  granularity.
- `en/linear_algebra/video1/004#m0003`, “the fourth row”: the gold row span
  moves from rank 1 to rank 5 as individual row entries are promoted. This
  illustrates how symbolic similarity can compete with a discourse-level
  group reference.

These cases were chosen after inspecting the paired rankings, transcript
context, candidate geometry, reviewed transcriptions, and source keyframes.
They illustrate recurring mechanisms rather than estimating category-level
error rates.

## Reporting decision

Place the candidate-count MRR table in the body because it supports the main
finding and avoids overinterpreting small object-type buckets. Put
object-type, mention-type, per-language, and single/multi-gold results in the
appendix. Keep spatial scores auxiliary: the task supplies gold candidates
and is not a detection evaluation.
