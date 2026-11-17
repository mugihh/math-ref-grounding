# Paper-ready table generation

Turns evaluation JSON files into the paper's tables, in Markdown (for review) and
LaTeX `booktabs` (for `\input`). Metrics are already computed by the evaluator;
this layer only aggregates over seeds and renders.

## Provenance

Each `evaluation.json` carries an optional `meta` block written by the evaluator:

```
python scripts/run_eval.py --predictions outputs/preds/random_seed0.jsonl \
    --output outputs/eval/random_seed0.json \
    --method random --seed 0 --split test@v1
# frozen-model runs also pass --model / --prompt-version / --condition vlm|vlm+latex
```

`meta.method` is required for reporting; `generate_tables` errors on any file
without it. The schema keeps `meta` optional so older metric-only files still
validate.

## Generate

```
python scripts/generate_tables.py --eval-dir outputs/eval --out outputs/tables
# or explicit files: --eval a.json b.json ...
# --percent shows 52.0 instead of 0.520; --decimals overrides the precision
```

Number format is a pure render choice: cells hold raw `[0, 1]` values and are
formatted at write time. Default is 3 decimals (`0.520`); `--percent` switches to
percentages (`52.0`, 1 decimal) and appends "Values are percentages." to captions.

Writes per-table `<slug>.md` and `<slug>.tex`, plus a combined `report.md`.

## Tables

| Slug | Content |
|---|---|
| `main` | 4 methods × R@1 / R@3 / MRR (primary ranking metrics) |
| `spatial` | Auxiliary spatial table (coverage, IoU, center distances) |
| `breakdown_object_type_{r1,mrr}` | Per gold object type |
| `breakdown_mention_type_{r1,mrr}` | Per implicit / explicit / unknown |
| `breakdown_candidate_count_{r1,mrr}` | Per candidate-set size bucket |
| `gold_count` | single-gold vs multi-gold, full R@1 / R@3 / MRR |

Methods are fixed in `report/aggregate.py` (`METHOD_ROWS`): Random, Class-prior,
CLIP dual-encoder (`condition=plain`), CLIP dual-encoder + LaTeX (`condition=latex`),
Frozen VLM (`condition=vlm`), Frozen VLM + LaTeX (`condition=vlm+latex`). Rows with
no matching eval files are skipped; missing breakdown buckets render as `—`. A
`(method, condition)` present in the eval files but absent from `METHOD_ROWS` is
not silently dropped — `generate_tables` prints a warning listing it.

Multi-seed methods show `mean ± std` (sample std); single-run methods show the
plain value. The seed count is stated once in each caption. Spatial is kept out
of the main table on purpose: it is auxiliary, not the task's success criterion.

## Paper formatting

- **Number format (decided)**: decimals in `[0, 1]`, the tool default (no
  `--percent`). Uniform across R@1 / R@3 / MRR, and consistent with the IR
  convention of reporting MRR as a decimal. `--percent` exists but renders MRR as
  a percentage, which reads oddly, so it is not used.
- **Main results table (decided)**: in the body.
- **Everything else (TBD, decide once real numbers land)**: which breakdown earns
  a body slot vs the appendix is driven by which one supports the claim — for a
  reference-resolution paper the mention-type (implicit/explicit) breakdown is a
  strong body candidate, but defer until the numbers show where the effect is.
  Per-table breakdown metric defaults to R@1 for readability.
