# Math Reference Grounding

Code and benchmark for **Grounding Referring Expressions to Handwritten
Mathematics in Instructional Conversation**.

Given a marked transcript mention, local transcript context, a keyframe,
and human-annotated candidate boxes, systems rank the candidates to identify
the visual referent. The benchmark contains 1,842 English and Japanese
examples from 206 usable keyframes across 21 instructional videos.

See the [reproduction guide](docs/reproduction.md) to get started and
[reports/paper](reports/paper/) for the paper's results. Source videos and
keyframe images are not redistributed; a reconstruction manifest covers
all 208 selected keyframes.

Fixed pipeline data contracts are documented in
[`docs/data_contracts.md`](docs/data_contracts.md), with machine-readable JSON
Schema files in [`schemas/`](schemas/).

The first experimental target is mention-to-box ranking:

- input: a transcript segment, a marked mention span, a keyframe image, and annotated candidate visual boxes
- output: a ranked list of candidate boxes and a top predicted referent
- first-stage assumption: use annotated gold boxes as candidates; automatic object detection is out of scope for the first baseline pass

## Project Layout

```text
data/
  benchmark/        # released image-free ranking examples
  keyframes/        # source and reconstruction manifest
  latex/            # reviewed candidate-level LaTeX
  raw/              # original annotations/images; ignored by git
  processed/        # generated examples; ignored by git
scripts/            # command-line entry points
src/mathref/        # package code
outputs/            # predictions, tables, figures; ignored by git
reports/paper/ # compact paper-result snapshot
docs/               # task, data, guideline, and evaluation documentation
```

## Public benchmark

The release-ready examples are:

- [`data/benchmark/examples.jsonl`](data/benchmark/examples.jsonl): ranking
  examples without candidate LaTeX;
- [`data/benchmark/examples_latex.jsonl`](data/benchmark/examples_latex.jsonl):
  the aligned reviewed-LaTeX condition.

Both files conform to
[`schemas/ranking_example.schema.json`](schemas/ranking_example.schema.json).
See [`data/benchmark/README.md`](data/benchmark/README.md) for data semantics
and [`docs/guidelines/`](docs/guidelines/) for the English and Japanese
annotation guidelines.

Install the package and validate the complete public release:

```bash
python3 -m pip install -e ".[dev]"
python3 scripts/validate_release.py
pytest -q
```

The expected release-validator summary is 1,842 examples, 206 benchmark
frames, and 208 source-manifest frames. Two source keyframes were retained in
the reconstruction manifest but excluded from the benchmark after annotation
validation.

## Experimental scope

The controlled first experiment uses:

1. random and class-prior sanity checks;
2. a frozen CLIP dual-encoder;
3. frozen VLM candidate ranking;
4. frozen CLIP and VLM ranking with reviewed candidate-level LaTeX metadata.

Class-prior ranking is retained only as a corpus-level weak sanity check for
the paper. It estimates `P(class)` from the complete benchmark (not a
held-out training split) and assigns each candidate
`P(class) / number_of_same_class_candidates_in_frame`, with seeded random
tie-breaking. Future trained systems must instead use video-level held-out
folds and estimate any learned statistics from training folds only.

Primary metrics are Accuracy / Recall@1, Recall@3, and MRR, with breakdowns by
visual object type, mention type, candidate-set size, and single-gold versus
multi-gold examples.
Spatially relaxed candidate-ranking metrics are reported separately for error
analysis; they compare the top-ranked gold candidate with the gold referent and
do not require a detector.

The compact official metrics, generated tables, compliance counts, aggregate
API cost, and analysis artifacts are under
[`reports/paper/`](reports/paper/). Raw model responses, per-example
predictions, API checkpoints, source media, and human-review workbooks are not
release artifacts.

## Maintainer pipeline from raw annotations

The following workflow rebuilds the non-learned pipeline from private raw
annotations. It is included for provenance; public users can begin with
`data/benchmark/examples.jsonl`. The commands assume annotation JSON and
reconstructed JPG files live under `data/raw/annotations/`.

```bash
# 1. Validate annotations
python3 scripts/validate_annotations.py \
  --input data/raw/annotations --output outputs/validation.json

# 2. Build ranking examples (image_path is stored root-free)
python3 scripts/build_dataset.py \
  --input data/raw/annotations --input-format annotation-dir \
  --output data/processed/examples.jsonl

# 3. Dataset statistics
python3 scripts/dataset_stats.py \
  --input data/processed/examples.jsonl --input-format examples-jsonl \
  --annotations data/raw/annotations \
  --output outputs/stats.json

# 4. Optional deterministic video-level split for future trained systems
python3 scripts/create_splits.py \
  --examples data/processed/examples.jsonl --output-dir data/splits --seed 13

# 5. Full-benchmark sanity baselines
python3 scripts/run_baseline.py --baseline random \
  --examples data/processed/examples.jsonl \
  --output outputs/preds/random_seed0.jsonl --seed 0
python3 scripts/run_baseline.py --baseline class-prior \
  --examples data/processed/examples.jsonl \
  --prior-examples data/processed/examples.jsonl \
  --output outputs/preds/class_prior_seed0.jsonl --seed 0

# 6. Evaluate (--output requires --method for reporting provenance)
python3 scripts/run_eval.py --predictions outputs/preds/random_seed0.jsonl \
  --output outputs/eval/random_seed0.json --method random --seed 0 --split all@v1
python3 scripts/run_eval.py --predictions outputs/preds/class_prior_seed0.jsonl \
  --output outputs/eval/class_prior_seed0.json --method class_prior --seed 0 --split all@v1

# 7. Generate paper tables (Markdown + LaTeX booktabs)
python3 scripts/generate_tables.py --eval-dir outputs/eval --out outputs/tables
```

The frozen-VLM conditions (`vlm`, `vlm+latex`) and the candidate-LaTeX pipeline
are separate steps; see [`docs/frozen_vlm.md`](docs/frozen_vlm.md),
[`docs/candidate_latex.md`](docs/candidate_latex.md), and
[`docs/reporting.md`](docs/reporting.md).

Keyframe images and source videos are not committed. Exact source-video,
download-section, clip-time, and checksum metadata for all 208 frames is in
[`data/keyframes/keyframes.jsonl`](data/keyframes/keyframes.jsonl); see
[`docs/keyframe_reconstruction.md`](docs/keyframe_reconstruction.md) for the
reconstruction commands and timestamp semantics.

## Frozen VLM Scaffold

Prepare labeled keyframes and answer-free VLM input manifests without running a
model. First reconstruct images under `local/keyframes` using
[keyframe reconstruction](docs/keyframe_reconstruction.md):

```bash
python3 scripts/prepare_vlm_inputs.py \
  --examples data/benchmark/examples.jsonl \
  --images-root local/keyframes \
  --rendered-dir outputs/vlm/v1/labeled \
  --output outputs/vlm/v1/inputs.jsonl \
  --prompt-version vlm-ranking-v4
```

After an external model runner writes raw responses, convert them into complete
prediction rankings:

```bash
python3 scripts/parse_vlm_outputs.py \
  --examples data/benchmark/examples.jsonl \
  --vlm-inputs outputs/vlm/v1/inputs.jsonl \
  --model-outputs outputs/vlm/v1/raw_outputs.jsonl \
  --output outputs/vlm/v1/predictions.jsonl
```

See [`docs/frozen_vlm.md`](docs/frozen_vlm.md) for the leakage boundary, JSONL
contracts, prompt versioning, and parser policy.

## Evaluation protocol

The frozen-model paper evaluates the complete 1,842-example benchmark and does
not define an official train/dev/test split. Primary metrics are Recall@1,
Recall@3, and MRR. For multi-gold examples, any gold ID is correct and MRR uses
the highest-ranked gold.

For future trained systems, split or cross-validate by
`(language, topic, video)` so that no video crosses folds. The split generator
remains available in `scripts/create_splits.py` for that purpose.

## Inter-annotator agreement

The pairwise evaluator is in `scripts/compute_iaa.py`. Private delivery
normalization tools are not included in this publication package. Compact primary and
exploratory result JSONs are released under `reports/paper/iaa/`. Raw
annotator deliveries and disagreement-level records are not redistributed.

## Licensing and citation

Software is licensed under Apache-2.0. Original annotation contributions,
reviewed candidate LaTeX, and annotation guidelines are released under CC BY
4.0. Underlying videos, imagery, and source-derived transcript text are
excluded from those grants; see [`LICENSES.md`](LICENSES.md) for the exact
scope.

Citation metadata is provided in [`CITATION.cff`](CITATION.cff).
