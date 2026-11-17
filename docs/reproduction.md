# Reproducing the released benchmark

Run commands from the repository root. The paper evaluates the complete
1,842-example benchmark with frozen models: there is no official test split.
The commands below use released data, not private raw annotations or previous
model checkpoints. Historical incremental-rerun scripts are not included in
this publication package.

## 1. Install and validate (no model API calls)

Use Python 3.10 or newer in an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/validate_release.py
python -m pytest -q
```

Expected data counts: 1,842 examples, 206 benchmark frames, and 208 source
manifest frames. Tests requiring optional model packages may be skipped.

## 2. Run an image-free sanity baseline

```bash
python scripts/run_baseline.py --baseline random \
  --examples data/benchmark/examples.jsonl \
  --output outputs/public_reproduction/random_seed0.jsonl --seed 0
python scripts/run_eval.py \
  --predictions outputs/public_reproduction/random_seed0.jsonl \
  --output outputs/public_reproduction/random_seed0_eval.json \
  --method random --seed 0 --split all@v1
```

The published random/class-prior summary aggregates seeds 0 through 4; one
seed is a smoke check, not that aggregate. Class-prior is a full-corpus weak
sanity check, not a held-out learned baseline.

## 3. Reconstruct images before visual-model runs

Follow [keyframe reconstruction](keyframe_reconstruction.md), installing its
downloader and FFmpeg prerequisites separately. Use `local/keyframes` as the
image root. `local/` and `outputs/` are ignored by Git. Source availability,
download permissions, and possible JPEG differences still apply.

## 4. Prepare the final VLM conditions

```bash
python scripts/prepare_vlm_inputs.py \
  --examples data/benchmark/examples.jsonl \
  --images-root local/keyframes --condition vlm \
  --prompt-version vlm-ranking-v4 \
  --rendered-dir outputs/public_reproduction/plain/labeled \
  --output outputs/public_reproduction/plain/inputs.jsonl
python scripts/prepare_vlm_inputs.py \
  --examples data/benchmark/examples_latex.jsonl \
  --images-root local/keyframes --condition vlm+latex \
  --prompt-version vlm-ranking-latex-v4 \
  --rendered-dir outputs/public_reproduction/latex/labeled \
  --output outputs/public_reproduction/latex/inputs.jsonl
```

Preparation does not call a model. Input manifests omit gold answers. The
LaTeX condition uses released **human-reviewed** metadata, not automatic OCR.

For model backends, optional dependencies, parsing, and evaluation, continue
with [frozen VLM documentation](frozen_vlm.md). Use the manifests above and
new output paths; do not seed from private historical checkpoints. Inspect
`python scripts/run_vlm.py --help` before running. OpenAI calls incur costs;
credentials belong in environment variables, never in committed files.
Model access and runtime differences may prevent exact reproduction of a
historical served-model result.

## Result provenance and limitations

Compare with [the frozen report snapshot](../reports/paper/README.md).
Full per-example predictions, source media, raw annotator deliveries, and OCR
review workbooks are not redistributed. Thus aggregate IAA and some analyses
can be inspected but cannot all be regenerated from the public package alone.

Post-ARR trained-ranker experiments are exploratory and separate from the
paper's frozen evaluation. Keep their video-level folds and training results
distinct from the full-benchmark frozen-model results.
