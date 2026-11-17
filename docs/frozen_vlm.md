# Frozen VLM Scaffold

The frozen-VLM scaffold prepares model inputs and parses model outputs without
downloading or running a model.

## Data Flow

```text
ranking examples JSONL
-> labeled keyframes + VLM input manifest JSONL
-> external frozen VLM
-> raw VLM output JSONL
-> prediction JSONL
-> evaluator
```

Ranking examples contain gold answers for evaluation. They must not be sent to
the model as a complete record. The generated VLM input manifest deliberately
omits `gold_visual_ids` and `gold_object_types`.

## Candidate Labels

Every human-annotated candidate box is drawn on the full keyframe. Display
labels are neutral aliases (`C01`, `C02`, and so on), assigned in candidate
input order. Neutral aliases prevent annotation IDs such as `equation_04` from
revealing object type through their spelling.

Candidate outlines keep deterministic colors. Label badges are placed
greedily near their boxes without covering any candidate region or overlapping
other label badges; when a badge must move away from its box, a colored leader
line connects it to the box corner. This avoids both truncated aliases and
occluded mathematical content in frames with dense or nested candidates while
preserving the original box geometry.

The manifest stores the alias-to-`visual_id` mapping so the parser can convert
model output back to the dataset and prediction contracts. A model runner
should send only `labeled_image_path` and `prompt` to the model; the mapping is
local bookkeeping.

## Prepare Inputs

```bash
# condition vlm (default): image + candidate aliases only
python3 scripts/prepare_vlm_inputs.py \
  --examples data/benchmark/examples.jsonl \
  --images-root local/keyframes \
  --rendered-dir outputs/vlm/v1/labeled \
  --output outputs/vlm/v1/inputs.jsonl

# condition vlm+latex: same, plus each candidate's recognized LaTeX in the prompt
python3 scripts/prepare_vlm_inputs.py \
  --examples data/benchmark/examples_latex.jsonl \
  --rendered-dir outputs/vlm_latex/v1/labeled \
  --output outputs/vlm_latex/v1/inputs.jsonl \
  --condition vlm+latex --images-root local/keyframes
```

One runner, two ablation conditions: `--condition vlm` or `--condition vlm+latex`.
The prompt version defaults to the one registered for the condition
(`vlm-ranking-v4` / `vlm-ranking-latex-v4`); `--prompt-version` overrides it.
The v4 prompts retain the v2 exact-count/compliance instructions and remove
terminal repeated-mention bookkeeping suffixes such as `_00` from marked
transcript mentions before model submission, including parenthesized
mathematics such as `g(t)_00`. This prevents accidental overlap between a
mention like `the equation_01` and a candidate annotation ID. The immutable
v1--v3 prompts remain registered for reproducing earlier runs. The
LaTeX condition reads each candidate's `latex` field (added by the candidate
LaTeX merge step), lists it per label, and shows `(no LaTeX available)` for
unrecognized candidates so none are dropped. The choice of VLM is orthogonal:
the manifest is model-agnostic and selected only at inference time.

The output validates against
[`vlm_input.schema.json`](../schemas/vlm_input.schema.json). Prompt text is
registered by immutable version name in `src/mathref/vlm/prompts.py`. Any prompt
wording change should receive a new version rather than silently changing an
existing experiment.

## Run the Model

`scripts/run_vlm.py` calls a model on each manifest row and writes raw outputs.
Backends are pluggable; the manifest is model-agnostic, so the same inputs feed
any backend:

```bash
# no-API dry run: deterministic ranking, exercises the whole chain
python3 scripts/run_vlm.py --inputs outputs/vlm/v1/inputs.jsonl \
  --out outputs/vlm/v1/raw_outputs.jsonl --backend mock

# OpenAI vision API (needs OPENAI_API_KEY); pin the model snapshot for the paper
python3 scripts/run_vlm.py --inputs outputs/vlm/v1/inputs.jsonl \
  --out outputs/vlm/v1/raw_outputs.jsonl --backend openai --model gpt-4o-2024-08-06

# self-hosted Qwen2.5-VL (GPU host; implemented, pending GPU verification)
# install once on the server: pip install -e '.[qwen]'
python3 scripts/run_vlm.py --inputs outputs/vlm/v1/inputs.jsonl \
  --out outputs/vlm/v1/raw_outputs.jsonl --backend qwen --device cuda:0 \
  --max-new-tokens 256 --resume
```

Use `--resume` for official runs. It appends and flushes one response per
example, reports a checkpoint every 25 responses, and skips example IDs already
present in the output file when the same command is submitted again.

Record the exact model name in the eval `meta.model` (and the API snapshot for
closed models) for reproducibility. Each row is one example:

```json
{"example_id":"example-001","response":"{\"ranking\":[\"C02\",\"C01\"]}"}
```

`response` may also be a parsed object or an array of candidate-label strings.
Rows validate against
[`vlm_output.schema.json`](../schemas/vlm_output.schema.json).

## Parse Predictions

```bash
python3 scripts/parse_vlm_outputs.py \
  --examples data/benchmark/examples.jsonl \
  --vlm-inputs outputs/vlm/v1/inputs.jsonl \
  --model-outputs outputs/vlm/v1/raw_outputs.jsonl \
  --output outputs/vlm/v1/predictions.jsonl
```

By default, the parser:

1. requires exactly one model output for every input example;
2. extracts the JSON `ranking`;
3. rejects candidate labels outside the known set;
4. removes duplicate labels while preserving their first occurrence;
5. appends omitted candidates in original candidate order;
6. maps neutral labels back to `visual_id`;
7. writes rows that validate against `prediction.schema.json`.

Appending omissions makes the output a complete ranking, as required by the
evaluator. The default `error` policy does not conceal unknown labels or
malformed responses: either raises an error.

For an official run whose raw compliance has already been audited, the parser
also supports an explicit deterministic fallback:

```bash
python scripts/parse_vlm_outputs.py \
  --examples data/processed/examples.jsonl \
  --vlm-inputs outputs/vlm/v1/inputs.jsonl \
  --model-outputs outputs/vlm/v1/raw_outputs.jsonl \
  --output outputs/vlm/v1/predictions.jsonl \
  --invalid-output-policy seeded-random \
  --fallback-seed 0
```

Under this opt-in policy, a malformed response or any response containing an
unknown label is replaced by a complete random permutation of the known
candidates. The permutation is seeded from SHA-256 of the fallback seed and
example ID, so it is stable across machines and input order. Duplicate known
labels are still kept at first occurrence, and omitted known labels are still
appended in candidate order. Report the audited malformed/unknown rate and the
resulting fallback rate alongside model results; never use fallback to conceal
raw prompt-compliance failures.

Audit raw responses before parsing so deterministic completion cannot hide
prompt-compliance failures:

```bash
python scripts/audit_vlm_outputs.py \
  --vlm-inputs outputs/vlm/v2/inputs.jsonl \
  --model-outputs outputs/vlm/v2/raw_outputs.jsonl \
  --out outputs/vlm/v2/audit.json --max-incomplete-rate 0.10
```

The command reports missing outputs, parse failures, unknown labels,
duplicates, and incomplete rankings. It exits nonzero if any output is
missing, any response fails to parse, any unknown label occurs, or the
incomplete-ranking rate exceeds the supplied threshold. This strict audit exit
status is intentional even when the separately invoked parser will use the
seeded-random fallback.

## Current Scope

Both `vlm` and `vlm+latex` conditions are implemented for the mock, OpenAI,
Qwen2.5-VL, and Sarashina backends. The paper's final GPT-5.4 and Qwen2.5-VL
reruns use prompt v4, placement-v3 rendering, and the complete 1,842-example
benchmark. The v4 manifests rewrite only prompt text/version in the v3
manifests; rendered images, candidate labels, candidate order, and all other
fields remain unchanged. Generated labeled images, manifests, raw outputs, and
predictions belong under ignored output directories and should not be
committed.
