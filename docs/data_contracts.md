# Data Contracts

The core experiment pipeline uses normalized-frame, ranking-example,
prediction, and evaluation contracts. Frozen-VLM experiments add two
intermediate JSONL contracts for model inputs and raw model outputs.
Machine-readable JSON Schema files live in [`schemas/`](../schemas/).

## Shared Conventions

- Text is UTF-8.
- IDs are non-empty strings and are unique within their stated scope.
- `bbox` is `[x, y, width, height]` in image pixel coordinates.
- `image_size` is `[width, height]`; it may be `null` only when the image cannot
  be read. Spatial evaluation is unavailable for those rows.
- Transcript `span` is `[start, end]` with an end-exclusive character offset.
- Object type is one of `equation`, `span`, `term`, `figure`, or `figure_part`.
- Multiple gold IDs are represented as an array, never as a scalar string.
- Candidate `latex` is optional and reserved for the frozen VLM + LaTeX
  ablation. Missing recognition should be represented as `null`, not by
  dropping the candidate.

## Normalized Frames JSONL

Schema: [`normalized_frame.schema.json`](../schemas/normalized_frame.schema.json)

One row represents one annotated keyframe before conversion to ranking
examples.

Required top-level fields:

| Field | Type | Meaning |
|---|---|---|
| `frame_id` | string | Stable `language/topic/video/frame` identifier |
| `language`, `topic`, `video` | string | Video grouping metadata |
| `image_id` | string | Source annotation image ID |
| `image_path` | string | Root-free logical keyframe id (e.g. `en/topic/video1/000.jpg`); resolve to a file with `--images-root` at read time |
| `image_size` | `[int, int]` or `null` | Image width and height |
| `annotation_path` | string | Path to the source annotation |
| `caption` | object | `caption_id` and complete transcript text |
| `boxes` | array | Normalized visual objects |
| `mentions` | array | Mention spans and linked gold IDs |
| `warnings` | object | Parser warnings such as dangling visual IDs |

Each box contains `visual_id`, `object_type`, `bbox`, `source`, and `auto`.
Each mention contains `mention_id`, `caption_id`, `text`, `span`,
`gold_visual_ids`, `gold_object_types`, and `relation_ids`.

## Ranking Examples JSONL

Schema: [`ranking_example.schema.json`](../schemas/ranking_example.schema.json)

One row represents one mention-to-box ranking problem. It includes frame and
image metadata, the complete local transcript, one marked mention, one or more
gold IDs, and all annotated candidate boxes from the same keyframe.
The mention additionally contains a deterministic `mention_type`:
`implicit`, `explicit`, or `unknown`. This field remains optional for
compatibility with existing generated files; the evaluator places missing
values in the `unknown` bucket.

Important invariants not expressible in JSON Schema:

- `example_id` equals the mention ID and is globally unique.
- Every `gold_visual_id` occurs exactly once in `candidates`.
- Candidate `visual_id` values are unique within an example.
- `mention.span` indexes `caption.text`.
- All examples from the same `(language, topic, video)` belong to one data
  split.

## Predictions JSONL

Schema: [`prediction.schema.json`](../schemas/prediction.schema.json)

One row represents a complete candidate ranking for one ranking example.
Required prediction fields are:

| Field | Type | Meaning |
|---|---|---|
| `example_id`, `frame_id` | string | Join keys |
| `mention` | object | Mention, including `mention_type`, copied from the ranking example |
| `gold_visual_ids` | string array | Gold IDs used by the evaluator |
| `gold_object_types` | string array | Breakdown labels |
| `image_size`, `candidates` | values copied from example | Spatial evaluation inputs |
| `ranked_visual_ids` | string array | Best-to-worst complete ranking |
| `scores` | object | Optional numeric score per candidate; `{}` is allowed |
| `baseline` | string | Stable method or experiment name |

`ranked_visual_ids` must be a permutation of all candidate IDs with no omitted,
unknown, or duplicate IDs. When `scores` is non-empty, its keys must be
candidate IDs. Frozen-model parsers should append omitted candidates
deterministically before writing this format.

## Frozen-VLM Input JSONL

Schema: [`vlm_input.schema.json`](../schemas/vlm_input.schema.json)

One row identifies the rendered keyframe, immutable prompt version, prompt
text, condition, and neutral candidate-label mapping for an example. It must
not contain `gold_visual_ids` or `gold_object_types`. The mapping is local
bookkeeping; a model runner sends the prompt and labeled image to the model.

## Raw Frozen-VLM Output JSONL

Schema: [`vlm_output.schema.json`](../schemas/vlm_output.schema.json)

One row contains `example_id` and the raw `response` returned by a model runner.
The response may be text containing JSON, a parsed object with a `ranking`
array, or an array of candidate-label strings. The parser rejects unknown
labels, removes duplicates, and appends omitted candidates deterministically
before producing prediction JSONL.

## Evaluation JSON

Schema: [`evaluation.schema.json`](../schemas/evaluation.schema.json)

The top level contains `count`, `recall_at_1`, `recall_at_3`, `mrr`, `spatial`,
`by_gold_object_type`, `by_mention_type`, `by_candidate_count`, and
`by_gold_count`.

The candidate-count keys are `1-5`, `6-10`, `11-15`, and `16+`. Gold-count
keys are `single_gold` and `multi_gold`. Every breakdown bucket contains the
same exact metrics plus its own spatial coverage and diagnostics.

The spatial object always contains:

- `count`: rows for which spatial metrics were available;
- `coverage`: spatial count divided by the evaluation count.

When coverage is non-zero it also contains mean normalized center distance,
median normalized center distance, distance-at-0.05/0.10, bidirectional
center-inside rates, bidirectional overlap fractions, and IoU. Center distance
is lower-is-better; the other spatial metrics are higher-is-better.

## Compatibility Policy

Adding an optional field is backward compatible. Renaming a field, changing its
type, changing coordinate conventions, or making an optional field required is
a contract change and should update the schema, documentation, and tests in the
same commit.
