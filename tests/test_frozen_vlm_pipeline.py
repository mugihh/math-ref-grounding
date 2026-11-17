import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from PIL import Image

from mathref.eval.metrics import evaluate_predictions
from mathref.vlm.parse import (
    build_vlm_predictions,
    complete_candidate_ranking,
    seeded_random_candidate_ranking,
)
from mathref.vlm.audit import audit_vlm_outputs
from mathref.vlm.prepare import prepare_vlm_inputs
from mathref.vlm.prompts import render_prompt
from mathref.vlm.run import (
    BACKENDS,
    MockVLMBackend,
    QwenVLMBackend,
    SarashinaVLMBackend,
    run_vlm,
    run_vlm_resumable,
)


SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"


def _schema(name: str) -> dict:
    with (SCHEMA_DIR / name).open(encoding="utf-8") as stream:
        return json.load(stream)


def _toy_example(tmp_path: Path) -> dict:
    image_path = tmp_path / "keyframe.png"
    Image.new("RGB", (240, 160), "white").save(image_path)
    return {
        "example_id": "en/algebra/video1/000#m0000",
        "frame_id": "en/algebra/video1/000",
        "language": "en",
        "topic": "algebra",
        "video": "video1",
        "image_path": str(image_path),
        "image_size": [240, 160],
        "caption": {
            "caption_id": "1",
            "text": "We now simplify this expression.",
        },
        "mention": {
            "mention_id": "en/algebra/video1/000#m0000",
            "text": "this expression",
            "span": [16, 31],
            "mention_type": "implicit",
        },
        "gold_visual_ids": ["term_01"],
        "gold_object_types": ["term"],
        "candidates": [
            {
                "visual_id": "term_00",
                "object_type": "term",
                "bbox": [20, 30, 40, 30],
                "source": "gold",
            },
            {
                "visual_id": "term_01",
                "object_type": "term",
                "bbox": [90, 30, 40, 30],
                "source": "gold",
            },
            {
                "visual_id": "span_00",
                "object_type": "span",
                "bbox": [20, 90, 110, 30],
                "source": "gold",
            },
        ],
    }


def test_toy_frozen_vlm_pipeline_end_to_end(tmp_path: Path) -> None:
    example = _toy_example(tmp_path)
    vlm_input = prepare_vlm_inputs(
        [example],
        tmp_path / "rendered",
        prompt_version="vlm-ranking-v1",
    )[0]

    assert Path(vlm_input["labeled_image_path"]).is_file()
    assert "gold_visual_ids" not in vlm_input
    assert [row["label"] for row in vlm_input["candidate_labels"]] == [
        "C01",
        "C02",
        "C03",
    ]
    assert "term_01" not in vlm_input["prompt"]
    Draft202012Validator(_schema("vlm_input.schema.json")).validate(vlm_input)

    model_output = {
        "example_id": example["example_id"],
        "response": 'Result: {"ranking": ["C02", "C01"]}',
    }
    Draft202012Validator(_schema("vlm_output.schema.json")).validate(model_output)
    prediction = build_vlm_predictions(
        [example],
        [vlm_input],
        [model_output],
    )[0]

    assert prediction["ranked_visual_ids"] == [
        "term_01",
        "term_00",
        "span_00",
    ]
    assert prediction["baseline"] == "frozen_vlm:vlm-ranking-v1:vlm"
    Draft202012Validator(_schema("prediction.schema.json")).validate(prediction)

    evaluation = evaluate_predictions([prediction])
    Draft202012Validator(_schema("evaluation.schema.json")).validate(evaluation)
    assert evaluation["count"] == 1
    assert evaluation["recall_at_1"] == 1.0
    assert evaluation["recall_at_3"] == 1.0
    assert evaluation["mrr"] == 1.0
    assert evaluation["spatial"]["coverage"] == 1.0
    assert evaluation["spatial"]["normalized_center_distance"] == 0.0
    assert evaluation["by_gold_object_type"]["term"]["count"] == 1
    assert evaluation["by_mention_type"]["implicit"]["count"] == 1
    assert evaluation["by_candidate_count"]["1-5"]["count"] == 1
    assert evaluation["by_gold_count"]["single_gold"]["count"] == 1


def test_vlm_latex_condition_injects_candidate_latex(tmp_path: Path) -> None:
    example = _toy_example(tmp_path)
    # Merge step would have populated `latex`; one candidate is left unrecognized.
    example["candidates"][0]["latex"] = "x^2"
    example["candidates"][1]["latex"] = None
    example["candidates"][2]["latex"] = "\\frac{a}{b}"

    vlm_input = prepare_vlm_inputs([example], tmp_path / "rendered", condition="vlm+latex")[0]

    assert vlm_input["condition"] == "vlm+latex"
    assert vlm_input["prompt_version"] == "vlm-ranking-latex-v4"
    # Recognized LaTeX is shown per label; the missing one is kept, not dropped.
    assert "C01: x^2" in vlm_input["prompt"]
    assert "C02: (no LaTeX available)" in vlm_input["prompt"]
    assert "C03: \\frac{a}{b}" in vlm_input["prompt"]
    # candidate_labels carry the latex, and the row still validates.
    latex_by_label = {row["label"]: row.get("latex") for row in vlm_input["candidate_labels"]}
    assert latex_by_label == {"C01": "x^2", "C02": None, "C03": "\\frac{a}{b}"}
    Draft202012Validator(_schema("vlm_input.schema.json")).validate(vlm_input)


def test_mock_vlm_backend_end_to_end(tmp_path: Path) -> None:
    example = _toy_example(tmp_path)
    inputs = prepare_vlm_inputs([example], tmp_path / "rendered")
    inputs_path = tmp_path / "inputs.jsonl"
    with inputs_path.open("w", encoding="utf-8") as stream:
        for row in inputs:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    outputs = run_vlm(inputs_path, MockVLMBackend())
    assert [o["example_id"] for o in outputs] == [example["example_id"]]
    for output in outputs:
        Draft202012Validator(_schema("vlm_output.schema.json")).validate(output)

    prediction = build_vlm_predictions([example], inputs, outputs)[0]
    # mock ranks labels in reverse (C03, C02, C01) -> reversed candidate order.
    assert prediction["ranked_visual_ids"] == ["span_00", "term_01", "term_00"]
    assert len(prediction["ranked_visual_ids"]) == len(example["candidates"])
    Draft202012Validator(_schema("prediction.schema.json")).validate(prediction)


def test_resumable_vlm_skips_checkpointed_examples(tmp_path: Path) -> None:
    inputs_path = tmp_path / "inputs.jsonl"
    output_path = tmp_path / "raw_outputs.jsonl"
    rows = [
        {
            "example_id": f"example-{index}",
            "candidate_labels": [{"label": "C01"}, {"label": "C02"}],
        }
        for index in range(2)
    ]
    inputs_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    output_path.write_text(
        json.dumps(
            {
                "example_id": "example-0",
                "response": {"ranking": ["C01", "C02"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert run_vlm_resumable(
        inputs_path,
        output_path,
        MockVLMBackend(),
    ) == (2, 2)
    checkpoint = output_path.read_text(encoding="utf-8")
    assert [json.loads(line)["example_id"] for line in checkpoint.splitlines()] == [
        "example-0",
        "example-1",
    ]

    assert run_vlm_resumable(
        inputs_path,
        output_path,
        MockVLMBackend(),
    ) == (2, 2)
    assert output_path.read_text(encoding="utf-8") == checkpoint


class _FakeQwenInputs(dict):
    def __init__(self) -> None:
        super().__init__(input_ids=[[10, 11, 12]])
        self.input_ids = self["input_ids"]
        self.moved_to = None

    def to(self, device: str) -> "_FakeQwenInputs":
        self.moved_to = device
        return self


class _FakeQwenModel:
    device = "cuda:0"

    def __init__(self) -> None:
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [[10, 11, 12, 20, 21]]


class _FakeQwenProcessor:
    def __init__(self) -> None:
        self.inputs = _FakeQwenInputs()
        self.messages = None
        self.decoded_ids = None
        self.decode_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return self.inputs

    def batch_decode(self, token_ids, **kwargs):
        self.decoded_ids = token_ids
        self.decode_kwargs = kwargs
        return ['{"ranking":["C02","C01"]}']


def test_qwen_backend_builds_deterministic_multimodal_request(tmp_path: Path) -> None:
    image_path = tmp_path / "labeled.png"
    image_path.touch()
    backend = QwenVLMBackend(max_new_tokens=256)
    backend._engine = _FakeQwenModel()
    backend._processor = _FakeQwenProcessor()

    response = backend.generate(
        {"labeled_image_path": str(image_path), "prompt": "Rank the candidates."}
    )

    assert response == '{"ranking":["C02","C01"]}'
    assert backend._processor.messages[0]["content"] == [
        {"type": "image", "image": str(image_path)},
        {"type": "text", "text": "Rank the candidates."},
    ]
    assert backend._processor.inputs.moved_to == "cuda:0"
    assert backend._engine.generate_kwargs["max_new_tokens"] == 256
    assert backend._engine.generate_kwargs["do_sample"] is False
    assert backend._processor.decoded_ids == [[20, 21]]
    assert backend._processor.decode_kwargs == {
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": False,
    }


def test_qwen_backend_rejects_missing_image(tmp_path: Path) -> None:
    backend = QwenVLMBackend()
    backend._engine = _FakeQwenModel()
    backend._processor = _FakeQwenProcessor()

    with pytest.raises(FileNotFoundError, match="Labeled image does not exist"):
        backend.generate(
            {
                "labeled_image_path": str(tmp_path / "missing.png"),
                "prompt": "Rank the candidates.",
            }
        )


class _FakeSarashinaInputs(dict):
    def __init__(self) -> None:
        super().__init__(input_ids=[[10, 11, 12]])
        self.input_ids = self["input_ids"]
        self.moved_to = None

    def to(self, device: str) -> "_FakeSarashinaInputs":
        self.moved_to = device
        return self


class _FakeSarashinaModel:
    device = "cuda:0"

    def __init__(self) -> None:
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [[10, 11, 12, 20, 21]]


class _FakeSarashinaProcessor:
    """Mirrors the sarashina2-vision model-card processor surface: text-only
    chat template, image passed via __call__, custom stopping criteria."""

    def __init__(self) -> None:
        self.inputs = _FakeSarashinaInputs()
        self.messages = None
        self.template_kwargs = None
        self.call_kwargs = None
        self.stop_strings = None
        self.stopping_criteria = object()
        self.decoded_ids = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.template_kwargs = kwargs
        return "<s><|prefix|><|file|><|suffix|>### 指示:\nRank the candidates.\n\n### 応答:\n"

    def __call__(self, *, text, images, padding, return_tensors):
        self.call_kwargs = {
            "text": text,
            "images": images,
            "padding": padding,
            "return_tensors": return_tensors,
        }
        return self.inputs

    def get_stopping_criteria(self, stop_strings):
        self.stop_strings = stop_strings
        return self.stopping_criteria

    def batch_decode(self, token_ids, **kwargs):
        self.decoded_ids = token_ids
        return ['{"ranking":["C02","C01"]}']


def test_sarashina_backend_is_registered_with_verified_default_model() -> None:
    assert BACKENDS["sarashina"] is SarashinaVLMBackend
    backend = SarashinaVLMBackend()
    assert backend.name == "sarashina"
    # HF ID verified 2026-07-04: https://huggingface.co/sbintuitions/sarashina2-vision-8b
    assert backend.model == "sbintuitions/sarashina2-vision-8b"


def test_sarashina_backend_builds_deterministic_multimodal_request(tmp_path: Path) -> None:
    image_path = tmp_path / "labeled.png"
    Image.new("RGB", (32, 32), "white").save(image_path)
    backend = SarashinaVLMBackend(max_new_tokens=256)
    backend._engine = _FakeSarashinaModel()
    backend._processor = _FakeSarashinaProcessor()

    response = backend.generate(
        {"labeled_image_path": str(image_path), "prompt": "Rank the candidates."}
    )

    assert response == '{"ranking":["C02","C01"]}'
    processor = backend._processor
    # Text goes through the chat template alone; the image rides the processor call.
    assert processor.messages == [{"role": "user", "content": "Rank the candidates."}]
    assert processor.template_kwargs == {"add_generation_prompt": True}
    assert processor.call_kwargs["text"] == [
        "<s><|prefix|><|file|><|suffix|>### 指示:\nRank the candidates.\n\n### 応答:\n"
    ]
    assert len(processor.call_kwargs["images"]) == 1
    assert processor.call_kwargs["images"][0].mode == "RGB"
    assert processor.call_kwargs["padding"] is True
    assert processor.call_kwargs["return_tensors"] == "pt"
    assert processor.inputs.moved_to == "cuda:0"
    generate_kwargs = backend._engine.generate_kwargs
    assert generate_kwargs["max_new_tokens"] == 256
    assert generate_kwargs["do_sample"] is False
    assert generate_kwargs["stopping_criteria"] is processor.stopping_criteria
    assert processor.stop_strings == ["\n###"]
    assert processor.decoded_ids == [[20, 21]]


def test_sarashina_backend_rejects_missing_image(tmp_path: Path) -> None:
    backend = SarashinaVLMBackend()
    backend._engine = _FakeSarashinaModel()
    backend._processor = _FakeSarashinaProcessor()

    with pytest.raises(FileNotFoundError, match="Labeled image does not exist"):
        backend.generate(
            {
                "labeled_image_path": str(tmp_path / "missing.png"),
                "prompt": "Rank the candidates.",
            }
        )


def test_renderer_changes_the_toy_image(tmp_path: Path) -> None:
    example = _toy_example(tmp_path)
    vlm_input = prepare_vlm_inputs([example], tmp_path / "rendered")[0]

    with Image.open(vlm_input["labeled_image_path"]) as rendered:
        assert rendered.getpixel((20, 30)) != (255, 255, 255)


def test_label_placement_avoids_dense_overlaps() -> None:
    from mathref.vlm.render import _place_label

    occupied: list[tuple[int, int, int, int]] = []
    box = (40, 40, 80, 80)
    for _ in range(12):
        rect = _place_label(box, (28, 16), occupied, (160, 120))
        assert all(
            min(rect[2], other[2]) <= max(rect[0], other[0])
            or min(rect[3], other[3]) <= max(rect[1], other[1])
            for other in occupied
        )
        occupied.append(rect)


def test_label_placement_prefers_space_outside_candidate_boxes() -> None:
    from mathref.vlm.render import _place_label

    candidate_boxes = [
        (20, 20, 80, 80),
        (40, 40, 100, 100),
    ]
    rect = _place_label(
        candidate_boxes[0],
        (28, 16),
        occupied=[],
        image_size=(160, 120),
        forbidden=candidate_boxes,
    )

    assert all(
        min(rect[2], box[2]) <= max(rect[0], box[0])
        or min(rect[3], box[3]) <= max(rect[1], box[1])
        for box in candidate_boxes
    )


def test_unknown_candidate_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown candidate labels"):
        complete_candidate_ranking(
            ["C02"],
            [{"label": "C01", "visual_id": "term_00"}],
        )


def test_duplicate_labels_are_removed_before_omissions_are_appended() -> None:
    ranking = complete_candidate_ranking(
        ["C02", "C02"],
        [
            {"label": "C01", "visual_id": "term_00"},
            {"label": "C02", "visual_id": "term_01"},
            {"label": "C03", "visual_id": "term_02"},
        ],
    )

    assert ranking == ["term_01", "term_00", "term_02"]


def test_seeded_random_fallback_is_stable_and_complete() -> None:
    candidates = [
        {"label": "C01", "visual_id": "term_00"},
        {"label": "C02", "visual_id": "term_01"},
        {"label": "C03", "visual_id": "term_02"},
        {"label": "C04", "visual_id": "term_03"},
    ]

    first = seeded_random_candidate_ranking("example-1", candidates, seed=0)
    second = seeded_random_candidate_ranking("example-1", candidates, seed=0)

    assert first == second
    assert set(first) == {candidate["visual_id"] for candidate in candidates}


@pytest.mark.parametrize(
    "response",
    [
        "not JSON",
        {"ranking": ["C99"]},
    ],
)
def test_build_predictions_can_fallback_for_invalid_output(
    tmp_path: Path,
    response,
) -> None:
    example = _toy_example(tmp_path)
    inputs = prepare_vlm_inputs([example], tmp_path / "rendered")
    outputs = [{"example_id": example["example_id"], "response": response}]

    prediction = build_vlm_predictions(
        [example],
        inputs,
        outputs,
        invalid_output_policy="seeded-random",
        fallback_seed=0,
    )[0]

    assert set(prediction["ranked_visual_ids"]) == {
        candidate["visual_id"] for candidate in example["candidates"]
    }
    Draft202012Validator(_schema("prediction.schema.json")).validate(prediction)


def test_unknown_prompt_version_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown prompt version"):
        render_prompt(_toy_example(tmp_path), ["C01"], "not-a-version")


def test_v2_prompt_states_exact_candidate_count_and_compliance_check(tmp_path: Path) -> None:
    prompt = render_prompt(
        _toy_example(tmp_path),
        ["C01", "C02", "C03"],
        "vlm-ranking-v2",
    )

    assert "exactly 3 candidate boxes" in prompt
    assert "exactly 3 labels" in prompt
    assert "every listed candidate label occurs exactly once" in prompt


def test_v3_prompt_strips_annotation_suffixes_from_marked_mentions(
    tmp_path: Path,
) -> None:
    example = _toy_example(tmp_path)
    example["caption"]["text"] = (
        "Use <<<this_00>>> in <<<the equation_01>>>; keep raw_name_02 outside tags."
    )
    example["mention"]["text"] = "this_00"

    prompt = render_prompt(
        example,
        ["C01", "C02", "C03"],
        "vlm-ranking-v3",
    )

    assert "<<<this>>>" in prompt
    assert "<<<the equation>>>" in prompt
    assert "Target mention:\nthis" in prompt
    assert "raw_name_02 outside tags" in prompt
    assert "this_00" not in prompt
    assert "equation_01" not in prompt


def test_v4_prompt_strips_suffixes_after_parenthesized_math(
    tmp_path: Path,
) -> None:
    example = _toy_example(tmp_path)
    example["caption"]["text"] = (
        "Compare <<<g(t)_00>>> with <<<h(t)_01>>>; "
        "keep raw_name_02 outside tags."
    )
    example["mention"]["text"] = "g(t)_00"

    prompt = render_prompt(
        example,
        ["C01", "C02", "C03"],
        "vlm-ranking-v4",
    )

    assert "<<<g(t)>>>" in prompt
    assert "<<<h(t)>>>" in prompt
    assert "Target mention:\ng(t)" in prompt
    assert "raw_name_02 outside tags" in prompt
    assert "g(t)_00" not in prompt
    assert "h(t)_01" not in prompt


def test_v3_preserves_parenthesized_suffix_for_provenance(
    tmp_path: Path,
) -> None:
    example = _toy_example(tmp_path)
    example["caption"]["text"] = "Compare <<<g(t)_00>>>."
    example["mention"]["text"] = "g(t)_00"

    prompt = render_prompt(
        example,
        ["C01", "C02", "C03"],
        "vlm-ranking-v3",
    )

    assert "<<<g(t)_00>>>" in prompt
    assert "Target mention:\ng(t)_00" in prompt


def test_vlm_output_audit_reports_compliance_failures(tmp_path: Path) -> None:
    example = _toy_example(tmp_path)
    rows = [
        prepare_vlm_inputs(
            [{**example, "example_id": f"ex{i}"}],
            tmp_path / f"rendered{i}",
        )[0]
        for i in range(3)
    ]
    outputs = [
        {"example_id": "ex0", "response": '{"ranking":["C01","C02","C03"]}'},
        {"example_id": "ex1", "response": '{"ranking":["C01","C01"]}'},
        {"example_id": "ex2", "response": '{"ranking":["C99"]}'},
    ]

    report = audit_vlm_outputs(rows, outputs)

    assert report["complete_rankings"] == 1
    assert report["duplicate_labels"] == 1
    assert report["unknown_labels"] == 1
    assert report["incomplete_rankings"] == 2
    assert report["incomplete_rate"] == pytest.approx(2 / 3)
