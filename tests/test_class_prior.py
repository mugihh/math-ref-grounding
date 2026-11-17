from collections import Counter

from mathref.baselines.class_prior import learn_class_prior, predict_examples


def _example(candidates: list[tuple[str, str]]) -> dict:
    return {
        "example_id": "example",
        "frame_id": "frame",
        "mention": {"text": "this"},
        "gold_visual_ids": ["term_0"],
        "gold_object_types": ["term"],
        "candidates": [
            {"visual_id": visual_id, "object_type": object_type, "bbox": [0, 0, 1, 1]}
            for visual_id, object_type in candidates
        ],
    }


def test_learn_class_prior_counts_training_gold_types() -> None:
    examples = [
        {"gold_object_types": ["span"]},
        {"gold_object_types": ["span", "term"]},
    ]
    assert learn_class_prior(examples) == Counter({"span": 2, "term": 1})


def test_candidate_score_divides_prior_by_class_candidates() -> None:
    example = _example(
        [("span_0", "span"), ("span_1", "span"), ("term_0", "term")]
    )
    row = predict_examples(
        [example],
        Counter({"span": 6, "term": 4}),
        seed=13,
    )[0]

    assert row["scores"]["span_0"] == 0.3
    assert row["scores"]["span_1"] == 0.3
    assert row["scores"]["term_0"] == 0.4
    assert row["ranked_visual_ids"][0] == "term_0"
    assert row["baseline"] == "class_prior_uniform"


def test_seeded_tie_breaking_is_reproducible() -> None:
    example = _example(
        [("term_0", "term"), ("term_1", "term"), ("term_2", "term")]
    )
    prior = Counter({"term": 3})
    first = predict_examples([example], prior, seed=7)
    second = predict_examples([example], prior, seed=7)
    assert first[0]["ranked_visual_ids"] == second[0]["ranked_visual_ids"]


def test_scores_condition_on_classes_present_in_frame() -> None:
    example = _example([("term_0", "term"), ("term_1", "term")])
    row = predict_examples(
        [example],
        Counter({"span": 9, "term": 1}),
        seed=13,
    )[0]
    assert row["scores"] == {"term_0": 0.5, "term_1": 0.5}


def test_unseen_candidate_classes_fall_back_to_uniform() -> None:
    example = _example([("figure_0", "figure"), ("term_0", "term")])
    row = predict_examples(
        [example],
        Counter({"span": 1}),
        seed=13,
    )[0]
    assert row["scores"] == {"figure_0": 0.5, "term_0": 0.5}
