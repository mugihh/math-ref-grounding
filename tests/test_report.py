import json
from pathlib import Path

from jsonschema import Draft202012Validator

from mathref.report.aggregate import group_rows, load_results, unmatched_methods
from mathref.report.generate import generate
from mathref.report.render import MISSING, Num, render_latex, render_markdown
from mathref.report.tables import all_tables, main_table
from mathref.utils.io import write_json

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"


def _schema(name: str) -> dict:
    with (SCHEMA_DIR / name).open(encoding="utf-8") as stream:
        return json.load(stream)


def _spatial(v: float) -> dict:
    return {"count": 10, "coverage": v, "iou": v, "median_normalized_center_distance": v,
            "center_distance_at_0_05": v, "center_distance_at_0_10": v}


def _bucket(v: float) -> dict:
    return {"count": 5, "recall_at_1": v, "recall_at_3": v, "mrr": v, "spatial": _spatial(v)}


def _result(v: float, method: str, seed=None, condition=None, model=None) -> dict:
    return {
        "count": 100, "recall_at_1": v, "recall_at_3": v + 0.1, "mrr": v + 0.05,
        "spatial": _spatial(v),
        "by_gold_object_type": {"equation": _bucket(v), "span": _bucket(v - 0.05)},
        "by_mention_type": {"implicit": _bucket(v), "explicit": _bucket(v + 0.05)},
        "by_candidate_count": {"1-5": _bucket(v), "6-10": _bucket(v - 0.1)},
        "by_gold_count": {"single_gold": _bucket(v), "multi_gold": _bucket(v - 0.2)},
        "meta": {"method": method, "model": model, "prompt_version": None,
                 "seed": seed, "split": "test@v1", "condition": condition},
    }


def test_results_with_meta_validate():
    validator = Draft202012Validator(_schema("evaluation.schema.json"))
    validator.validate(_result(0.4, "random", seed=0))


def test_load_requires_meta(tmp_path):
    good = tmp_path / "r0.json"
    write_json(good, _result(0.4, "random", seed=0))
    bad = tmp_path / "bad.json"
    payload = _result(0.4, "random")
    del payload["meta"]
    write_json(bad, payload)
    load_results([good])  # ok
    try:
        load_results([bad])
        raise AssertionError("expected ValueError for missing meta")
    except ValueError:
        pass


def test_main_table_aggregates_seeds():
    results = [
        _result(0.40, "random", seed=0), _result(0.50, "random", seed=1),
        _result(0.70, "class_prior", seed=0),
        _result(0.80, "frozen_vlm", condition="vlm"),
        _result(0.85, "frozen_vlm", condition="vlm+latex"),
    ]
    rows = group_rows(results)
    assert [r["display"] for r in rows] == ["Random", "Class-prior", "Frozen VLM", "Frozen VLM + LaTeX"]
    table = main_table(rows)
    # Random R@1 mean of 0.40/0.50 = 0.45 with a std; single-run rows have no std.
    random_r1 = table.rows[0][1]
    assert isinstance(random_r1, Num) and abs(random_r1.mean - 0.45) < 1e-9 and random_r1.std is not None
    vlm_r1 = table.rows[2][1]
    assert isinstance(vlm_r1, Num) and vlm_r1.std is None


def test_clip_rows_and_unmatched_warning():
    results = [
        _result(0.16, "random", seed=0),
        _result(0.25, "clip_dual_encoder", condition="plain"),
        _result(0.27, "clip_dual_encoder", condition="latex"),
        _result(0.99, "mystery_method", condition="weird"),
    ]
    displays = [row["display"] for row in group_rows(results)]
    assert "CLIP dual-encoder" in displays
    assert "CLIP dual-encoder + LaTeX" in displays
    # an unrecognized (method, condition) is surfaced, not silently dropped.
    assert ("mystery_method", "weird") in unmatched_methods(results)
    assert ("random", None) not in unmatched_methods(results)


def test_vlm_models_are_separate_rows_not_aggregated_as_seeds():
    results = [
        _result(0.50, "frozen_vlm", seed=0, condition="vlm", model="gpt-5.4-2026-03-05"),
        _result(0.29, "frozen_vlm", seed=0, condition="vlm", model="Qwen/Qwen2.5-VL-7B-Instruct"),
        _result(0.72, "frozen_vlm", seed=0, condition="vlm+latex", model="gpt-5.4-2026-03-05"),
    ]
    rows = group_rows(results)
    assert [row["display"] for row in rows] == [
        "GPT-5.4",
        "GPT-5.4 + LaTeX",
        "Qwen2.5-VL-7B",
    ]
    assert all(len(row["results"]) == 1 for row in rows)


def test_percent_formatting():
    cell = Num(0.4521, 0.0123)
    assert cell.markdown() == "0.452 ± 0.012"
    assert cell.markdown(percent=True) == "45.2 ± 1.2"
    assert cell.latex(percent=True) == "45.2 $\\pm$ 1.2"
    assert Num(0.8).markdown(percent=True, decimals=0) == "80"


def test_missing_bucket_renders_dash():
    # object_type table asks for 'term'/'figure' buckets absent from these results.
    results = [_result(0.4, "random", seed=0)]
    tables = all_tables(results)
    md = render_markdown(tables["breakdown_object_type_r1"])
    assert MISSING in md


def test_generate_writes_md_and_latex(tmp_path):
    paths = []
    for i, v in enumerate((0.40, 0.50)):
        p = tmp_path / f"random_{i}.json"
        write_json(p, _result(v, "random", seed=i))
        paths.append(p)
    out = tmp_path / "tables"
    written = generate(paths, out)
    assert (out / "main.md").exists() and (out / "main.tex").exists()
    assert (out / "report.md").exists()
    tex = (out / "main.tex").read_text()
    assert "\\toprule" in tex and "\\bottomrule" in tex and "$\\pm$" in tex
    # no raw non-ASCII (e.g. bare ±) leaks into LaTeX, and underscores are escaped.
    assert "±" not in tex
    obj_tex = (out / "breakdown_object_type_r1.tex").read_text()
    assert "figure\\_part" in obj_tex and "figure_part" not in obj_tex.replace("figure\\_part", "")
    # combined report includes the spatial and breakdown tables too.
    report = (out / "report.md").read_text()
    assert "Auxiliary spatial evaluation" in report and "Breakdown by" in report
