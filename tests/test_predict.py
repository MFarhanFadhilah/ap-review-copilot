from pathlib import Path

import pandas as pd
import pytest

from ml.predict import confidence_band, format_confidence, main, predict_records

KEYS = {"predicted_label", "confidence", "confidence_band", "runner_up_label", "route", "review_reasons",
        "top_reasons", "validation_warnings", "validation_errors"}


def test_output_schema(bundle, good_record, cat1_record):
    results = predict_records([good_record, cat1_record], bundle)
    assert len(results) == 2
    for r in results:
        assert set(r) == KEYS
        assert r["predicted_label"] in bundle["classes"]
        assert 0 <= r["confidence"] <= 1
        assert r["confidence_band"] in {"high", "medium", "low"}
        assert r["route"] in {"auto", "review"}
        assert isinstance(r["review_reasons"], list) and isinstance(r["validation_warnings"], list)
        assert 1 <= len(r["top_reasons"]) <= 3 and all(isinstance(t, str) for t in r["top_reasons"])
        assert r["validation_errors"] == []
        assert (r["route"] == "auto") == (r["review_reasons"] == [])


def test_known_examples(bundle, good_record, cat1_record):
    cat2, cat1 = predict_records([good_record, cat1_record], bundle)
    assert cat2["predicted_label"] == "Cat2" and cat2["route"] == "review"
    assert cat2["review_reasons"] == ["rare class (Cat2 predicted)"]
    assert cat1["predicted_label"] == "Cat1" and cat1["route"] == "auto" and cat1["confidence_band"] == "high"


def test_routing_rules_through_predict(bundle, cat1_record):
    new_vendor, credit, odd_doc = predict_records([
        {**cat1_record, "Col1": "Brand New Vendor Ltd"},
        {**cat1_record, "Col3": -4500.0},
        {**cat1_record, "Col7": "Doc9"},
    ], bundle)
    assert "unseen vendor" in new_vendor["review_reasons"]
    assert "credit note" in credit["review_reasons"]
    assert "validation warning (unknown document type)" in odd_doc["review_reasons"]
    assert all(r["route"] == "review" for r in (new_vendor, credit, odd_doc))


def test_invalid_record_is_not_scored(bundle, good_record):
    bad, ok = predict_records([{**good_record, "Col3": "abc"}, good_record], bundle)
    assert bad["predicted_label"] is None and bad["confidence"] is None
    assert bad["route"] == "review" and bad["review_reasons"] == ["validation error"]
    assert bad["validation_errors"] == ["Col3 (amount) is not a number"]
    assert ok["predicted_label"] == "Cat2"   # one bad row does not stop the others


def test_accepts_a_dataframe(bundle, good_record):
    assert predict_records(pd.DataFrame([good_record]), bundle)[0]["predicted_label"] == "Cat2"


@pytest.mark.parametrize("p, band", [(0.95, "high"), (0.9, "high"), (0.8, "medium"), (0.7, "medium"), (0.5, "low")])
def test_confidence_bands(p, band):
    assert confidence_band(p) == band


def test_cli_writes_one_row_per_input(tmp_path, good_record, cat1_record, bundle):
    src = tmp_path / "in.csv"
    pd.DataFrame([good_record, cat1_record, {**good_record, "Col2": "00128"}]).to_csv(src, index=False)
    out = main(["--input", str(src), "--output", str(tmp_path / "out.csv")])
    written = pd.read_csv(tmp_path / "out.csv", dtype={"Col2": str})
    assert len(out) == len(written) == 3
    assert {"predicted_label", "confidence_band", "route", "review_reasons", "reason_1", "reason_2", "reason_3"} <= set(written.columns)
    assert written["Col2"].iloc[2] == "00128"   # IDs are read as text, leading zeros kept


def test_bundle_trained_from_the_command_line_loads_elsewhere(tmp_path):
    """A model saved by `python -m ml.train` must load in another program (no classes pickled from __main__)."""
    import subprocess
    import sys

    import joblib
    out = tmp_path / "bundle.joblib"
    subprocess.run([sys.executable, "-m", "ml.train", "--until-split", "--output", str(out)],
                   cwd=Path(__file__).resolve().parent.parent, check=True, capture_output=True)
    bundle = joblib.load(out)
    assert bundle["threshold"] > 0 and type(bundle["model"]).__module__ == "ml.calibration"


@pytest.mark.parametrize("p, text", [(0.9996, ">99.9%"), (0.9994, "99.9%"), (0.6656, "66.6%"), (0.05, "5.0%"), (None, "—")])
def test_one_confidence_format(p, text):
    assert format_confidence(p) == text
