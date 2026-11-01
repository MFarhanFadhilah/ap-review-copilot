"""The dashboard's processing function (ml.batch), on the hold-out file and on bad inputs."""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

from ml.batch import UNSUPPORTED_MESSAGE, process_file, process_upload, upload_error
from ml.clean import FEATURE_COLS, standardize_records
from ml.predict import predict_records, read_table
from ml.split import HOLDOUT_DIR


@pytest.fixture(scope="module")
def holdout():
    features = read_table(HOLDOUT_DIR / "holdout_features.xlsx")
    labels = pd.read_csv(HOLDOUT_DIR / "holdout_labels.csv")
    return features, labels


def _expected_duplicates(features: pd.DataFrame) -> int:
    """Rows that repeat an earlier row, counted independently with plain pandas."""
    std = standardize_records(features.reindex(columns=FEATURE_COLS))
    return int(std.duplicated(subset=FEATURE_COLS).sum())


def test_smoke_holdout_file(bundle, holdout):
    """The hold-out file, exactly as saved, goes through the whole pipeline."""
    result = process_file(HOLDOUT_DIR / "holdout_features.xlsx", "holdout_features.xlsx", bundle)
    features, _ = holdout
    assert result.ok and len(result.table) == len(features)
    assert set(result.table["route"]) <= {"auto", "review"}
    assert result.table["predicted_label"].notna().all()
    assert result.metrics is None                     # no label column, no accuracy
    assert int(result.table["duplicate_of_row"].notna().sum()) == _expected_duplicates(features)


def test_holdout_with_labels_reports_metrics_computed_here(bundle, holdout):
    """Reported metrics equal the same metrics computed independently in this test."""
    features, labels = holdout
    result = process_upload(features.assign(ClassificationLabel=labels["ClassificationLabel"].to_numpy()), bundle)
    dedup = result.metrics["duplicate copies removed"]

    keep = ~standardize_records(features.reindex(columns=FEATURE_COLS)).duplicated(subset=FEATURE_COLS).to_numpy()
    preds = predict_records(features[keep], bundle)
    y_true = labels["label"].to_numpy()[keep]
    y_pred = np.array([r["predicted_label"] for r in preds])
    auto = np.array([r["route"] == "auto" for r in preds])
    present = sorted(set(y_true) | set(y_pred))
    assert dedup["rows"] == keep.sum()
    assert dedup["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))
    assert dedup["macro_f1"] == pytest.approx(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0))
    assert dedup["automation_rate"] == pytest.approx(auto.mean())
    assert dedup["auto_precision"] == pytest.approx((y_true[auto] == "Cat1").mean())
    assert dedup["non_cat1_auto"] == int((auto & (y_true != "Cat1")).sum())
    assert any("Label column" in text for _, text in result.messages)


def test_empty_file(bundle):
    assert process_upload(pd.DataFrame(), bundle).messages == [("error", "The file is empty: no rows to score.")]
    result = process_file(io.BytesIO(b""), "empty.csv", bundle)
    assert not result.ok and result.messages[0][1].startswith("The file is empty")


def test_missing_required_column(bundle, good_record):
    result = process_upload(pd.DataFrame([good_record]).drop(columns="Col3"), bundle)
    assert not result.ok and "Missing required column(s): Col3" in result.messages[0][1]


def test_missing_optional_column_extra_column_and_header_case(bundle, good_record):
    frame = pd.DataFrame([good_record]).drop(columns="Col4").rename(columns={"Col1": " col1 "}).assign(Notes="x")
    result = process_upload(frame, bundle)
    texts = " ".join(t for _, t in result.messages)
    assert result.ok and "Optional column(s) missing, treated as blank: Col4" in texts and "Ignored extra column(s): Notes" in texts


def test_bad_types_are_flagged_not_scored(bundle, good_record):
    result = process_upload(pd.DataFrame([{**good_record, "Col3": "abc"}, {**good_record, "Col5": "not a date"}, good_record]), bundle)
    table = result.table
    assert table["predicted_label"].isna().tolist() == [True, True, False]
    assert table.loc[0, "validation_errors"] == "Col3 (amount) is not a number"
    assert (table.loc[:1, "route"] == "review").all()


def test_duplicates_are_flagged(bundle, good_record, cat1_record):
    result = process_upload(pd.DataFrame([good_record, cat1_record, good_record]), bundle)
    assert result.table["duplicate_of_row"].tolist()[2] == 1
    assert pd.isna(result.table["duplicate_of_row"].tolist()[0])
    assert any("exact duplicates" in t for _, t in result.messages)


def test_blank_rows_skipped_and_unreadable_files(bundle, good_record):
    frame = pd.DataFrame([good_record, {c: None for c in good_record}])
    result = process_upload(frame, bundle)
    assert len(result.table) == 1 and any("blank row" in t for _, t in result.messages)
    assert process_file(io.BytesIO(b"x"), "lines.pdf", bundle).messages[0][1].startswith("Unsupported file type .pdf")
    assert "Could not read" in process_file(io.BytesIO(b"garbage"), "lines.xlsx", bundle).messages[0][1]


def test_unreadable_labels_are_left_out_of_accuracy(bundle, good_record):
    frame = pd.DataFrame([good_record, good_record | {"Col3": 241.0}]).assign(label=["Category_2", "???"])
    result = process_upload(frame, bundle)
    assert result.metrics["all scored rows"]["rows"] == 1


BAD_FILES = Path(__file__).parent / "fixtures" / "bad_files"


@pytest.mark.parametrize("name, ok, expected", [
    ("missing_column.xlsx", True, "Optional column(s) missing, treated as blank: Col4"),
    ("bad_amount.csv", True, "3 row(s) could not be scored"),
    ("empty.csv", False, "The file is empty"),
    ("with_label.xlsx", True, "Label column 'ClassificationLabel' found"),
    ("injection_attempt.csv", True, "Scored 1 row(s)"),
])
def test_fixture_files(bundle, name, ok, expected):
    result = process_file(BAD_FILES / name, name, bundle)
    assert result.ok is ok
    assert any(expected in text for _, text in result.messages)


def test_missing_optional_column_routes_every_row_to_review(bundle):
    result = process_file(BAD_FILES / "missing_column.xlsx", "missing_column.xlsx", bundle)
    assert result.missing_columns == ["Col4"]
    assert (result.table["route"] == "review").all()
    assert result.table["review_reasons"].str.contains("input file had no Col4").all()


def test_unsupported_upload_message(bundle):
    assert upload_error("not_a_table.txt") == UNSUPPORTED_MESSAGE == "Unsupported file type, upload CSV, XLSX or XLS"
    assert upload_error("lines.XLSX") is None and upload_error("lines.csv") is None
    assert process_file(BAD_FILES / "not_a_table.txt", "not_a_table.txt", bundle).ok is False
