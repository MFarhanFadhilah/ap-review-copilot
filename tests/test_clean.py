import pytest

from ml.clean import clean_training_data, normalize_label, validate_record


# --------------------------------------------------------------------------
# Label mapping
# --------------------------------------------------------------------------
@pytest.mark.parametrize("raw, expected", [
    ("Category_3", "Cat3"), ("Category 3", "Cat3"), ("Category _3", "Cat3"),
    ("Categry_6", "Cat6"), ("category_1", "Cat1"), ("Category_05", "Cat5"),
])
def test_label_spellings_map_to_class_number(raw, expected):
    assert normalize_label(raw) == expected


def test_label_without_a_number_is_an_error():
    with pytest.raises(ValueError):
        normalize_label("Category")


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def test_cleaning_row_counts(raw_small):
    clean, log, _ = clean_training_data(raw_small)
    assert log.rows_in == 9
    assert log.rows_out == len(clean) == 3
    removed = {s["step"]: s["rows_removed"] for s in log.steps}
    assert removed == {"unparseable amount or date": 1, "exact duplicates": 4, "conflicting labels": 1}


def test_exact_duplicates_are_kept_once_and_reported(raw_small):
    clean, _, dup_report = clean_training_data(raw_small)
    assert (clean["Col1"] == "Ven10 Ven11").sum() == 1
    copies = dup_report.set_index(["Col1", "label"])["n_copies"]
    assert copies[("Ven10 Ven11", "Cat1")] == 2
    assert copies[("Ven20", "Cat1")] == 3


def test_conflicting_labels_resolve_to_the_majority(raw_small):
    clean, _, _ = clean_training_data(raw_small)
    conflict = clean[clean["Col1"] == "Ven20"]
    assert len(conflict) == 1
    assert conflict["label"].iloc[0] == "Cat1"   # 3 copies of Cat1 against 2 of Cat2


def test_cleaning_normalises_labels(raw_small):
    clean, log, _ = clean_training_data(raw_small)
    assert set(clean["label"]) == {"Cat1", "Cat3"}
    assert log.label_mapping["Category _3"] == "Cat3"


# --------------------------------------------------------------------------
# validate_record
# --------------------------------------------------------------------------
def test_valid_record_has_no_errors_or_warnings(good_record):
    assert validate_record(good_record) == {"errors": [], "warnings": [], "warning_codes": []}


@pytest.mark.parametrize("field", ["Col1", "Col3", "Col5", "Col6"])
def test_missing_required_field_is_one_error(good_record, field):
    result = validate_record({**good_record, field: None})
    assert result["errors"] == [f"{field} is required but missing"]


def test_non_numeric_amount_and_bad_date(good_record):
    errors = validate_record({**good_record, "Col3": "abc", "Col5": "not a date"})["errors"]
    assert errors == ["Col3 (amount) is not a number", "Col5 (period) is not a valid date"]


@pytest.mark.parametrize("change, code", [
    ({"Col2": "3.17Q+11"}, "ref_id_corrupted"),
    ({"Col4": ""}, "description_missing"),
    ({"Col3": -50.0}, "credit_note"),
    ({"Col7": "Doc9"}, "unknown_doc_type"),
])
def test_warnings_have_codes(good_record, change, code):
    result = validate_record({**good_record, **change})
    assert result["errors"] == []
    assert result["warning_codes"] == [code]
    assert len(result["warnings"]) == 1
