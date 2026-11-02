"""The synthetic data generator: deterministic, every quirk present, output usable by the pipeline."""
from pathlib import Path

import pandas as pd
import pytest

from data.synthetic.generate import DEFAULT_CONFIG, generate, load_config, near_duplicates_path
from data.synthetic.measure import measure
from ml.clean import FEATURE_COLS, clean_training_data, load_raw, validate_record

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "sample.csv"
CFG = load_config(DEFAULT_CONFIG)


@pytest.fixture(scope="module")
def quirks():
    return measure(SAMPLE, CFG)


def _value(quirks, name):
    return quirks[name]["measured"]


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------
def test_same_seed_gives_identical_files(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    generate(CFG, 3000, 7, a)
    generate(CFG, 3000, 7, b)
    assert a.read_bytes() == b.read_bytes()
    assert near_duplicates_path(a).read_bytes() == near_duplicates_path(b).read_bytes()


def test_different_seed_gives_different_files(tmp_path):
    generate(CFG, 3000, 7, tmp_path / "a.csv")
    generate(CFG, 3000, 8, tmp_path / "b.csv")
    assert (tmp_path / "a.csv").read_bytes() != (tmp_path / "b.csv").read_bytes()


def test_committed_sample_matches_a_fresh_run(tmp_path):
    """The committed sample is exactly what the config and seed produce (CI checks this too)."""
    out = tmp_path / "sample.csv"
    generate(CFG, CFG["rows"], CFG["seed"], out)
    assert out.read_bytes() == SAMPLE.read_bytes()
    assert near_duplicates_path(out).read_bytes() == near_duplicates_path(SAMPLE).read_bytes()


# --------------------------------------------------------------------------
# Quirks, each near its configured rate
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cls, tol", [("Cat1", 0.005), ("Cat2", 0.005), ("Cat3", 0.002), ("Cat4", 0.002), ("Cat6", 0.002)])
def test_class_shares(quirks, cls, tol):
    q = quirks[f"class share {cls} (unique rows)"]
    assert q["measured"] == pytest.approx(q["configured"], abs=tol)


def test_cat5_is_nearly_absent_and_always_a_credit_note(quirks):
    assert _value(quirks, "Cat5 rows") == CFG["classes"]["cat5_rows"]
    assert _value(quirks, "Cat5 rows that are credit notes") == 1.0


def test_label_spellings(quirks):
    assert _value(quirks, "label spellings") == 11


def test_exact_duplicates_big_line_and_conflict(quirks):
    q = quirks["extra exact copies, share of rows (random + big line)"]
    assert q["measured"] == pytest.approx(q["configured"], abs=0.01)
    b = CFG["duplicates"]["big_line"]
    assert _value(quirks, "big line: copies / accounts") == f"{b['copies']} / {b['accounts']}"
    assert _value(quirks, "groups of identical lines with conflicting labels") == 1


def test_near_duplicates_are_planted_and_documented(quirks):
    assert _value(quirks, "near-duplicate pairs, share of rows") == pytest.approx(CFG["near_duplicates"]["rate"], abs=0.001)
    assert _value(quirks, "near-duplicate pairs valid (same vendor/amount/account/description, differ, <= 1 period apart)") == 1.0
    assert _value(quirks, "near-duplicate copies in the next period") == pytest.approx(
        CFG["near_duplicates"]["next_period_share"], abs=0.12)
    assert _value(quirks, "near-duplicate copies that are also exact duplicates") == 0


def test_ground_truth_is_not_in_the_raw_file():
    assert list(pd.read_csv(SAMPLE, nrows=1).columns) == [*FEATURE_COLS, "ClassificationLabel"]


def test_corrupted_ids_come_from_one_vendor(quirks):
    assert _value(quirks, "corrupted IDs from one vendor") >= 0.9
    assert _value(quirks, "that vendor's lines corrupted") == pytest.approx(CFG["ref_ids"]["corrupted_vendor_share"], abs=0.1)
    assert _value(quirks, "ID shapes present") == 5


def test_date_convention_change_and_legacy_rows(quirks):
    assert _value(quirks, "month-start dates before the switch") == 1.0
    assert _value(quirks, "month-end dates from the switch on") == 1.0
    assert _value(quirks, "legacy rows (before the main block)") == pytest.approx(CFG["timeline"]["legacy_share"], abs=0.003)


def test_fixed_fee_vendor_label_changes_twice(quirks):
    assert _value(quirks, "fixed-fee vendor label changes") == 2


def test_doc_type_follows_account_group(quirks):
    assert _value(quirks, "doc type matches the account group's usual type") >= 0.99


def test_new_vendors_late_in_the_timeline(quirks):
    q = quirks[f"lines in the last {CFG['new_vendors']['months']} months from vendors new in that window"]
    assert q["measured"] == pytest.approx(q["configured"], abs=0.03)


def test_some_description_tokens_carry_signal(quirks):
    assert _value(quirks, "of which >= 80% Cat2") >= 5
    assert _value(quirks, "of which >= 80% Cat2") < _value(quirks, "common description tokens (>= 10 rows)") / 2


def test_missing_descriptions_and_credit_notes(quirks):
    for name in ["empty descriptions, share of rows", "credit notes, share of rows"]:
        assert quirks[name]["measured"] == pytest.approx(quirks[name]["configured"], abs=0.01)


# --------------------------------------------------------------------------
# The pipeline can use the output unchanged
# --------------------------------------------------------------------------
def test_sample_passes_cleaning_and_validation():
    raw = load_raw(SAMPLE)
    clean, log, _ = clean_training_data(raw)
    removed = {s["step"]: s["rows_removed"] for s in log.steps}
    assert removed["unparseable amount or date"] == 0
    assert removed["exact duplicates"] > 0 and removed["conflicting labels"] > 0
    records = raw[FEATURE_COLS].astype(object).where(raw[FEATURE_COLS].notna(), None).to_dict("records")
    assert all(validate_record(r)["errors"] == [] for r in records)


def test_chunked_and_parquet_output(tmp_path):
    """Several chunks keep row ids consistent; Parquet holds the same rows as CSV."""
    cfg = {**CFG, "chunk_rows": 2000}
    generate(cfg, 5000, 3, tmp_path / "x.csv")
    generate(cfg, 5000, 3, tmp_path / "x.parquet", fmt="parquet")
    csv = load_raw(tmp_path / "x.csv")
    parquet = load_raw(tmp_path / "x.parquet")
    assert len(csv) == len(parquet) == 5000
    pd.testing.assert_frame_equal(csv.fillna(""), parquet.fillna(""), check_dtype=False)
    q = measure(tmp_path / "x.csv", cfg)
    assert q["near-duplicate pairs valid (same vendor/amount/account/description, differ, <= 1 period apart)"]["measured"] == 1.0


# --------------------------------------------------------------------------
# Structural targets (ADR 0004), at the sample size and ten times larger
# --------------------------------------------------------------------------
STRUCTURE = {"rows from vendors with more than one class": 0.45, "vendors with one class": 0.89,
             "corrupted IDs, share of rows": 0.06}


def test_structure_matches_targets_in_the_sample(quirks):
    for name, target in STRUCTURE.items():
        assert quirks[name]["measured"] == pytest.approx(target, abs=0.03), name


def test_structure_holds_at_ten_times_the_rows(tmp_path):
    """Vendor counts grow with rows and sizes come from fixed quantiles, so structure does not drift with size."""
    out = tmp_path / "big.csv"
    generate(CFG, 60_000, 43, out)
    q = measure(out, CFG)
    for name, target in STRUCTURE.items():
        assert q[name]["measured"] == pytest.approx(target, abs=0.03), name


def test_generator_is_versioned():
    assert CFG["generator_version"] == "1.0"
