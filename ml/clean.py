"""Data cleaning and validation.

Two entry points:

* ``clean_training_data(raw_df)``: used once per training run. Normalises the
  labels, standardises every column, removes exact duplicates, resolves rows
  whose features are identical but whose labels disagree, and returns a
  ``CleaningLog`` that records every row removed and why.
* ``standardize_records(df)`` / ``validate_record(record)``: used at
  inference time (CLI, API, agent). They apply exactly the same column
  standardisation as training, so the model never sees data in a format it
  was not trained on.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FEATURE_COLS = ["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7"]
LABEL_COL = "ClassificationLabel"
TARGET_COL = "label"
VALID_DOC_TYPES = {"NoDoc", "Doc1", "Doc2", "Doc3"}
REQUIRED_COLS = ["Col1", "Col3", "Col5", "Col6"]   # a line cannot be scored without these

# Excel turns long numeric IDs into scientific notation (e.g. 3.17E+11). The
# anonymisation then replaced the letter E with a random letter, which gives
# values like "3.17Q+11". The original digits cannot be recovered.
_SCI_NOTATION = re.compile(r"^\d\.\d+[A-Z]\+\d+$")


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------
def normalize_label(raw: Any) -> str:
    """Map a raw label spelling to a canonical class name.

    Source files spell the same class many ways ("Category_3", "Category 3",
    "Category _3", "Categry_6", "category_1", ...). The only reliable signal
    in every spelling is the class number, so we keep that and discard the
    rest. A label with no digit is an error, not something to guess.
    """
    digits = re.sub(r"\D", "", str(raw))
    if not digits:
        raise ValueError(f"Label {raw!r} contains no class number")
    return f"Cat{int(digits)}"


# --------------------------------------------------------------------------
# Column standardisation (shared by training and inference)
# --------------------------------------------------------------------------
def _ref_id_to_str(value: Any) -> str:
    """Return the reference ID as text, undoing Excel's numeric parsing."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def standardize_records(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise the seven feature columns and add data-quality flags.

    Works on one row or many, with or without a label column.
    """
    out = df.copy()

    out["Col1"] = out["Col1"].fillna("").astype(str).str.strip()
    out["Col2"] = out["Col2"].map(_ref_id_to_str)
    out["Col3"] = pd.to_numeric(out["Col3"], errors="coerce")
    out["Col4"] = out["Col4"].fillna("").astype(str).str.strip()
    out["Col6"] = out["Col6"].fillna("").astype(str).str.strip()
    out["Col7"] = out["Col7"].fillna("NoDoc").astype(str).str.strip()

    # The source switched from month-start to month-end dates part-way through.
    # Both mean "this month's posting period", so we store the month start.
    dates = pd.to_datetime(out["Col5"], errors="coerce")
    out["period"] = dates.dt.to_period("M").dt.to_timestamp()

    # Data-quality flags. These are kept as columns so they can be reported
    # and, where useful, used as model features.
    out["flag_ref_id_corrupted"] = out["Col2"].str.match(_SCI_NOTATION)
    out["flag_description_missing"] = out["Col4"].eq("")
    out["flag_credit_note"] = out["Col3"] < 0
    out["flag_unknown_doc_type"] = ~out["Col7"].isin(VALID_DOC_TYPES)
    return out


# --------------------------------------------------------------------------
# Single-record validation (used by the API and the agent)
# --------------------------------------------------------------------------
def validate_record(record: dict) -> dict:
    """Check one incoming record and report problems in plain language.

    Returns ``{"errors": [...], "warnings": [...], "warning_codes": [...]}``.
    Errors mean the record cannot be scored; warnings mean it can be scored but
    a reviewer should know. ``warning_codes`` holds one short code per warning,
    in the same order, so the routing policy can act on them without parsing text.
    """
    errors: list[str] = []
    warnings: list[str] = []
    codes: list[str] = []

    for col in REQUIRED_COLS:
        if record.get(col) in (None, ""):
            errors.append(f"{col} is required but missing")

    if record.get("Col3") not in (None, ""):
        try:
            float(record.get("Col3"))
        except (TypeError, ValueError):
            errors.append("Col3 (amount) is not a number")

    if record.get("Col5") not in (None, "") and pd.isna(pd.to_datetime(record.get("Col5"), errors="coerce")):
        errors.append("Col5 (period) is not a valid date")

    if not errors:
        row = standardize_records(pd.DataFrame([record]).reindex(columns=FEATURE_COLS)).iloc[0]
        checks = [
            ("ref_id_corrupted", row["flag_ref_id_corrupted"],
             "Col2 (reference ID) looks like an Excel scientific-notation value; the original ID is lost"),
            ("description_missing", row["flag_description_missing"],
             "Col4 (description) is empty; the prediction relies on vendor, account and amount only"),
            ("credit_note", row["flag_credit_note"], "Col3 is negative (likely a credit note)"),
            ("unknown_doc_type", row["flag_unknown_doc_type"],
             f"Col7 value {record.get('Col7')!r} is not a known document type"),
        ]
        for code, raised, message in checks:
            if raised:
                codes.append(code)
                warnings.append(message)

    return {"errors": errors, "warnings": warnings, "warning_codes": codes}


# --------------------------------------------------------------------------
# Training-data cleaning with an audit log
# --------------------------------------------------------------------------
@dataclass
class CleaningLog:
    rows_in: int = 0
    rows_out: int = 0
    label_mapping: dict = field(default_factory=dict)
    steps: list = field(default_factory=list)

    def add(self, step: str, rows_removed: int, reason: str) -> None:
        self.steps.append({"step": step, "rows_removed": int(rows_removed), "reason": reason})

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=str))


def clean_training_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningLog, pd.DataFrame]:
    """Clean the labelled training file.

    Returns ``(clean_df, log, duplicate_report)``. ``duplicate_report`` lists
    every group of identical postings. These are removed from training, but
    they may be duplicate payments, which is worth showing to the client.
    """
    log = CleaningLog(rows_in=len(raw))
    df = raw.copy()

    # 1. Labels
    log.label_mapping = {str(k): normalize_label(k) for k in sorted(df[LABEL_COL].unique(), key=str)}
    df[TARGET_COL] = df[LABEL_COL].map(log.label_mapping)

    # 2. Standardise columns and add flags
    df = standardize_records(df)
    bad_amount = df["Col3"].isna() | df["period"].isna()
    df = df[~bad_amount]
    log.add("unparseable amount or date", bad_amount.sum(), "Cannot be used for training or scoring")

    # 3. Exact duplicates (features and label all identical)
    key = FEATURE_COLS
    df["n_copies"] = df.groupby(key + [TARGET_COL], dropna=False)[TARGET_COL].transform("size")
    dup_report = (
        df[df["n_copies"] > 1]
        .drop_duplicates(key + [TARGET_COL])
        .loc[:, key + [TARGET_COL, "n_copies"]]
        .sort_values("n_copies", ascending=False)
    )
    before = len(df)
    df = df.drop_duplicates(key + [TARGET_COL], keep="first")
    log.add(
        "exact duplicates",
        before - len(df),
        "Identical rows would leak between train and test and over-weight some invoices. "
        f"{len(dup_report)} groups kept once each and exported for review as possible duplicate postings.",
    )

    # 4. Same features, different labels: keep the majority label.
    n_labels = df.groupby(key, dropna=False)[TARGET_COL].transform("nunique")
    conflicts = df[n_labels > 1]
    if len(conflicts):
        # Majority over the original (pre-dedup) rows, so copy counts act as votes.
        votes = conflicts.groupby(key + [TARGET_COL], dropna=False)["n_copies"].sum().reset_index()
        winners = votes.sort_values("n_copies", ascending=False).drop_duplicates(key)
        resolved = conflicts.merge(winners[key + [TARGET_COL]], on=key + [TARGET_COL], how="inner")
        before = len(df)
        df = pd.concat([df[n_labels <= 1], resolved], ignore_index=True)
        log.add(
            "conflicting labels",
            before - len(df),
            f"{conflicts.groupby(key, dropna=False).ngroups} group(s) of identical rows with different labels; "
            "kept the majority label. This disagreement is a ceiling on achievable accuracy.",
        )

    df = df.drop(columns=[LABEL_COL]).sort_values(["period", "Col1"]).reset_index(drop=True)
    log.rows_out = len(df)
    return df, log, dup_report.reset_index(drop=True)


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read a labelled file: CSV, Parquet or Excel, chosen by the file suffix.

    CSV keeps Col2 as text, so IDs such as "00128" keep their leading zeros
    (the same rule as ``predict.read_table``). Excel stays the default for any
    other suffix, as before.
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype={"Col2": str})
    if suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_excel(path)
