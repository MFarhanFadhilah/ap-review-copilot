"""Score an uploaded file end to end, with plain-language messages for every problem.

Used by the dashboard (Tab 2) and tested on the hold-out file. The steps are:

1. **Schema**: match column names (case and spaces ignored) to Col1-Col7.
   Missing required columns stop the run; missing optional ones are filled
   with blanks. Extra columns are ignored. A label column (e.g.
   ``ClassificationLabel``) is ignored for prediction and, if present, used
   to report accuracy.
2. **Rows**: an empty file stops the run; completely blank rows are dropped;
   exact duplicates (same Col1-Col7 after standardisation) are flagged.
3. **Predict and route**: ``predict.predict_records``, which validates every
   row (``clean.validate_record``), standardises it, predicts and routes it
   (``policy.route``). Rows with bad types are not scored and say why.

Nothing here re-implements validation, features, prediction or routing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from ml.clean import FEATURE_COLS, REQUIRED_COLS, normalize_label, standardize_records
from ml.predict import flatten_results, predict_records, read_table

LABEL_NAMES = {"classificationlabel", "label", "category", "true_label"}
SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls"}
UNSUPPORTED_MESSAGE = "Unsupported file type, upload CSV, XLSX or XLS"


def upload_error(name: str) -> str | None:
    """The message to show for an upload that cannot be scored by its type, or None if the type is fine."""
    from pathlib import Path
    return None if Path(name).suffix.lower() in SUPPORTED_SUFFIXES else UNSUPPORTED_MESSAGE


@dataclass
class BatchResult:
    table: pd.DataFrame | None = None            # the input columns + predictions, one row per input row
    results: list[dict] = field(default_factory=list)   # predict_records output, same order as ``table``
    records: list[dict] = field(default_factory=list)   # the Col1-Col7 values actually scored, same order
    messages: list[tuple[str, str]] = field(default_factory=list)   # (level, text): error / warning / info
    metrics: dict | None = None                  # accuracy etc. when a label column was supplied
    missing_columns: list[str] = field(default_factory=list)   # optional columns absent from the whole file

    @property
    def ok(self) -> bool:
        return self.table is not None

    def add(self, level: str, text: str) -> None:
        self.messages.append((level, text))


def _match_columns(columns) -> tuple[dict, str | None, list[str]]:
    """Map uploaded column names to Col1-Col7; find a label column; list the extras."""
    mapping, label, extra = {}, None, []
    wanted = {c.lower(): c for c in FEATURE_COLS}
    for col in columns:
        key = str(col).strip().lower().replace(" ", "")
        if key in wanted and wanted[key] not in mapping.values():
            mapping[col] = wanted[key]
        elif key in LABEL_NAMES and label is None:
            label = col
        else:
            extra.append(str(col))
    return mapping, label, extra


def process_upload(raw: pd.DataFrame, bundle: dict, top_k: int = 3) -> BatchResult:
    """Validate the schema, flag duplicates, predict and route every row of ``raw``."""
    result = BatchResult()
    if raw is None or raw.shape[1] == 0 or len(raw) == 0:
        result.add("error", "The file is empty: no rows to score.")
        return result

    mapping, label_col, extra = _match_columns(raw.columns)
    missing_required = [c for c in REQUIRED_COLS if c not in mapping.values()]
    if missing_required:
        result.add("error", f"Missing required column(s): {', '.join(missing_required)}. "
                            f"Expected Col1 to Col7; found: {', '.join(map(str, raw.columns))}.")
        return result
    missing_optional = [c for c in FEATURE_COLS if c not in mapping.values()]
    if missing_optional:
        result.missing_columns = missing_optional
        result.add("warning", f"Optional column(s) missing, treated as blank: {', '.join(missing_optional)}. "
                              "Every row is sent to review with the reason 'input file had no ...'.")
    if extra:
        result.add("info", f"Ignored extra column(s): {', '.join(extra)}.")
    if label_col is not None:
        result.add("info", f"Label column '{label_col}' found: ignored for prediction, used only to report accuracy.")

    data = raw.rename(columns=mapping).reindex(columns=FEATURE_COLS)
    data = data.astype(object).where(data.notna(), None)
    blank = data.apply(lambda r: all(v is None or str(v).strip() == "" for v in r), axis=1)
    if blank.all():
        result.add("error", "Every row is blank in Col1 to Col7: nothing to score.")
        return result
    if blank.any():
        result.add("warning", f"{int(blank.sum())} completely blank row(s) skipped.")
    data, labels = data[~blank], (raw.loc[~blank, label_col] if label_col is not None else None)

    # Exact duplicates, compared after the same standardisation the model uses.
    std = standardize_records(data)
    first = std.groupby(FEATURE_COLS, dropna=False, sort=False).ngroup()
    first_row = pd.Series(np.arange(1, len(data) + 1), index=data.index).groupby(first.values).transform("min")
    position = pd.Series(np.arange(1, len(data) + 1), index=data.index)
    duplicate_of = first_row.where(first_row != position)
    if duplicate_of.notna().any():
        result.add("warning", f"{int(duplicate_of.notna().sum())} row(s) are exact duplicates of an earlier row "
                              "(flagged in 'duplicate_of_row'); possible duplicate postings.")

    result.records = data.to_dict("records")
    result.results = predict_records(data, bundle, top_k=top_k, missing_columns=missing_optional)
    flat = flatten_results(result.results, top_k)
    table = pd.concat([pd.DataFrame({"row": position.to_numpy()}), data.reset_index(drop=True), flat], axis=1)
    table["duplicate_of_row"] = duplicate_of.astype("Int64").array

    unscored = table["predicted_label"].isna()
    if unscored.any():
        result.add("warning", f"{int(unscored.sum())} row(s) could not be scored (bad or missing values); "
                              "see 'validation_errors'. They are routed to review.")
    if labels is not None:
        table["true_label"] = [_safe_label(v) for v in labels]
        result.metrics = _metrics(table, result)
    result.table = table
    n_auto = int((table["route"] == "auto").sum())
    result.add("info", f"Scored {len(table):,} row(s): {n_auto:,} auto-labelled, {len(table) - n_auto:,} to review.")
    return result


def _safe_label(value):
    try:
        return None if value is None or pd.isna(value) or str(value).strip() == "" else normalize_label(value)
    except ValueError:
        return None


def _metrics(table: pd.DataFrame, result: BatchResult) -> dict | None:
    scored = table[table["true_label"].notna() & table["predicted_label"].notna()]
    unusable = int(table["true_label"].isna().sum())
    if unusable:
        result.add("warning", f"{unusable} row(s) have a missing or unreadable label and are left out of accuracy.")
    if scored.empty:
        return None
    out = {}
    for name, rows in [("all scored rows", scored), ("duplicate copies removed", scored[scored["duplicate_of_row"].isna()])]:
        labels = sorted(set(rows["true_label"]) | set(rows["predicted_label"]))
        auto = rows["route"] == "auto"
        out[name] = {
            "rows": len(rows),
            "accuracy": accuracy_score(rows["true_label"], rows["predicted_label"]),
            "macro_f1": f1_score(rows["true_label"], rows["predicted_label"], labels=labels, average="macro", zero_division=0),
            "automation_rate": float(auto.mean()),
            "auto_precision": float((rows.loc[auto, "true_label"] == "Cat1").mean()) if auto.any() else None,
            "non_cat1_auto": int((auto & (rows["true_label"] != "Cat1")).sum()),
        }
    return out


def process_file(source, name: str, bundle: dict, top_k: int = 3) -> BatchResult:
    """Read an uploaded CSV / XLSX / XLS (path or file-like) and process it; unreadable files get a clear error."""
    result = BatchResult()
    try:
        raw = read_table(source, name)
    except pd.errors.EmptyDataError:
        result.add("error", "The file is empty: no rows to score.")
        return result
    except Exception as exc:   # unsupported type, corrupt file, wrong encoding, a spreadsheet with no sheet, ...
        text = str(exc).rstrip(".")
        if text.startswith("unsupported file type"):
            result.add("error", text[0].upper() + text[1:] + ".")
        else:
            result.add("error", f"Could not read '{name}' as a table ({type(exc).__name__}: {text}).")
        return result
    return process_upload(raw, bundle, top_k)
