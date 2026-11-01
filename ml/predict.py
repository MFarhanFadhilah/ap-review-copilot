"""Score new transaction lines: label, confidence band, route and reasons.

Library use (CLI, API and agent all call this)::

    from ml.predict import predict_records
    predict_records([{"Col1": "...", "Col2": "...", "Col3": 180.0, ...}])

Command line::

    python -m ml.predict --input new_lines.xlsx --output predictions.csv
    python -m ml.predict --input lines.csv --output out.csv --model models/eval_model.joblib

Nothing here re-implements a rule. Validation is ``clean.validate_record``,
column standardisation is ``clean.standardize_records``, routing is
``policy.route`` and the reasons are ``explain.explain``. This module only
wires them together and shapes the output.
"""
from __future__ import annotations

import argparse
import functools
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from ml.clean import FEATURE_COLS, standardize_records, validate_record
from ml.explain import explain
from ml.policy import AUTO_CLASS, route

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(open(ROOT / "config.yaml"))
DEFAULT_MODEL = ROOT / CFG["paths"]["model_dir"] / "model.joblib"
BANDS = CFG["routing"]["confidence_bands"]


@functools.lru_cache(maxsize=4)
def load_bundle(path: str | None = None) -> dict:
    """Load a saved model bundle once per process (see ``ml/train.py`` for its contents)."""
    return joblib.load(Path(path) if path else DEFAULT_MODEL)


def confidence_band(p: float) -> str:
    """High / medium / low, from the calibrated probability of the predicted class."""
    if p >= BANDS["high"]:
        return "high"
    return "medium" if p >= BANDS["medium"] else "low"


def format_confidence(p: float | None) -> str:
    """The one confidence format: one decimal place, and ">99.9%" at the top."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "—"
    return ">99.9%" if p >= 0.9995 else f"{p:.1%}"


def _as_records(records) -> list[dict]:
    """Accept a DataFrame or a list of dicts; keep the seven input columns; blanks become None."""
    df = records if isinstance(records, pd.DataFrame) else pd.DataFrame(list(records))
    df = df.reindex(columns=FEATURE_COLS).astype(object)
    return df.where(df.notna(), None).to_dict("records")


def predict_records(records, bundle: dict | None = None, top_k: int = 3, missing_columns=()) -> list[dict]:
    """Return one result per input record, in input order.

    Each result has: ``predicted_label``, ``confidence`` (calibrated),
    ``confidence_band``, ``runner_up_label``, ``route`` ("auto" or "review"),
    ``review_reasons``, ``top_reasons`` (plain-language, strongest first),
    ``validation_warnings`` and ``validation_errors``. A record that fails
    validation is not scored: its label and confidence are None and it is
    routed to review with the reason "validation error".
    """
    bundle = bundle or load_bundle()
    rows = _as_records(records)
    checks = [validate_record(r) for r in rows]
    results: list[dict | None] = [None] * len(rows)

    for i, check in enumerate(checks):
        if check["errors"]:
            results[i] = {
                "predicted_label": None, "confidence": None, "confidence_band": None, "runner_up_label": None,
                "route": "review", "review_reasons": ["validation error"], "top_reasons": [],
                "validation_warnings": check["warnings"], "validation_errors": check["errors"],
            }

    scorable = [i for i, check in enumerate(checks) if not check["errors"]]
    if scorable:
        X = standardize_records(pd.DataFrame([rows[i] for i in scorable], columns=FEATURE_COLS))
        model = bundle["model"]
        classes = list(model.classes_)
        proba = model.predict_proba(X)
        pred = np.array(classes)[proba.argmax(axis=1)]
        routing = route(
            pred,
            proba[:, classes.index(AUTO_CLASS)],
            is_credit_note=X["flag_credit_note"],
            vendor_seen=X["Col1"].isin(set(bundle["known_vendors"])),
            threshold=bundle["threshold"],
            warning_codes=[checks[i]["warning_codes"] for i in scorable],
            missing_columns=missing_columns,
        )
        explanations = explain(model, X, bundle["feature_means"], top_k=top_k)

        for k, i in enumerate(scorable):
            e = explanations[k]
            results[i] = {
                "predicted_label": e["predicted"],
                "confidence": round(e["confidence"], 4),
                "confidence_band": confidence_band(e["confidence"]),
                "runner_up_label": e["runner_up"],
                "route": routing["route"].iloc[k],
                "review_reasons": [r for r in routing["review_reasons"].iloc[k].split("; ") if r],
                "top_reasons": [f"{r['feature']} ({r['contribution']:+.2f})" for r in e["reasons"]],
                "validation_warnings": checks[i]["warnings"],
                "validation_errors": [],
            }
    return results


def read_table(source, name: str | None = None) -> pd.DataFrame:
    """Read a CSV, XLSX or XLS file (a path, or a file-like object with its file name).

    Excel files use the same reader as training (``clean.load_raw``); CSV is
    read as text so IDs such as "00128" keep their leading zeros.
    """
    suffix = Path(name or str(source)).suffix.lower()
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(source)
    if suffix == ".csv":
        return pd.read_csv(source, dtype=str)
    raise ValueError(f"unsupported file type {suffix or '(none)'}: use .csv, .xlsx or .xls")


def flatten_results(results: list[dict], top_k: int = 3) -> pd.DataFrame:
    """One flat row per result (lists joined with "; ", reasons as reason_1..reason_k), for CSV output."""
    rows = []
    for r in results:
        row = {k: v for k, v in r.items() if not isinstance(v, list)}
        row["review_reasons"] = "; ".join(r["review_reasons"])
        for n in range(top_k):
            row[f"reason_{n + 1}"] = r["top_reasons"][n] if n < len(r["top_reasons"]) else ""
        row["validation_warnings"] = "; ".join(r["validation_warnings"])
        row["validation_errors"] = "; ".join(r["validation_errors"])
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description="Label transaction lines and route them to auto-label or review.")
    parser.add_argument("--input", required=True, help="CSV or Excel file with columns Col1 to Col7")
    parser.add_argument("--output", required=True, help="CSV file to write")
    parser.add_argument("--model", help="model bundle (default: models/model.joblib)")
    parser.add_argument("--top-k", type=int, default=3, help="number of reasons per line (default 3)")
    args = parser.parse_args(argv)

    data = read_table(Path(args.input))
    results = predict_records(data, load_bundle(args.model), top_k=args.top_k)
    out = pd.concat([data.reset_index(drop=True), flatten_results(results, args.top_k)], axis=1)
    out.to_csv(args.output, index=False)

    routes = Counter(r["route"] for r in results)
    reasons = Counter(reason for r in results for reason in r["review_reasons"])
    print(f"{len(results)} lines → {routes['auto']} auto-labelled, {routes['review']} to review")
    for reason, n in reasons.most_common():
        print(f"  review reason: {reason:<40} {n}")
    print(f"written to {args.output}")
    return out


if __name__ == "__main__":
    main()
