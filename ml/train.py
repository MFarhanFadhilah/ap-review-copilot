"""Fit the production model on all labelled data and save it to ``models/``.

Usage::

    python -m ml.train                     # production model: all labelled months
    python -m ml.train --until-split \
        --output models/eval_model.joblib    # training months only, for the validation-month check

What it does, in order:

1. Re-cleans the development rows (the raw file minus the 10% hold-out, ``ml/split.py``)
   with ``clean.clean_training_data`` (never trusts a stale CSV).
2. Drops Cat5. It has two rows, both credit notes; the model cannot learn it
   and the routing policy sends every credit note to review instead.
3. Sets the auto-label threshold from out-of-fold predictions, so the
   threshold is never measured on rows the model was fitted on.
4. Fits the calibrated pipeline on all remaining rows.
5. Saves one joblib bundle (model + threshold + known vendors + explanation
   baseline + the training lines for the agent + metadata) and
   a human-readable ``training_summary.json`` next to it.

The model definition lives here and ``evaluate.py`` imports it, so the model that
was evaluated and the model that is shipped are built by the same function.
"""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from ml.calibration import RareClassCalibratedCV
from ml.clean import FEATURE_COLS, clean_training_data, load_raw
from ml.explain import feature_means
from ml.features import build_preprocessor
from ml.policy import AUTO_CLASS, choose_threshold
from ml.split import load_development_raw

ROOT = Path(__file__).resolve().parent.parent
UNLEARNABLE = ["Cat5"]


def build_model(C: float = 10, random_state: int = 42) -> Pipeline:
    """The chosen model: shared features + L2 logistic regression with balanced class weights."""
    return Pipeline([
        ("features", build_preprocessor()),
        ("model", LogisticRegression(class_weight="balanced", C=C, max_iter=5000, random_state=random_state)),
    ])


def build_calibrated_model(C: float = 10, random_state: int = 42, method: str = "temperature") -> CalibratedClassifierCV:
    """Wrap the model in temperature scaling.

    Temperature scaling divides all scores by one number before the softmax.
    That makes the confidence less over-stated without changing which class
    wins, so the labels stay those of the uncalibrated model.
    ``ensemble=False`` keeps one underlying model fitted on all rows.
    """
    return RareClassCalibratedCV(
        build_model(C, random_state),
        method=method,
        ensemble=False,
        cv=StratifiedKFold(5, shuffle=True, random_state=random_state),
    )


def out_of_fold_proba(model, X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> np.ndarray:
    """Probabilities for each row from a model that never saw that row."""
    cv = StratifiedKFold(5, shuffle=True, random_state=random_state + 1)
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=-1)


def fit_threshold(X: pd.DataFrame, y: pd.Series, target_precision: float, C: float = 10,
                  random_state: int = 42) -> tuple[float, dict]:
    """Choose the auto-label threshold on out-of-fold calibrated probabilities."""
    model = build_calibrated_model(C, random_state)
    proba = out_of_fold_proba(model, X, y, random_state)
    classes = np.array(sorted(y.unique()))
    pred = classes[proba.argmax(axis=1)]
    p_auto = proba[:, list(classes).index(AUTO_CLASS)]
    eligible = (X["Col3"] >= 0).to_numpy()  # credit notes are never auto-labelled
    threshold = choose_threshold(y, pred, p_auto, eligible, target_precision)

    auto = eligible & (pred == AUTO_CLASS) & (p_auto >= threshold)
    stats = {
        "oof_automation_rate": float(auto.mean()),
        "oof_auto_precision": float((y.to_numpy()[auto] == AUTO_CLASS).mean()) if auto.any() else None,
    }
    return threshold, stats


def train(cfg: dict, until_split: bool = False, output: str | Path | None = None,
          data: str | Path | None = None) -> dict:
    """Fit and save the bundle. ``until_split`` restricts training to the training months.

    ``data`` trains on a whole separate file instead of the development set. It is for
    extra generated evaluation sets, which have no hold-out of their own (``ml/evaluate.py --data``).
    """
    raw_path = Path(data) if data else ROOT / cfg["paths"]["raw_data"]
    raw = load_raw(raw_path) if data else load_development_raw(cfg)   # never the hold-out
    df, clean_log, _ = clean_training_data(raw)
    df = df[~df["label"].isin(UNLEARNABLE)].reset_index(drop=True)
    if until_split:
        df = df[df["period"] <= pd.Timestamp(cfg["split"]["train_end_period"])].reset_index(drop=True)
    X, y = df, df["label"]

    C, seed = cfg["model"]["C"], cfg["model"]["random_state"]
    target = cfg["routing"]["target_cat1_precision"]
    threshold, oof_stats = fit_threshold(X, y, target, C, seed)
    model = build_calibrated_model(C, seed).fit(X, y)

    trained_at = datetime.now(UTC)
    data_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    periods = [f"{df['period'].min():%Y-%m}", f"{df['period'].max():%Y-%m}"]
    metadata = {
        "model_version": f"{trained_at:%Y%m%d-%H%M%S}-{data_sha[:8]}" + ("-eval" if until_split else ""),
        "trained_at": trained_at.isoformat(timespec="seconds"),
        "raw_data_sha256": data_sha,
        "rows_used": len(df),
        "class_counts": y.value_counts().sort_index().to_dict(),
        "excluded_classes": UNLEARNABLE,
        "periods": periods,
        "model": f"temperature-calibrated LogisticRegression(C={C}, class_weight='balanced')",
        "target_cat1_precision": target,
        "threshold": threshold,
        "threshold_note": (
            f"Set out-of-fold on the labelled development months {periods[0]} to {periods[1]} "
            "(the 90% left after the hold-out, ml/split.py). "
            + ("Training months only: the evaluation bundle used to measure routing on the validation months."
               if until_split else
               "This is the production threshold. `python -m ml.evaluate` reports a different one, set on the "
               "training months only (to the split date), which is used only to evaluate the policy on the validation months.")
        ),
        **oof_stats,
        "sklearn_version": sklearn.__version__,
        "cleaning_steps": clean_log.steps,
    }
    bundle = {
        "model": model,
        "threshold": threshold,
        "known_vendors": sorted(df["Col1"].unique()),
        "feature_means": feature_means(model, X),   # baseline for ml.explain
        # The labelled lines the model learned from: the agent's similar-invoice search,
        # vendor history and novelty check use them, so serving never needs the raw file.
        "reference_lines": df[FEATURE_COLS + ["period", "label"]].reset_index(drop=True),
        "classes": list(model.classes_),
        "metadata": metadata,
    }

    model_path = Path(output) if output else ROOT / cfg["paths"]["model_dir"] / "model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    summary_name = "training_summary.json" if model_path.name == "model.joblib" else f"{model_path.stem}_summary.json"
    (model_path.parent / summary_name).write_text(json.dumps(metadata, indent=2, default=str))
    return metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fit the classifier and save the model bundle.")
    parser.add_argument("--until-split", action="store_true",
                        help="train on the training months only (up to config split.train_end_period)")
    parser.add_argument("--output", help="where to save the bundle (default: models/model.joblib)")
    args = parser.parse_args()
    summary = train(yaml.safe_load(open(ROOT / "config.yaml")), args.until_split, args.output)
    print(json.dumps({k: v for k, v in summary.items() if k != "cleaning_steps"}, indent=2, default=str))
