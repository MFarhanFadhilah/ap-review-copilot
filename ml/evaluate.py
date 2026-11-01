"""Measure the model and the routing policy on months it never saw.

    python -m ml.evaluate      # writes models/evaluation/validation_summary.json and prints it

Steps:

1. Fit the evaluation bundle on the training months only (up to
   ``config.yaml → split.train_end_period``), with the same ``train()`` as the
   shipped model. Its threshold is set out-of-fold on those months only.
2. Score the validation months (the later months of the development set)
   through ``predict_records``, the same path as the CLI and the API.
3. Report accuracy, macro-F1, per-class precision and recall, the share of
   lines auto-labelled and the Cat1 precision of those auto-labels.
4. Compare with a vendor-lookup rule (each vendor's most common training
   label), the obvious thing to beat.
5. Score the 10% hold-out file with the shipped model (``models/model.joblib``).

Why macro-F1 is the headline: Cat1 is about 89% of lines, so a model that
always says Cat1 is 89% accurate and useless. Macro-F1 weights every class
equally. It is computed over the classes the model can predict (Cat5 is not
learned, see ``train.py``) that occur in the months scored. Cat5 lines still
count in accuracy and routing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from ml.batch import process_upload
from ml.clean import FEATURE_COLS, clean_training_data
from ml.features import time_split
from ml.policy import AUTO_CLASS
from ml.predict import load_bundle, predict_records, read_table
from ml.split import HOLDOUT_DIR, load_development_raw
from ml.train import train

ROOT = Path(__file__).resolve().parent.parent


def scored_classes(classes: list[str], *label_lists) -> list[str]:
    """The model's classes that occur in the truth or in any prediction being compared.

    A class with no lines and no predictions would count as an F1 of 0 and
    penalise everyone for nothing. Comparing two methods on different class
    sets would be unfair, so both are scored on this one shared set.
    """
    seen = set().union(*map(set, label_lists))
    return [c for c in classes if c in seen]


def routing_metrics(y_true: pd.Series, results: list[dict], labels: list[str]) -> dict:
    """Accuracy, macro-F1, per-class scores and routing numbers for scored lines."""
    pred = pd.Series([r["predicted_label"] for r in results], index=y_true.index)
    auto = pd.Series([r["route"] == "auto" for r in results], index=y_true.index)
    p, r, f, n = precision_recall_fscore_support(y_true, pred, labels=labels, zero_division=0)
    return {
        "lines": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1_classes": labels,
        "automation_rate": float(auto.mean()),
        "auto_precision": float((y_true[auto] == AUTO_CLASS).mean()) if auto.any() else None,
        "review_lines": int((~auto).sum()),
        "non_cat1_auto": int((auto & (y_true != AUTO_CLASS)).sum()),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "lines": int(n[i])}
                      for i, c in enumerate(labels)},
    }


def vendor_lookup(train_df: pd.DataFrame, valid: pd.DataFrame) -> pd.Series:
    """Each vendor's most common training label; vendors never seen get Cat1."""
    lookup = train_df.groupby("Col1")["label"].agg(lambda s: s.value_counts().index[0])
    return valid["Col1"].map(lookup).fillna(AUTO_CLASS)


def evaluate(cfg: dict, eval_model: str | Path | None = None) -> dict:
    eval_model = Path(eval_model or ROOT / cfg["paths"]["model_dir"] / "eval_model.joblib")
    train(cfg, until_split=True, output=eval_model)
    bundle = load_bundle(str(eval_model))

    clean, _, _ = clean_training_data(load_development_raw(cfg))
    train_df, valid = time_split(clean, cfg["split"]["train_end_period"])
    results = predict_records(valid[FEATURE_COLS], bundle)
    baseline = vendor_lookup(train_df, valid)
    labels = scored_classes(list(bundle["classes"]), valid["label"], baseline,
                            [r["predicted_label"] for r in results])

    summary = routing_metrics(valid["label"], results, labels)
    summary.update(
        threshold=float(bundle["threshold"]),
        validation_months=[f"{valid['period'].min():%Y-%m}", f"{valid['period'].max():%Y-%m}"],
        unseen_vendor_share=float((~valid["Col1"].isin(set(train_df["Col1"]))).mean()),
        vendor_lookup_macro_f1=float(f1_score(valid["label"], baseline, labels=labels, average="macro", zero_division=0)),
        vendor_lookup_per_class_f1={c: float(x) for c, x in zip(labels, f1_score(
            valid["label"], baseline, labels=labels, average=None, zero_division=0))},
    )

    shipped = ROOT / cfg["paths"]["model_dir"] / "model.joblib"
    if shipped.exists() and (HOLDOUT_DIR / "holdout_labels.csv").exists():
        features = read_table(HOLDOUT_DIR / "holdout_features.xlsx")
        labels = pd.read_csv(HOLDOUT_DIR / "holdout_labels.csv")
        result = process_upload(features.assign(ClassificationLabel=labels["ClassificationLabel"].to_numpy()),
                                load_bundle(str(shipped)))
        summary["holdout"] = {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                              for k, v in result.metrics["duplicate copies removed"].items()}
        summary["holdout"]["rows_as_delivered"] = int(len(result.table))

    out = ROOT / cfg["paths"]["model_dir"] / "evaluation" / "validation_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    print(json.dumps(evaluate(yaml.safe_load(open(ROOT / "config.yaml"))), indent=2))
