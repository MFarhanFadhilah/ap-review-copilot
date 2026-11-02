"""Measure the model and the routing policy on months it never saw.

    python -m ml.evaluate      # writes models/evaluation/validation_summary.json and prints it
    python -m ml.evaluate --data data/synthetic/out/eval_60k.csv     # a separate, larger set

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
5. Compare routing with the vendor lookup used as a router, which auto-labels a
   line only if every training line of its vendor is Cat1.
6. Put 95% bootstrap intervals on macro-F1 and Cat2 F1 for both methods and on
   their difference, and split the results by vendors seen and not seen in training.
7. Score the 10% hold-out file with the shipped model (``models/model.joblib``).

``--data FILE`` runs steps 1 to 6 on a separate generated file instead, for example
a larger set made with another seed (``models/evaluation/validation_<name>.json``).

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
from ml.clean import FEATURE_COLS, clean_training_data, load_raw
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


def _per_class(y_true, pred, labels: list[str]) -> dict:
    p, r, f, n = precision_recall_fscore_support(y_true, pred, labels=labels, zero_division=0)
    return {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "lines": int(n[i])}
            for i, c in enumerate(labels)}


def lookup_router(train_df: pd.DataFrame, valid: pd.DataFrame) -> pd.Series:
    """The vendor lookup used as a router: auto-label a line only if every training line of its vendor is Cat1."""
    pure_cat1 = train_df.groupby("Col1")["label"].agg(lambda s: bool((s == AUTO_CLASS).all()))
    return valid["Col1"].map(pure_cat1).fillna(False).astype(bool)


def routing_summary(y_true: pd.Series, auto: pd.Series) -> dict:
    """Auto-label share, Cat1 precision of the auto-labels, and non-Cat1 lines auto-labelled."""
    return {"automation_rate": float(auto.mean()),
            "auto_precision": float((y_true[auto] == AUTO_CLASS).mean()) if auto.any() else None,
            "auto_lines": int(auto.sum()),
            "non_cat1_auto": int((auto & (y_true != AUTO_CLASS)).sum())}


def vendor_lookup(train_df: pd.DataFrame, valid: pd.DataFrame) -> pd.Series:
    """Each vendor's most common training label; vendors never seen get Cat1."""
    lookup = train_df.groupby("Col1")["label"].agg(lambda s: s.value_counts().index[0])
    return valid["Col1"].map(lookup).fillna(AUTO_CLASS)


def _macro_f1_codes(y: np.ndarray, p: np.ndarray, k: int) -> float:
    """Macro-F1 over classes 0..k-1 from integer codes (fast enough to repeat thousands of times)."""
    tp = np.bincount(y[y == p], minlength=k)[:k]
    pred = np.bincount(p, minlength=k)[:k]
    true = np.bincount(y, minlength=k)[:k]
    denom = pred + true
    return float(np.mean(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)))


def bootstrap(y_true, model_pred, lookup_pred, labels: list[str], n: int = 2000, seed: int = 0) -> dict:
    """95% intervals for macro-F1 and Cat2 F1 of both methods, and for their difference.

    Lines are resampled with replacement and both methods are scored on the same resample
    (a paired bootstrap), so the interval for the difference is narrower than the two
    single intervals suggest. Labels outside ``labels`` (Cat5) count as a class nobody predicts.
    """
    code = {c: i for i, c in enumerate(labels)}
    other = len(labels)                                  # anything not scored, e.g. Cat5
    enc = lambda s: np.array([code.get(v, other) for v in s])   # noqa: E731
    y, m, b = enc(y_true), enc(model_pred), enc(lookup_pred)
    k, cat2 = len(labels), code.get("Cat2")
    rng = np.random.default_rng(seed)
    draws = {"model_macro_f1": [], "lookup_macro_f1": [], "difference": [], "model_cat2_f1": [], "lookup_cat2_f1": []}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        mf, lf = _macro_f1_codes(y[i], m[i], k), _macro_f1_codes(y[i], b[i], k)
        draws["model_macro_f1"].append(mf)
        draws["lookup_macro_f1"].append(lf)
        draws["difference"].append(mf - lf)
        if cat2 is not None:
            for name, pred in (("model_cat2_f1", m[i]), ("lookup_cat2_f1", b[i])):
                yt, pt = y[i] == cat2, pred == cat2
                denom = yt.sum() + pt.sum()
                draws[name].append(2 * (yt & pt).sum() / denom if denom else 0.0)
    return {name: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for name, v in draws.items() if v}


def segment(y_true: pd.Series, model_pred: pd.Series, lookup_pred: pd.Series, mask: pd.Series, labels: list[str]) -> dict:
    """Accuracy and macro-F1 of both methods on one slice of the validation lines."""
    y, m, b = y_true[mask], model_pred[mask], lookup_pred[mask]
    present = [c for c in labels if c in set(y) | set(m) | set(b)]
    return {
        "lines": int(mask.sum()),
        "model_accuracy": float(accuracy_score(y, m)) if len(y) else None,
        "lookup_accuracy": float(accuracy_score(y, b)) if len(y) else None,
        "model_macro_f1": float(f1_score(y, m, labels=present, average="macro", zero_division=0)) if len(y) else None,
        "lookup_macro_f1": float(f1_score(y, b, labels=present, average="macro", zero_division=0)) if len(y) else None,
        "class_lines": {c: int((y == c).sum()) for c in labels},
    }


def evaluate(cfg: dict, eval_model: str | Path | None = None, data: str | Path | None = None,
             out: str | Path | None = None, n_bootstrap: int = 2000) -> dict:
    """Time-split evaluation. ``data`` evaluates a separate generated file instead of the
    development set; that file has no hold-out, so the hold-out section is skipped."""
    model_dir = ROOT / cfg["paths"]["model_dir"]
    eval_model = Path(eval_model or model_dir / ("eval_model.joblib" if data is None else f"eval_{Path(data).stem}.joblib"))
    train(cfg, until_split=True, output=eval_model, data=data)
    bundle = load_bundle(str(eval_model))

    raw = load_raw(data) if data else load_development_raw(cfg)
    clean, _, _ = clean_training_data(raw)
    train_df, valid = time_split(clean, cfg["split"]["train_end_period"])
    results = predict_records(valid[FEATURE_COLS], bundle)
    model_pred = pd.Series([r["predicted_label"] for r in results], index=valid.index)
    baseline = vendor_lookup(train_df, valid)
    labels = scored_classes(list(bundle["classes"]), valid["label"], baseline, model_pred)
    seen = valid["Col1"].isin(set(train_df["Col1"]))

    summary = routing_metrics(valid["label"], results, labels)
    summary.update(
        data=str(Path(data).name) if data else cfg["paths"]["raw_data"],
        threshold=float(bundle["threshold"]),
        training_lines=int(len(train_df)),
        validation_months=[f"{valid['period'].min():%Y-%m}", f"{valid['period'].max():%Y-%m}"],
        unseen_vendor_share=float((~seen).mean()),
        vendor_lookup_macro_f1=float(f1_score(valid["label"], baseline, labels=labels, average="macro", zero_division=0)),
        vendor_lookup_per_class=_per_class(valid["label"], baseline, labels),
        bootstrap_95=bootstrap(valid["label"], model_pred, baseline, labels, n=n_bootstrap),
        routing_comparison={
            "model_and_policy": routing_summary(valid["label"], pd.Series([r["route"] == "auto" for r in results],
                                                                           index=valid.index)),
            "vendor_lookup_router": routing_summary(valid["label"], lookup_router(train_df, valid)),
        },
        segments={"seen_vendors": segment(valid["label"], model_pred, baseline, seen, labels),
                  "unseen_vendors": segment(valid["label"], model_pred, baseline, ~seen, labels)},
    )

    shipped = model_dir / "model.joblib"
    if data is None and shipped.exists() and (HOLDOUT_DIR / "holdout_labels.csv").exists():
        features = read_table(HOLDOUT_DIR / "holdout_features.xlsx")
        hold = pd.read_csv(HOLDOUT_DIR / "holdout_labels.csv")
        result = process_upload(features.assign(ClassificationLabel=hold["ClassificationLabel"].to_numpy()),
                                load_bundle(str(shipped)))
        summary["holdout"] = {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                              for k, v in result.metrics["duplicate copies removed"].items()}
        summary["holdout"]["rows_as_delivered"] = int(len(result.table))

    out = Path(out) if out else model_dir / "evaluation" / (
        "validation_summary.json" if data is None else f"validation_{Path(data).stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Time-split evaluation of the model and a vendor-lookup baseline.")
    parser.add_argument("--data", help="a separate generated file to evaluate instead of the development set")
    parser.add_argument("--bootstrap", type=int, default=2000, help="bootstrap resamples for the 95%% intervals")
    args = parser.parse_args()
    print(json.dumps(evaluate(yaml.safe_load(open(ROOT / "config.yaml")), data=args.data,
                              n_bootstrap=args.bootstrap), indent=2))
