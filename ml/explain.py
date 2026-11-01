"""Plain-language explanations for the logistic-regression model.

The model is linear: for each class it adds up ``weight × feature value`` and
the class with the highest total wins. So the reasons behind one prediction
can be read off exactly, with no approximation:

* **Global:** the largest weights per class are the tokens and fields that
  push hardest towards that class (``global_top_features``).
* **Local:** for one line,
  ``(weight[predicted] − weight[runner-up]) × (value − training average)``
  is how much each input moved the decision from the runner-up class to the
  predicted class, compared with a typical line. Inputs are added up per field
  (all vendor columns, all account columns, ...) and the largest totals are
  the reasons (``explain``).

Why "minus the training average": every line has exactly one document type
and most have the same one (NoDoc), so measured from zero that common value
would appear as a "reason" on almost every line. Measured from the average,
only what is unusual about this line counts. This is exactly what SHAP's
linear explainer computes for independent features, without the dependency.

Class names are always read from the outer ``model.classes_``: inside a
``CalibratedClassifierCV`` the logistic regression is fitted on encoded labels
(0, 1, 2, ...) in the same sorted order.

Temperature calibration divides every score by the same number, so it never
changes which feature matters most; the weights shown are the uncalibrated ones.
The confidence shown is the calibrated probability.

Used by the CLI, the API, the agent and the dashboard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from ml.features import CATEGORICAL_COLS, add_derived_columns

_CATEGORICAL_NAMES = {
    "vendor_id": "vendor",
    "Col6": "account",
    "account_group": "account group",
    "Col7": "document type",
    "ref_shape": "reference-ID shape",
}
_FIXED_NAMES = {
    "numeric__log_amount": "amount size",
    "numeric__ref_len": "reference-ID length",
    "flags__is_credit_note": "credit note (negative amount)",
    "flags__is_round_amount": "round amount (whole hundreds)",
    "flags__flag_description_missing": "description is empty",
}


# --------------------------------------------------------------------------
# Getting at the linear model
# --------------------------------------------------------------------------
def linear_pipeline(model) -> Pipeline:
    """Return the fitted features + LogisticRegression pipeline inside ``model``."""
    if isinstance(model, CalibratedClassifierCV):
        if len(model.calibrated_classifiers_) != 1:
            raise ValueError("expected ensemble=False (one underlying model)")
        return model.calibrated_classifiers_[0].estimator
    return model


def readable_feature_names(model) -> np.ndarray:
    """Map every model input column to a readable name, e.g. "description token Tok575"."""
    pipe = linear_pipeline(model)
    columns = pipe.named_steps["features"].named_steps["columns"]
    edges = np.expm1(columns.named_transformers_["amount_band"].bin_edges_[0])
    return np.array([_readable(n, edges) for n in columns.get_feature_names_out()])


def _readable(name: str, band_edges: np.ndarray) -> str:
    block, _, value = name.partition("__")
    if name in _FIXED_NAMES:
        return _FIXED_NAMES[name]
    if block == "vendor_tokens":
        return f"vendor-name token {value}"
    if block == "description":
        return f"description word pair '{value}'" if " " in value else f"description token {value}"
    if block == "amount_band":
        i = int(float(value.rsplit("_", 1)[1]))
        return f"amount band {i + 1} of {len(band_edges) - 1} ({band_edges[i]:,.0f} to {band_edges[i + 1]:,.0f})"
    if block == "categorical":
        for col in sorted(CATEGORICAL_COLS, key=len, reverse=True):
            if value.startswith(col + "_"):
                return f"{_CATEGORICAL_NAMES[col]} {value[len(col) + 1:]}"
    return name


# --------------------------------------------------------------------------
# Global explanation
# --------------------------------------------------------------------------
def global_top_features(model, n: int = 10) -> pd.DataFrame:
    """The ``n`` largest positive weights for each class, with readable names."""
    pipe = linear_pipeline(model)
    lr = pipe.named_steps["model"]
    names = readable_feature_names(model)
    rows = []
    for k, cls in enumerate(model.classes_):
        for rank, j in enumerate(np.argsort(-lr.coef_[k])[:n], start=1):
            rows.append({"class": cls, "rank": rank, "feature": names[j], "weight": lr.coef_[k, j]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Local explanation
# --------------------------------------------------------------------------
def feature_means(model, X: pd.DataFrame) -> np.ndarray:
    """Average value of every model input over the training rows ``X``: the explanation baseline."""
    features = linear_pipeline(model).named_steps["features"]
    return np.asarray(features.transform(X).mean(axis=0)).ravel()


def _field_groups(model) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Assign every model input to a field, so one-hot columns become one reason.

    Returns ``(group_id per input, group names, is_description_input)``.
    Description tokens and word pairs keep one group each: they are the most
    specific evidence a reviewer can look up.
    """
    columns = linear_pipeline(model).named_steps["features"].named_steps["columns"]
    keys = []
    for name in columns.get_feature_names_out():
        block, _, value = name.partition("__")
        if block == "vendor_tokens":
            keys.append("vendor")
        elif block == "categorical":
            col = next(c for c in sorted(CATEGORICAL_COLS, key=len, reverse=True) if value.startswith(c + "_"))
            keys.append(_CATEGORICAL_NAMES[col])
        elif block == "amount_band":
            keys.append("amount band")
        else:
            keys.append(name)
    group_names = list(dict.fromkeys(keys)) + [_OTHER_WORDS]
    index = {g: k for k, g in enumerate(group_names)}
    is_description = np.array([k.startswith("description__") for k in keys])
    return np.array([index[k] for k in keys]), group_names, is_description


_OTHER_WORDS = "description words this line lacks"
_ABSENT_FLAGS = {
    "flags__is_credit_note": "not a credit note",
    "flags__is_round_amount": "not a round amount",
    "flags__flag_description_missing": "description is present",
}


def explain(model, records: pd.DataFrame, baseline: np.ndarray, top_k: int = 5) -> list[dict]:
    """Explain each row of ``records`` (output of ``clean.standardize_records``).

    ``baseline`` is ``feature_means`` of the training data (saved in the model
    bundle as ``feature_means``). Pass zeros to measure from an empty line instead.

    Contributions are added up per field (vendor, account, account group,
    document type, reference-ID shape, amount band), so a field appears once,
    named by this line's own value. Each description word present on the line
    is its own reason; the effect of words the line lacks is one summed item.

    Returns one dict per row with the prediction, its calibrated confidence,
    the runner-up class, and the ``top_k`` reasons that moved the decision
    from the runner-up to the prediction, strongest first. ``against`` holds
    the strongest evidence for the runner-up instead.
    """
    pipe = linear_pipeline(model)
    lr = pipe.named_steps["model"]
    classes = list(model.classes_)
    X = pipe.named_steps["features"].transform(records).tocsr()
    proba = model.predict_proba(records)
    names = readable_feature_names(model)
    raw_names = pipe.named_steps["features"].named_steps["columns"].get_feature_names_out()
    group_id, group_names, is_description = _field_groups(model)
    other = len(group_names) - 1
    derived = add_derived_columns(records)

    out = []
    for i in range(X.shape[0]):
        order = np.argsort(-proba[i])
        pred, runner = order[0], order[1]
        x = X.getrow(i).toarray().ravel()
        contrib = (lr.coef_[pred] - lr.coef_[runner]) * (x - baseline)
        ids = np.where(is_description & (x == 0), other, group_id)
        totals = np.bincount(ids, weights=contrib, minlength=len(group_names))
        row = derived.iloc[i]

        def label(g):
            name = group_names[g]
            if name == _OTHER_WORDS:
                return name
            cols = np.flatnonzero(ids == g)
            if name in _CATEGORICAL_NAMES.values() or name == "amount band":
                return _field_label(name, row, names, cols, x)
            j = cols[0]
            if x[j] == 0 and raw_names[j] in _ABSENT_FLAGS:
                return _ABSENT_FLAGS[raw_names[j]]
            return _with_value(names[j], x[j] - baseline[j], row)

        rank = np.argsort(-totals)
        reasons = [{"feature": label(g), "contribution": float(totals[g])} for g in rank[:top_k] if totals[g] > 0]
        against = [{"feature": label(g), "contribution": float(totals[g])} for g in rank[::-1][:2] if totals[g] < 0]
        out.append({
            "predicted": classes[pred],
            "confidence": float(proba[i, pred]),
            "runner_up": classes[runner],
            "runner_up_confidence": float(proba[i, runner]),
            "reasons": reasons,
            "against": against,
        })
    return out


def _field_label(field: str, row: pd.Series, names: np.ndarray, cols: np.ndarray, x: np.ndarray) -> str:
    """Name a grouped field by this line's own value."""
    if field == "vendor":
        return f"vendor {row['Col1']}"
    if field == "account":
        return f"account {row['Col6']}"
    if field == "account group":
        return f"account group {row['account_group']}"
    if field == "document type":
        return f"document type {row['Col7']}"
    if field == "reference-ID shape":
        return f"reference-ID shape: {row['ref_shape']}"
    present = [names[j] for j in cols if x[j] > 0]   # amount band: the line's own band
    return present[0] if present else field


def _with_value(label: str, delta: float, row: pd.Series) -> str:
    """Phrase a numeric feature with this line's own value."""
    side = "above" if delta > 0 else "below"
    if label == "amount size":
        return f"amount size ({abs(row['Col3']):,.2f}, {side} the typical line)"
    if label == "reference-ID length":
        return f"reference-ID length ({int(row['ref_len'])} characters, {side} the typical line)"
    return label


def reason_text(explanation: dict) -> str:
    """Render one explanation as a short paragraph a reviewer can read."""
    e = explanation
    head = (f"Predicted {e['predicted']} ({e['confidence']:.0%} calibrated confidence; "
            f"next most likely {e['runner_up']} at {e['runner_up_confidence']:.0%}).")
    if not e["reasons"]:
        return head + " No single feature stands out; the prediction rests on the model's baseline."
    lines = [head, f"Main reasons it is {e['predicted']} rather than {e['runner_up']}:"]
    lines += [f"  • {r['feature']} (+{r['contribution']:.2f})" for r in e["reasons"]]
    if e["against"]:
        lines.append(f"Pointing to {e['runner_up']} instead: "
                     + "; ".join(f"{r['feature']} ({r['contribution']:.2f})" for r in e["against"]))
    return "\n".join(lines)
