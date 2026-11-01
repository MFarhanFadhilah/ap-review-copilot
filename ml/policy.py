"""Routing policy: which predictions are applied automatically and which go to a person.

The model suggests a label for every line. This module decides what happens to
that suggestion. The rule is deliberately conservative:

* Auto-label only when the model predicts Cat1 with calibrated confidence at
  or above a threshold.
* Everything else goes to review: any Cat2-Cat6 prediction (rare and costly
  categories always get a human check), credit notes (the only Cat5 examples
  are credit notes and the model has never seen Cat5), vendors with no
  labelled history, low-confidence Cat1 predictions, and lines with a
  validation warning that means the model is seeing something it never saw
  in training (``REVIEW_WARNINGS``).

Other validation warnings (a corrupted reference ID, an empty description)
are reported to the reviewer but do not change the route: the training data
contains such lines, so the model has learned how to handle them.

The same functions are used in ``evaluate.py`` (to measure the policy), in
``train.py`` (to set the threshold) and at serving time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

AUTO_CLASS = "Cat1"

# Validation warnings (codes from clean.validate_record) that force a review.
REVIEW_WARNINGS = {"unknown_doc_type": "unknown document type"}


def choose_threshold(y_true, pred, p_auto, eligible, target_precision: float) -> float:
    """Return the lowest confidence threshold whose auto-labelled set meets the target precision.

    Candidates are rows that are ``eligible`` (not blocked by a business rule)
    and predicted as ``AUTO_CLASS``. They are ranked by confidence, and the
    threshold is set at the lowest confidence where the precision of
    everything at or above it is still at least ``target_precision``.
    Must be called on out-of-fold predictions, never on the rows the model was fitted on.
    """
    y_true, pred, p_auto, eligible = map(np.asarray, (y_true, pred, p_auto, eligible))
    cand = eligible & (pred == AUTO_CLASS)
    order = np.argsort(-p_auto[cand], kind="stable")
    correct = (y_true[cand] == AUTO_CLASS)[order]
    precision = np.cumsum(correct) / np.arange(1, len(correct) + 1)
    passing = np.flatnonzero(precision >= target_precision)
    if len(passing) == 0:
        return 1.01  # nothing can be automated safely
    return float(p_auto[cand][order][passing.max()])


def route(pred, p_auto, is_credit_note, vendor_seen, threshold: float, warning_codes=None,
          missing_columns=()) -> pd.DataFrame:
    """Return one row per line with ``route`` ("auto" or "review") and the reasons for review.

    ``warning_codes`` is optional: one list of ``clean.validate_record`` codes per line.
    ``missing_columns`` lists optional input columns absent from the whole file
    (e.g. Col4): the model then sees those fields as blank on every line, so
    every line goes to review with the reason "input file had no ColN".
    """
    pred, p_auto = np.asarray(pred), np.asarray(p_auto)
    is_credit_note, vendor_seen = np.asarray(is_credit_note, bool), np.asarray(vendor_seen, bool)
    if warning_codes is None:
        warning_codes = [[] for _ in range(len(pred))]

    reasons = [[] for _ in range(len(pred))]
    for i in range(len(pred)):
        if pred[i] != AUTO_CLASS:
            reasons[i].append(f"rare class ({pred[i]} predicted)")
        elif p_auto[i] < threshold:
            reasons[i].append("low confidence")
        if is_credit_note[i]:
            reasons[i].append("credit note")
        if not vendor_seen[i]:
            reasons[i].append("unseen vendor")
        for code in warning_codes[i]:
            if code in REVIEW_WARNINGS:
                reasons[i].append(f"validation warning ({REVIEW_WARNINGS[code]})")
        reasons[i].extend(f"input file had no {col}" for col in missing_columns)
    return pd.DataFrame({
        "route": ["review" if r else "auto" for r in reasons],
        "review_reasons": ["; ".join(r) for r in reasons],
    })
