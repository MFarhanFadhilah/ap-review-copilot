"""Exploration helpers used by the dashboard.

* ``add_eda_columns``: the binned and derived fields the exploration looks at
  (amount band, reference-ID shape, account group, month). The ID shape and
  account group come from ``features.py``, so the analysis describes exactly
  what the model sees.
* ``minority_table``: per value of a field, the class counts and the share of
  lines that are not Cat1.
* ``cramers_v`` / ``association_matrix``: bias-corrected Cramér's V between
  every pair of fields, including the label.
* ``amount_outliers``: lines whose absolute amount is extreme on a log scale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from ml.features import ref_id_shape

ASSOCIATION_FIELDS = {
    "vendor (Col1)": "Col1", "account (Col6)": "Col6", "account group": "account_group",
    "doc type (Col7)": "Col7", "amount band": "amount_band", "ref-ID shape": "ref_shape",
    "month": "month", "credit note": "flag_credit_note", "label": "label",
}


def add_eda_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the exploration fields to a cleaned frame (output of ``clean_training_data``)."""
    out = df.copy()
    out["amount_band"] = pd.qcut(out["Col3"].abs(), 10, duplicates="drop")
    out["ref_shape"] = out["Col2"].map(ref_id_shape)
    out["account_group"] = out["Col6"].str.split().str[0]
    out["month"] = out["period"].dt.strftime("%Y-%m")
    return out


def minority_table(df: pd.DataFrame, col: str, min_rows: int = 1) -> pd.DataFrame:
    t = pd.crosstab(df[col], df["label"])
    t["rows"] = t.sum(axis=1)
    t["minority_%"] = (100 * (1 - t["Cat1"] / t["rows"])).round(1)
    return t[t.rows >= min_rows].sort_values("minority_%", ascending=False)


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Bias-corrected Cramér's V (Bergsma 2013); the correction matters for high-cardinality fields like vendor."""
    t = pd.crosstab(a, b)
    if min(t.shape) < 2:
        return np.nan
    chi2 = chi2_contingency(t, correction=False)[0]
    n = t.values.sum()
    r, k = t.shape
    phi2 = max(0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2 / max(1e-9, min(kc - 1, rc - 1))))


def association_matrix(df: pd.DataFrame, fields: dict = ASSOCIATION_FIELDS) -> pd.DataFrame:
    """Cramér's V for every pair of fields; ``df`` must have the ``add_eda_columns`` fields."""
    names = list(fields)
    V = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            V.loc[a, b] = V.loc[b, a] = cramers_v(df[fields[a]].astype(str), df[fields[b]].astype(str))
    return V


def amount_outliers(amounts: pd.Series, k: float = 1.5) -> pd.Series:
    """True where log10(|amount|) lies more than ``k`` IQRs outside the middle half (zero amounts excluded)."""
    log_amount = np.log10(amounts.abs().where(amounts != 0))
    q1, q3 = log_amount.quantile([0.25, 0.75])
    spread = q3 - q1
    return (log_amount < q1 - k * spread) | (log_amount > q3 + k * spread)
