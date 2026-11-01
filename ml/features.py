"""Feature engineering, shared by training and inference.

The model is always ``Pipeline([("features", build_preprocessor()), ("model", ...)])``.
The preprocessor takes the output of ``clean.standardize_records`` and turns it
into one sparse matrix. Because the same object is fitted at training time and
saved with the model, serving can never compute a feature differently.

Every choice here came out of the exploration of the data:

* Col1 vendor: word tokens (similar vendor names share information) plus the
  full name as one category (the vendor itself is the strongest single signal).
* Col2 reference ID: shape only. Letters are randomised by the anonymisation
  and some IDs are corrupted by Excel, so the raw value would only memorise.
* Col3 amount: log of the absolute amount, a credit-note flag, a round-number
  flag, and amount bands (Cat2 is concentrated in small amounts, which a linear
  model cannot express with log-amount alone).
* Col4 description: TF-IDF on tokens and token pairs.
* Col5 period: deliberately NOT a feature. It is used for the time split only.
* Col6 account: full code plus account group (its first token).
* Col7 doc type: one-hot. Mostly redundant with account group, kept because it
  costs nothing.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer, MinMaxScaler, OneHotEncoder

# Tokens look like "Tok123". Keep them whole and case-sensitive.
TOKEN_PATTERN = r"\S+"

CATEGORICAL_COLS = ["vendor_id", "Col6", "account_group", "Col7", "ref_shape"]
NUMERIC_COLS = ["log_amount", "ref_len"]
FLAG_COLS = ["is_credit_note", "is_round_amount", "flag_description_missing"]
AMOUNT_BANDS = 10

_SCI_NOTATION = re.compile(r"^\d\.\d+[A-Z]\+\d+$")


# --------------------------------------------------------------------------
# Derived columns
# --------------------------------------------------------------------------
def ref_id_shape(value: str) -> str:
    """Describe a reference ID by its shape, never its content."""
    value = str(value)
    if value == "":
        return "missing"
    if _SCI_NOTATION.match(value):
        return "corrupted"
    if re.search(r"[\s\-/_.]", value):
        return "separator"
    if value.isdigit():
        return "numeric"
    return "alphanumeric"


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the engineered columns. Expects the output of ``standardize_records``."""
    out = df.copy()
    amount = out["Col3"].astype(float)
    out["vendor_id"] = out["Col1"]
    out["account_group"] = out["Col6"].str.split().str[0].fillna("")
    out["ref_shape"] = out["Col2"].map(ref_id_shape)
    out["ref_len"] = out["Col2"].str.len().astype(float)
    out["log_amount"] = np.log1p(amount.abs())
    out["is_credit_note"] = (amount < 0).astype(float)
    # Whole hundreds, excluding zero: fixed fees and contracts tend to be round.
    out["is_round_amount"] = ((amount != 0) & (amount.abs() % 100 == 0)).astype(float)
    out["flag_description_missing"] = out["Col4"].eq("").astype(float)
    return out


# --------------------------------------------------------------------------
# Preprocessor
# --------------------------------------------------------------------------
def build_preprocessor(min_df: int = 1, ngram_max: int = 2) -> Pipeline:
    """Return the unfitted feature pipeline.

    Every output column is non-negative, so the same features work for
    Naive Bayes, linear models and LightGBM. That keeps the model comparison
    fair: only the learner changes between rungs of the ladder.
    """
    text = dict(token_pattern=TOKEN_PATTERN, lowercase=False, min_df=min_df)
    columns = ColumnTransformer(
        [
            ("vendor_tokens", CountVectorizer(binary=True, **text), "Col1"),
            ("description", TfidfVectorizer(ngram_range=(1, ngram_max), sublinear_tf=True, **text), "Col4"),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            # clip=True keeps unseen extreme amounts inside [0, 1] at serving time.
            ("numeric", MinMaxScaler(clip=True), NUMERIC_COLS),
            (
                "amount_band",
                KBinsDiscretizer(n_bins=AMOUNT_BANDS, encode="onehot", strategy="quantile",
                                 quantile_method="averaged_inverted_cdf"),
                ["log_amount"],
            ),
            ("flags", "passthrough", FLAG_COLS),
        ],
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )
    return Pipeline([("derive", FunctionTransformer(add_derived_columns)), ("columns", columns)])


# --------------------------------------------------------------------------
# Time split
# --------------------------------------------------------------------------
def time_split(df: pd.DataFrame, train_end_period: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on periods up to and including ``train_end_period``; test on later ones."""
    cutoff = pd.Timestamp(train_end_period)
    return df[df["period"] <= cutoff].copy(), df[df["period"] > cutoff].copy()
