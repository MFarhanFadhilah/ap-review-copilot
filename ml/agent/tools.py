"""The agent's tools. Each one is a thin wrapper over code that already exists:

* ``validate_record``       -> ``clean.validate_record``
* ``predict_and_route``     -> ``predict.predict_records`` (model + ``policy.route``)
* ``explain_prediction``    -> ``explain.explain``
* ``find_similar_invoices`` -> cosine similarity in the model's own feature space
* ``vendor_history``        -> label counts from the lines the model was trained on
* ``check_suspicious_text`` -> a fixed regex for instruction-like text in the line's fields
* ``value_frequencies``     -> how common each categorical value is among the training lines

None of them writes anything or changes a decision. ``TOOL_SPECS`` describes
them in the JSON-schema form an LLM tool call would use, and every call the
agent makes is recorded with its input and output for the transcript.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from ml.clean import FEATURE_COLS, standardize_records, validate_record
from ml.explain import explain, linear_pipeline
from ml.features import add_derived_columns
from ml.predict import predict_records

# Instruction-like text in an invoice field. Deterministic, so whether a note may
# mention "instructions" never depends on what the LLM thinks it saw.
SUSPICIOUS_TEXT = re.compile(
    r"\b(ignore|disregard|instructions?|system|prompt|assistant|override|you are now|"
    r"(?:label|classify|mark|categori[sz]e) (?:this|it|the line)(?: line)? (?:as|to)|label this)\b",
    re.IGNORECASE)
TEXT_FIELDS = ["Col1", "Col2", "Col4", "Col6", "Col7"]
FREQUENCY_FIELDS = {"vendor": "Col1", "account": "Col6", "account group": "account_group",
                    "document type": "Col7", "reference-ID shape": "ref_shape"}


def suspicious_text(record: dict) -> dict:
    """``flag`` is True when any text field contains instruction-like words; ``fields`` shows where."""
    hits = {c: str(record.get(c)) for c in TEXT_FIELDS
            if record.get(c) is not None and SUSPICIOUS_TEXT.search(str(record.get(c)))}
    return {"flag": bool(hits), "fields": hits}

_RECORD_SCHEMA = {
    "type": "object",
    "description": "One transaction line with columns Col1 to Col7",
    "properties": {c: {"type": ["string", "number", "null"]} for c in FEATURE_COLS},
    "required": ["Col1", "Col3", "Col5", "Col6"],
}
TOOL_SPECS = [
    {"name": "validate_record", "description": "Check a line for missing or malformed fields and data-quality warnings.",
     "input_schema": {"type": "object", "properties": {"record": _RECORD_SCHEMA}, "required": ["record"]}},
    {"name": "predict_and_route", "description": "Model label, calibrated confidence, and the policy route (auto or review) with reasons.",
     "input_schema": {"type": "object", "properties": {"record": _RECORD_SCHEMA}, "required": ["record"]}},
    {"name": "explain_prediction", "description": "Top reasons for the predicted label rather than the runner-up, in plain language.",
     "input_schema": {"type": "object", "properties": {"record": _RECORD_SCHEMA}, "required": ["record"]}},
    {"name": "find_similar_invoices",
     "description": "The most similar training lines (cosine similarity on the model's features) with their labels.",
     "input_schema": {"type": "object", "properties": {"record": _RECORD_SCHEMA, "k": {"type": "integer"}}, "required": ["record"]}},
    {"name": "vendor_history", "description": "How the vendor's past lines were labelled, or 'new vendor'.",
     "input_schema": {"type": "object", "properties": {"vendor": {"type": "string"}}, "required": ["vendor"]}},
    {"name": "check_suspicious_text", "description": "Whether any field contains instruction-like text (fixed regex).",
     "input_schema": {"type": "object", "properties": {"record": _RECORD_SCHEMA}, "required": ["record"]}},
    {"name": "value_frequencies", "description": "Share of training lines with each categorical value on this line.",
     "input_schema": {"type": "object", "properties": {"record": _RECORD_SCHEMA}, "required": ["record"]}},
]


class AgentTools:
    """Holds the model bundle and the similarity index built from its training lines."""

    def __init__(self, bundle: dict, novelty_percentile: float = 1.0, similar_k: int = 5):
        self.bundle = bundle
        self.similar_k = similar_k
        self.reference = bundle["reference_lines"]
        self._features = linear_pipeline(bundle["model"]).named_steps["features"]
        self._index = normalize(self._features.transform(self.reference).tocsr())
        self.novelty_threshold = float(np.percentile(self._nearest_to_itself(), novelty_percentile))
        self.novelty_percentile = novelty_percentile
        derived = add_derived_columns(standardize_records(self.reference[FEATURE_COLS]))
        self._shares = {name: derived[col].value_counts(normalize=True) for name, col in FREQUENCY_FIELDS.items()}

    # ---------------------------------------------------------------- tools
    def validate_record(self, record: dict) -> dict:
        return validate_record(record)

    def predict_and_route(self, record: dict, missing_columns=()) -> dict:
        return predict_records([record], self.bundle, missing_columns=missing_columns)[0]

    def explain_prediction(self, record: dict, top_k: int = 5) -> dict:
        x = standardize_records(pd.DataFrame([record], columns=FEATURE_COLS))
        return explain(self.bundle["model"], x, self.bundle["feature_means"], top_k=top_k)[0]

    def find_similar_invoices(self, record: dict, k: int | None = None) -> dict:
        k = k or self.similar_k
        sims = self._similarities(record)
        top = np.argsort(-sims)[:k]
        lines = self.reference.iloc[top]
        return {
            "nearest_similarity": float(sims[top[0]]),
            "novel": bool(sims[top[0]] < self.novelty_threshold),
            "novelty_threshold": self.novelty_threshold,
            "similar": [
                {"similarity": round(float(sims[j]), 3), "label": row["label"], "vendor": row["Col1"],
                 "amount": float(row["Col3"]), "description": row["Col4"], "account": row["Col6"],
                 "period": f"{pd.Timestamp(row['period']):%Y-%m}"}
                for j, (_, row) in zip(top, lines.iterrows())
            ],
        }

    def vendor_history(self, vendor: str) -> dict:
        past = self.reference[self.reference["Col1"] == str(vendor).strip()]
        if past.empty:
            return {"vendor": vendor, "status": "new vendor", "lines": 0, "label_counts": {}}
        return {
            "vendor": vendor, "status": "known vendor", "lines": int(len(past)),
            "label_counts": {k: int(v) for k, v in past["label"].value_counts().items()},
            "first_period": f"{past['period'].min():%Y-%m}", "last_period": f"{past['period'].max():%Y-%m}",
        }

    def check_suspicious_text(self, record: dict) -> dict:
        return suspicious_text(record)

    def value_frequencies(self, record: dict) -> dict:
        """For each categorical field: this line's value and its share of the training lines (0 if never seen)."""
        derived = add_derived_columns(standardize_records(pd.DataFrame([record], columns=FEATURE_COLS))).iloc[0]
        out = {}
        for name, col in FREQUENCY_FIELDS.items():
            value = derived[col]
            share = float(self._shares[name].get(value, 0.0))
            out[name] = {"value": value, "share": share, "text": f"{value}: {share:.1%} of training lines"}
        return out

    def call(self, name: str, **kwargs):
        """Dispatch by tool name, so every call goes through one place."""
        if name not in {spec["name"] for spec in TOOL_SPECS}:
            raise ValueError(f"unknown tool {name!r}")
        return getattr(self, name)(**kwargs)

    # ------------------------------------------------------------- internals
    def _similarities(self, record: dict) -> np.ndarray:
        x = standardize_records(pd.DataFrame([record], columns=FEATURE_COLS))
        q = normalize(self._features.transform(x).tocsr())
        return (self._index @ q.T).toarray().ravel()

    def _nearest_to_itself(self, chunk: int = 500) -> np.ndarray:
        """For every training line, the similarity of its nearest *other* training line."""
        n = self._index.shape[0]
        best = np.empty(n)
        for start in range(0, n, chunk):
            block = (self._index[start:start + chunk] @ self._index.T).toarray()
            block[np.arange(block.shape[0]), np.arange(start, start + block.shape[0])] = -1.0
            best[start:start + block.shape[0]] = block.max(axis=1)
        return best
