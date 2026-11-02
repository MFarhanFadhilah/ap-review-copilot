"""Shared fixtures. Small hand-made data wherever possible; the prediction,
API, agent, batch and dashboard tests load the saved model (models/model.joblib),
trained on the committed synthetic sample."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _line(vendor, amount, label, ref="A100", desc="Tok1 Tok2", period="2023-05-31", account="Acc2 Acc9", doc="NoDoc"):
    return {"Col1": vendor, "Col2": ref, "Col3": amount, "Col4": desc, "Col5": pd.Timestamp(period),
            "Col6": account, "Col7": doc, "ClassificationLabel": label}


@pytest.fixture
def raw_small() -> pd.DataFrame:
    """Nine raw rows that exercise every cleaning step:

    * 2 identical copies of one line (label spelt "category_1")      -> kept once
    * 5 identical lines, 3 labelled Cat1 and 2 Cat2                    -> conflict, Cat1 wins 3 to 2
    * 1 normal line with the odd spelling "Category _3"
    * 1 line whose amount cannot be parsed                             -> dropped
    """
    rows = [
        _line("Ven10 Ven11", 100.0, "category_1"),
        _line("Ven10 Ven11", 100.0, "category_1"),
        *[_line("Ven20", 250.0, "Category_1", ref="B7")] * 3,
        *[_line("Ven20", 250.0, "Category 2", ref="B7")] * 2,
        _line("Ven30", 2450.0, "Category _3", ref="C9"),
        _line("Ven40", "not a number", "Category_1"),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def good_record() -> dict:
    """A valid Cat2 line from the synthetic sample's validation months.

    Picked by a fixed rule: the first validation-month line that is really Cat2,
    comes from a vendor whose every line is Cat2, and goes to review for the one
    reason "rare class (Cat2 predicted)".
    """
    return {"Col1": "Ven171", "Col2": "35462730", "Col3": 341.84, "Col4": "Tok1364",
            "Col5": "2023-09-30", "Col6": "Acc5 Acc182", "Col7": "NoDoc"}


@pytest.fixture
def cat1_record() -> dict:
    """A synthetic line the model labels Cat1 with high confidence.

    Picked by a fixed rule: the first validation-month Cat1 line from a vendor whose
    every line is Cat1, auto-labelled at >= 99% confidence, that still goes to review
    as a new vendor, as a credit note or with an unknown document type.
    """
    return {"Col1": "Ven111 Ven336", "Col2": "67807-27", "Col3": 4500.0, "Col4": "Tok784 Tok1364",
            "Col5": "2023-09-30", "Col6": "Acc7 Acc173", "Col7": "Doc3"}


@pytest.fixture(scope="session")
def bundle():
    path = ROOT / "models" / "model.joblib"
    if not path.exists():
        pytest.skip("models/model.joblib not found; run `python -m ml.split && python -m ml.train` first")
    from ml.predict import load_bundle
    return load_bundle(str(path))
