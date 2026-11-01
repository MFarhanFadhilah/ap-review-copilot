import numpy as np

from ml.policy import choose_threshold, route

THRESHOLD = 0.5


def _route_one(pred="Cat1", p=0.99, credit=False, seen=True, codes=None):
    out = route([pred], [p], [credit], [seen], THRESHOLD, [codes or []])
    return out["route"].iloc[0], out["review_reasons"].iloc[0]


def test_confident_cat1_from_known_vendor_is_auto_labelled():
    assert _route_one() == ("auto", "")


def test_rare_class_goes_to_review_however_confident():
    for cls in ["Cat2", "Cat3", "Cat4", "Cat6"]:
        r, reasons = _route_one(pred=cls, p=0.999)
        assert r == "review" and reasons == f"rare class ({cls} predicted)"


def test_credit_note_goes_to_review():
    assert _route_one(credit=True) == ("review", "credit note")


def test_unseen_vendor_goes_to_review():
    assert _route_one(seen=False) == ("review", "unseen vendor")


def test_low_confidence_cat1_goes_to_review():
    assert _route_one(p=0.49) == ("review", "low confidence")


def test_only_listed_validation_warnings_force_review():
    assert _route_one(codes=["unknown_doc_type"]) == ("review", "validation warning (unknown document type)")
    assert _route_one(codes=["ref_id_corrupted", "description_missing"]) == ("auto", "")


def test_reasons_accumulate():
    r, reasons = _route_one(pred="Cat2", credit=True, seen=False)
    assert r == "review"
    assert reasons.split("; ") == ["rare class (Cat2 predicted)", "credit note", "unseen vendor"]


def test_choose_threshold_meets_target_precision():
    # 10 Cat1 predictions ranked by confidence; the two least confident are wrong.
    y = np.array(["Cat1"] * 8 + ["Cat2"] * 2)
    p = np.linspace(0.99, 0.50, 10)
    t = choose_threshold(y, np.array(["Cat1"] * 10), p, np.ones(10, bool), target_precision=0.85)
    auto = p >= t
    assert (y[auto] == "Cat1").mean() >= 0.85
    assert auto.sum() == 9   # 8 of 9 correct = 0.889; all 10 would be 0.80


def test_choose_threshold_ignores_blocked_rows():
    y = np.array(["Cat1", "Cat2"])
    t = choose_threshold(y, np.array(["Cat1", "Cat1"]), np.array([0.9, 0.95]), np.array([True, False]), 0.98)
    assert t == 0.9


def test_missing_input_column_sends_every_line_to_review():
    out = route(["Cat1", "Cat1"], [0.99, 0.99], [False, False], [True, True], THRESHOLD, missing_columns=("Col4",))
    assert out["route"].tolist() == ["review", "review"]
    assert out["review_reasons"].tolist() == ["input file had no Col4"] * 2
