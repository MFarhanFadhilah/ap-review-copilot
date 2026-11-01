import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client(bundle):
    with TestClient(app) as c:
        yield c


def test_health(client, bundle):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["threshold"] == bundle["threshold"]
    assert body["model_version"] == bundle["metadata"]["model_version"]


def test_predict_one_record_returns_a_list_of_one(client, good_record):
    r = client.post("/predict", json=good_record)
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1 and body[0]["predicted_label"] == "Cat2" and body[0]["route"] == "review"


def test_predict_a_list_keeps_order(client, good_record, cat1_record):
    body = client.post("/predict", json=[cat1_record, good_record]).json()
    assert [b["predicted_label"] for b in body] == ["Cat1", "Cat2"]


@pytest.mark.parametrize("change, field, message", [
    ({"Col3": "abc"}, "Col3", "Col3 (amount) is not a number"),
    ({"Col1": None}, "Col1", "Col1 is required but missing"),
    ({"Col5": "not a date"}, "Col5", "Col5 (period) is not a valid date"),
    ({"Col1": 5}, "Col1", "Col1 has the wrong type (input should be a valid string)"),
    ({"Extra": 1}, "Extra", "Extra is not an expected field (send only Col1 to Col7)"),
])
def test_invalid_input_gets_422_with_plain_message(client, good_record, change, field, message):
    r = client.post("/predict", json={**good_record, **change})
    assert r.status_code == 422
    assert r.json() == {"errors": [{"record": 1, "field": field, "message": message}]}


def test_error_names_the_record_in_a_list(client, good_record):
    r = client.post("/predict", json=[good_record, {**good_record, "Col3": None}])
    assert r.status_code == 422
    assert r.json()["errors"] == [{"record": 2, "field": "Col3", "message": "Col3 is required but missing"}]


@pytest.mark.parametrize("payload", [[], "hello", 42])
def test_payload_that_is_not_records_gets_422(client, payload):
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
    assert "send one record" in r.json()["errors"][0]["message"]
