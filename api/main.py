"""Minimal HTTP API around ``ml.predict``.

Run::

    uvicorn api.main:app --reload

Endpoints:

* ``GET /health``: whether the model is loaded, its version and threshold.
* ``POST /predict``: one record (a JSON object) or a list of records. Always
  returns a list of results, in input order. Invalid input gets HTTP 422 with
  one plain-language message per problem, e.g.
  ``{"errors": [{"record": 1, "field": "Col3", "message": "Col3 (amount) is not a number"}]}``.

Field checks come from ``clean.validate_record``, the same function the CLI
and the training pipeline use, so the API cannot accept a record the model
would reject elsewhere.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from ml.clean import validate_record
from ml.predict import DEFAULT_MODEL, load_bundle, predict_records

MAX_RECORDS = 1000
STATE: dict[str, Any] = {}


class Record(BaseModel):
    """One transaction line, in the same seven columns as the training file."""
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "Col1": "Ven171", "Col2": "35462730", "Col3": 341.84, "Col4": "Tok1364",
        "Col5": "2023-09-30", "Col6": "Acc5 Acc182", "Col7": "NoDoc"}})   # a synthetic Cat2 line

    Col1: str | None = Field(None, description="vendor (tokenised)")
    Col2: str | int | float | None = Field(None, description="reference ID")
    Col3: float | str | None = Field(None, description="amount")
    Col4: str | None = Field(None, description="description (tokenised)")
    Col5: str | None = Field(None, description="posting period, any date format")
    Col6: str | None = Field(None, description="account (tokenised)")
    Col7: str | None = Field(None, description="document type: NoDoc, Doc1, Doc2 or Doc3")

    @model_validator(mode="after")
    def _business_rules(self):
        errors = validate_record(self.model_dump())["errors"]
        if errors:
            raise ValueError("; ".join(errors))
        return self


RECORDS = TypeAdapter(list[Record])


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = os.environ.get("MODEL_PATH")
    try:
        STATE["bundle"] = load_bundle(path)
        STATE["error"] = None
    except Exception as exc:   # report it on /health instead of refusing to start
        STATE["bundle"], STATE["error"] = None, f"{type(exc).__name__}: {exc}"
    STATE["path"] = path or str(DEFAULT_MODEL)
    yield
    STATE.clear()


app = FastAPI(title="Transaction line classifier", version="1.0", lifespan=lifespan)


def _plain(errors: list[dict]) -> list[dict]:
    """Turn Pydantic's error list into one plain-language message per problem."""
    out = []
    for e in errors:
        loc = [p for p in e.get("loc", []) if p != "body"]
        record = loc[0] + 1 if loc and isinstance(loc[0], int) else 1
        field = next((p for p in loc if isinstance(p, str) and p.startswith("Col")), None)
        if e["type"] == "extra_forbidden":
            out.append({"record": record, "field": str(loc[-1]),
                        "message": f"{loc[-1]} is not an expected field (send only Col1 to Col7)"})
        elif e["type"] == "value_error":
            # validate_record's messages, joined with "; " by Record._business_rules: one entry each.
            for part in str(e.get("ctx", {}).get("error", e["msg"])).split("; "):
                first = part.split(" ")[0]
                out.append({"record": record, "field": first if first.startswith("Col") else field, "message": part})
        elif e["type"] in {"string_type", "float_type", "float_parsing", "int_type"}:
            out.append({"record": record, "field": field, "message": f"{field} has the wrong type ({e['msg'].lower()})"})
        else:
            out.append({"record": record, "field": field, "message": e["msg"]})
    return out


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"errors": _plain(exc.errors())})


@app.get("/health")
def health() -> dict:
    bundle = STATE.get("bundle")
    if bundle is None:
        return {"status": "error", "model_loaded": False, "model_path": STATE.get("path"), "error": STATE.get("error")}
    meta = bundle["metadata"]
    return {
        "status": "ok",
        "model_loaded": True,
        "model_version": meta.get("model_version"),
        "trained_at": meta.get("trained_at"),
        "threshold": bundle["threshold"],
        "classes": bundle["classes"],
    }


@app.post("/predict")
def predict(payload: Any = Body(..., examples=[Record.model_config["json_schema_extra"]["example"]])) -> list[dict]:
    """Score one record or a list of records."""
    if STATE.get("bundle") is None:
        raise HTTPException(status_code=503, detail="model not loaded; see /health")
    items = [payload] if isinstance(payload, dict) else payload
    if not isinstance(items, list) or not items:
        raise RequestValidationError([{"type": "value_error", "loc": ("body",), "msg": "send one record (a JSON "
                                       "object) or a non-empty list of records", "ctx": {"error": "send one record "
                                       "(a JSON object) or a non-empty list of records"}}])
    if len(items) > MAX_RECORDS:
        raise RequestValidationError([{"type": "value_error", "loc": ("body",), "msg": "too many records",
                                       "ctx": {"error": f"send at most {MAX_RECORDS} records per request"}}])
    try:
        records = RECORDS.validate_python(items)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from None
    return predict_records([r.model_dump() for r in records], STATE["bundle"])
