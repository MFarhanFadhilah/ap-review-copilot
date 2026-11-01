"""The review agent: validate, predict and route, gather evidence, write a note, log it.

The flow is fixed in code, not chosen by the LLM:

1. ``validate_record``
2. ``predict_and_route`` (the model and ``policy.py``; this is the decision)
3. only if the line goes to review: ``explain_prediction``, ``find_similar_invoices``,
   ``vendor_history``; the novelty check runs on every scorable line
4. a reviewer note: what the line is, why it was flagged, what to check
5. one audit record in ``logs/agent_audit.jsonl``

Guardrails, enforced here and not left to the prompt:

* **The decision is never the LLM's.** Label, confidence and route are copied
  from step 2 after the note is written; whatever label or route the LLM
  returned is overwritten, and an attempt to change them is logged.
* **The note may not introduce a category.** A note naming a category that is
  not in the evidence is discarded and the template note is used instead.
* **Invoice text is data.** It is passed inside tagged blocks that the system
  prompt says are untrusted, and the LLM has no tool that can change anything.
* **Any LLM failure falls back to the template**, so a decision is always
  produced and logged. The result, the audit record (``mode`` =
  ``template_fallback``, ``fallback_reason``) and a Python ``logging`` warning
  all say why.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

from ml.clean import FEATURE_COLS
from ml.predict import format_confidence

from .notes import NoteFallback, TemplateNoteWriter, _suspicious_note, template_explanation, template_note
from .tools import AgentTools

log = logging.getLogger("ml.agent")
_CATEGORY = re.compile(r"\bCat\s?\d\b")
# Sentences about instruction-like text; allowed only when the code's suspicious_text flag is set.
_INSTRUCTION_TALK = re.compile(r"instruction|injection|treat(ed)? (it )?with suspicion|suspicious", re.IGNORECASE)
# A decimal that is not a percentage, or the word "weight": the explanation must not quote raw weights.
_RAW_NUMBER = re.compile(r"(?<![\w.])[+\-−]?\d*\.\d+(?!\d*\s*%)|\bweights?\b", re.IGNORECASE)
EXPLANATION_MAX_WORDS = 170   # "about 150 words", with a little room


def input_hash(record: dict) -> str:
    canonical = json.dumps({c: record.get(c) for c in FEATURE_COLS}, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


class ReviewAgent:
    def __init__(self, bundle: dict, note_writer=None, audit_log: str | Path = "logs/agent_audit.jsonl",
                 novelty_percentile: float = 1.0, similar_k: int = 5):
        self.tools = AgentTools(bundle, novelty_percentile, similar_k)
        self.note_writer = note_writer or TemplateNoteWriter()
        self.audit_log = Path(audit_log)
        self.model_version = bundle["metadata"].get("model_version")

    # ------------------------------------------------------------------ run
    def run(self, record: dict, missing_columns=()) -> dict:
        steps = []

        def call(name, **kwargs):
            out = self.tools.call(name, **kwargs)
            steps.append({"tool": name, "output": out})
            return out

        record = {c: record.get(c) for c in FEATURE_COLS}
        check = call("validate_record", record=record)
        decision_raw = call("predict_and_route", record=record, missing_columns=tuple(missing_columns))
        decision = {
            "label": decision_raw["predicted_label"],
            "confidence": decision_raw["confidence"],
            "confidence_band": decision_raw["confidence_band"],
            "runner_up": decision_raw["runner_up_label"],
            "route": decision_raw["route"],
            "review_reasons": decision_raw["review_reasons"],
            "top_reasons": decision_raw["top_reasons"],
            "validation_warnings": check["warnings"],
            "validation_errors": check["errors"],
        }
        facts = {"record": record, "decision": decision, "missing_columns": list(missing_columns),
                 "suspicious": call("check_suspicious_text", record=record)}

        if decision["label"] is not None:
            similar = call("find_similar_invoices", record=record)
            facts["novelty"] = {k: similar[k] for k in ("novel", "nearest_similarity", "novelty_threshold")}
            if decision["route"] == "review":
                facts["explanation"] = call("explain_prediction", record=record)
                facts["similar"] = similar
                facts["vendor_history"] = call("vendor_history", vendor=record["Col1"])
                facts["frequencies"] = call("value_frequencies", record=record)

        note, guard = None, {"mode": self.note_writer.mode, "llm_label_overridden": False, "fallback_reason": None}
        if decision["route"] == "review":
            note, guard = self._write_note(facts, guard)

        result = {
            **{k: decision[k] for k in ("label", "confidence", "confidence_band", "route", "review_reasons",
                                        "top_reasons", "validation_warnings", "validation_errors")},
            "novelty": facts.get("novelty"),
            "note": note,
            "note_text": render_note(note, decision) if note else None,
            "mode": guard["mode"],
            "guardrail": guard,
            "steps": steps,
        }
        self._audit(record, result)
        return result

    # ----------------------------------------------------------- guardrails
    # What each kind of text is called, how it is written, its offline template and its text fields.
    _KINDS = {
        "note": ("write", template_note, ("summary",), ("why_flagged", "what_to_check")),
        "explanation": ("explain", template_explanation, ("verdict", "runner_up"), ("reasons", "checks")),
    }

    def _write_note(self, facts: dict, guard: dict, kind: str = "note") -> tuple[dict, dict]:
        """Ask the writer for a note or explanation, then enforce the guardrails on it."""
        decision = facts["decision"]
        method, _, text_fields, list_fields = self._KINDS[kind]
        try:
            note = getattr(self.note_writer, method)(facts)
        except NoteFallback as exc:
            return self._fallback(facts, guard, str(exc), kind)
        except Exception as exc:   # a writer bug must never block the decision either
            return self._fallback(facts, guard, f"{kind} writer failed ({type(exc).__name__}: {exc})", kind)

        if note.get("label") != decision["label"] or note.get("route") != decision["route"]:
            guard["llm_label_overridden"] = True
            guard["llm_returned"] = {"label": note.get("label"), "route": note.get("route")}

        allowed = {decision["label"], decision["runner_up"]}
        allowed |= {s["label"] for s in facts.get("similar", {}).get("similar", [])}
        allowed |= set(facts.get("vendor_history", {}).get("label_counts", {}))
        if "credit note" in decision["review_reasons"]:
            allowed.add("Cat5")   # the note should say a credit note may be Cat5, which the model cannot predict
        text = " ".join([*(note.get(f, "") for f in text_fields), *(x for f in list_fields for x in note.get(f, []))])
        named = {m.replace(" ", "") for m in _CATEGORY.findall(text)}
        if named - allowed:
            return self._fallback(facts, guard, f"LLM {kind} named categories not in the evidence: {sorted(named - allowed)}", kind)

        # Instruction-like text: mention it only when the code's flag says so, and always when it does.
        flagged = bool((facts.get("suspicious") or {}).get("flag"))
        check_field = list_fields[-1]
        if not flagged:
            stripped = 0
            for f in text_fields:
                sentences = re.split(r"(?<=[.!?])\s+", note.get(f, ""))
                kept = [x for x in sentences if not _INSTRUCTION_TALK.search(x)]
                stripped += len(sentences) - len(kept)
                note[f] = " ".join(kept)
            for f in list_fields:
                kept = [x for x in note.get(f, []) if not _INSTRUCTION_TALK.search(x)]
                stripped += len(note.get(f, [])) - len(kept)
                note[f] = kept
            if stripped:
                guard["stripped_instruction_sentences"] = stripped
        elif not any(_INSTRUCTION_TALK.search(x) for x in note.get(check_field, [])):
            note[check_field] = [_suspicious_note(facts), *note.get(check_field, [])]

        if kind == "explanation":
            body = " ".join([*(note.get(f, "") for f in text_fields), *(x for f in list_fields for x in note.get(f, []))])
            words = len(body.split())
            if words > EXPLANATION_MAX_WORDS:
                return self._fallback(facts, guard, f"LLM explanation too long ({words} words)", kind)
            if _RAW_NUMBER.search(body):
                return self._fallback(facts, guard, "LLM explanation quoted a raw weight or score", kind)

        if guard["llm_label_overridden"]:
            log.warning("LLM returned label/route %s; overwritten with the model's %s/%s",
                        guard["llm_returned"], decision["label"], decision["route"])
        note["label"], note["route"] = decision["label"], decision["route"]   # the decision always wins
        return note, guard

    def _fallback(self, facts: dict, guard: dict, reason: str, kind: str = "note") -> tuple[dict, dict]:
        """Use the template, and say why in the result, the audit log and the Python log."""
        log.warning("LLM %s not used, template %s written instead: %s", kind, kind, reason)
        guard.update(mode="template_fallback", fallback_reason=reason)
        return self._KINDS[kind][1](facts), guard

    # -------------------------------------------------------------- explain
    def explain_line(self, record: dict, missing_columns=()) -> dict:
        """Explain any line, auto or review: top feature contributions, similar lines, vendor history.

        The label and route come from ``predict_and_route`` and are never
        changed; the writer (template or LLM) only explains them.
        """
        steps = []

        def call(name, **kwargs):
            out = self.tools.call(name, **kwargs)
            steps.append({"tool": name, "output": out})
            return out

        record = {c: record.get(c) for c in FEATURE_COLS}
        check = call("validate_record", record=record)
        raw = call("predict_and_route", record=record, missing_columns=tuple(missing_columns))
        decision = {"label": raw["predicted_label"], "confidence": raw["confidence"],
                    "confidence_band": raw["confidence_band"], "runner_up": raw["runner_up_label"],
                    "route": raw["route"], "review_reasons": raw["review_reasons"],
                    "top_reasons": raw["top_reasons"], "validation_warnings": check["warnings"],
                    "validation_errors": check["errors"]}
        guard = {"mode": self.note_writer.mode, "llm_label_overridden": False, "fallback_reason": None}
        result = {"label": decision["label"], "route": decision["route"], "review_reasons": decision["review_reasons"],
                  "confidence": decision["confidence"], "explanation": None, "text": None, "novelty": None,
                  "mode": guard["mode"], "guardrail": guard, "steps": steps}
        if decision["label"] is None:
            result["text"] = "This line could not be scored: " + "; ".join(check["errors"]) + "."
            self._audit(record, {**result, "note_text": result["text"]}, kind="explanation")
            return result

        facts = {"record": record, "decision": decision, "missing_columns": list(missing_columns),
                 "explanation": call("explain_prediction", record=record),
                 "similar": call("find_similar_invoices", record=record),
                 "vendor_history": call("vendor_history", vendor=record["Col1"]),
                 "frequencies": call("value_frequencies", record=record),
                 "suspicious": call("check_suspicious_text", record=record)}
        decision["runner_up_confidence"] = facts["explanation"]["runner_up_confidence"]
        explanation, guard = self._write_note(facts, guard, kind="explanation")
        result.update(explanation=explanation, text=render_explanation(explanation, decision), mode=guard["mode"],
                      guardrail=guard, novelty={k: facts["similar"][k] for k in ("novel", "nearest_similarity", "novelty_threshold")},
                      evidence=facts)
        self._audit(record, {**result, "note_text": result["text"]}, kind="explanation")
        return result

    # ---------------------------------------------------------------- audit
    def _audit(self, record: dict, result: dict, kind: str = "decision") -> None:
        self.audit_log.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "kind": kind,
            "input_hash": input_hash(record),
            "model_version": self.model_version,
            "mode": result["mode"],
            "label": result["label"],
            "confidence": result["confidence"],
            "route": result["route"],
            "review_reasons": result["review_reasons"],
            "novel": (result["novelty"] or {}).get("novel"),
            "note": result["note_text"],
            "llm_label_overridden": result["guardrail"]["llm_label_overridden"],
            "fallback_reason": result["guardrail"]["fallback_reason"],
        }
        with self.audit_log.open("a") as f:
            f.write(json.dumps(entry, default=str) + "\n")


def render_note(note: dict, decision: dict) -> str:
    """The note as a reviewer sees it. The header is written by code from the
    decision, so it is authoritative whatever the note body says."""
    header = (f"Model label: {decision['label']} ({decision['confidence_band']} confidence"
              + (f", {format_confidence(decision['confidence'])}" if decision["confidence"] is not None else "")
              + f") · Route: {decision['route']} · Reasons: {'; '.join(decision['review_reasons']) or 'none'}")
    lines = [header, "", note["summary"], "", "Why it was flagged:"]
    lines += [f"- {w}" for w in note["why_flagged"]]
    lines += ["", "What to check:"] + [f"- {c}" for c in note["what_to_check"]]
    return "\n".join(lines)


def render_explanation(explanation: dict, decision: dict) -> str:
    """The explanation as a reviewer sees it; the header comes from the decision, not the writer."""
    header = (f"{decision['label']} · {format_confidence(decision['confidence'])} confidence ({decision['confidence_band']}) · "
              f"route: {decision['route']}" + (f" ({'; '.join(decision['review_reasons'])})" if decision["review_reasons"] else ""))
    lines = [header, "", f"Verdict: {explanation['verdict']}", "", "Why:"]
    lines += [f"- {r}" for r in explanation["reasons"]] or ["- no single feature stands out"]
    lines += ["", f"Runner-up: {explanation['runner_up']}"]
    if explanation["checks"]:
        lines += ["", "What to check:"] + [f"- {c}" for c in explanation["checks"]]
    return "\n".join(lines)


def from_config(cfg: dict, bundle: dict, note_writer=None) -> ReviewAgent:
    from .notes import make_note_writer
    a = cfg["agent"]
    return ReviewAgent(bundle, note_writer or make_note_writer(cfg), a["audit_log"], a["novelty_percentile"], a["similar_k"])
