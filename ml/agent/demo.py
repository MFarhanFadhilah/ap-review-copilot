"""Run the review agent on five validation-month lines and write the transcript.

    python -m ml.agent.demo                 # offline unless ANTHROPIC_API_KEY is set
    python -m ml.agent.demo --output docs/agent_demo.md

The model is the evaluation bundle, fitted on the training months only, so
every demo line is one the model has never seen (the shipped model was
trained on these months too, and would find each line as its own neighbour).
The bundle is built on the first run if it is missing.

The five lines are picked by fixed rules, not by hand:
1. a confident Cat1: the first auto-labelled validation line that is really Cat1, confidence >= 0.99
2. a Cat2: the first line predicted Cat2 that really is Cat2, from a known vendor
3. a rare class: the first line really in Cat3, Cat4 or Cat6 that the model labels correctly
4. a new vendor: the first line from a vendor with no training history that the model labels Cat1
5. a credit note: the first Cat5 line (Cat5 lines are credit notes)

If the validation months hold no line for a rule (Cat5 is nearly absent), that
example is skipped and the transcript says so.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from ml.clean import FEATURE_COLS, clean_training_data
from ml.features import time_split
from ml.predict import load_bundle, predict_records
from ml.split import load_development_raw
from ml.train import train

from .agent import from_config

ROOT = Path(__file__).resolve().parents[2]


def pick_examples(valid: pd.DataFrame, bundle: dict) -> dict[str, int]:
    preds = pd.DataFrame(predict_records(valid[FEATURE_COLS], bundle), index=valid.index)
    y = valid["label"]
    known = valid["Col1"].isin(set(bundle["known_vendors"]))
    pick = {
        "a confident Cat1": (preds["route"] == "auto") & (y == "Cat1") & (preds["confidence"] >= 0.99),
        "a Cat2": (preds["predicted_label"] == "Cat2") & (y == "Cat2") & known,
        "a rare class": y.isin(["Cat3", "Cat4", "Cat6"]) & (preds["predicted_label"] == y),
        "a new vendor": ~known & (preds["predicted_label"] == "Cat1"),
        "a credit note": y == "Cat5",
    }
    # A rule with no matching line is skipped (None) rather than crashing the demo.
    return {name: (mask[mask].index[0] if mask.any() else None) for name, mask in pick.items()}


def _record_table(record: dict) -> list[str]:
    names = {"Col1": "vendor", "Col2": "reference ID", "Col3": "amount", "Col4": "description",
             "Col5": "period", "Col6": "account", "Col7": "document type"}
    rows = ["| field | value |", "|---|---|"]
    rows += [f"| {c} ({names[c]}) | `{record[c]}` |" for c in FEATURE_COLS]
    return rows


def _tool_lines(step: dict) -> list[str]:
    name, out = step["tool"], step["output"]
    if name == "validate_record":
        text = f"errors: {out['errors'] or 'none'} · warnings: {out['warnings'] or 'none'}"
    elif name == "predict_and_route":
        text = (f"label **{out['predicted_label']}**, confidence {out['confidence']:.1%} ({out['confidence_band']}), "
                f"runner-up {out['runner_up_label']} → route **{out['route']}**"
                + (f" ({'; '.join(out['review_reasons'])})" if out["review_reasons"] else ""))
    elif name == "explain_prediction":
        text = "; ".join(f"{r['feature']} ({r['contribution']:+.2f})" for r in out["reasons"])
    elif name == "find_similar_invoices":
        mix = ", ".join(f"{k} ×{v}" for k, v in Counter(s["label"] for s in out["similar"]).most_common())
        text = (f"nearest similarity {out['nearest_similarity']:.2f} (novelty mark {out['novelty_threshold']:.2f}"
                f"{', **novel**' if out['novel'] else ''}); top {len(out['similar'])} labels: {mix}")
    elif name == "check_suspicious_text":
        text = ("**instruction-like text** in " + ", ".join(out["fields"])) if out["flag"] else "no instruction-like text"
    elif name == "value_frequencies":
        text = "; ".join(v["text"] for v in out.values())
    else:
        text = (out["status"] if out["status"] == "new vendor" else
                f"{out['lines']} lines, {out['first_period']} to {out['last_period']}: "
                + ", ".join(f"{k} {v}" for k, v in out["label_counts"].items()))
    return [f"- `{name}` → {text}"]


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(description="Run the review agent on five sample validation-month lines.")
    parser.add_argument("--output", default=str(ROOT / "docs" / "agent_demo.md"))
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    eval_path = ROOT / cfg["paths"]["model_dir"] / "eval_model.joblib"
    if not eval_path.exists():
        train(cfg, until_split=True, output=eval_path)
    bundle = load_bundle(str(eval_path))

    df, _, _ = clean_training_data(load_development_raw(cfg))
    _, valid = time_split(df, cfg["split"]["train_end_period"])
    examples = pick_examples(valid, bundle)

    agent = from_config(cfg, bundle)
    out = [
        "# Review agent: demo transcript",
        "",
        f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M UTC} by `python -m ml.agent.demo` · "
        f"note writer: **{agent.note_writer.mode}** · model `{bundle['metadata']['model_version']}` "
        f"(fitted on the training months only; threshold {bundle['threshold']:.3f}).",
        "",
        f"Up to five lines from the validation months ({valid['period'].min():%b %Y} to {valid['period'].max():%b %Y}), "
        "which the model has never seen, picked by fixed rules "
        "(see `ml/agent/demo.py`). For each: the input, every tool call in order, the decision and the reviewer note. "
        "The label and route always come from the model and `ml/policy.py`; the note only explains them. "
        "The true label is shown for the reader only; the agent never sees it.",
        "",
    ]
    for n, (title, idx) in enumerate(examples.items(), start=1):
        if idx is None:
            out += [f"## {n}. {title}", "", "No line in the validation months matches this rule, so it is skipped.", ""]
            continue
        record = {c: valid.loc[idx, c] for c in FEATURE_COLS}
        record["Col5"] = f"{pd.Timestamp(record['Col5']):%Y-%m-%d}"
        result = agent.run(record)
        out += [f"## {n}. {title}", "", *_record_table(record), "",
                f"True label (hidden from the agent): **{valid.loc[idx, 'label']}**", "", "**Tool calls**", ""]
        for step in result["steps"]:
            out += _tool_lines(step)
        out += ["", f"**Decision:** {result['label']} · route **{result['route']}**"
                + (f" · {'; '.join(result['review_reasons'])}" if result["review_reasons"] else ""), ""]
        if result["note_text"]:
            out += [f"**Reviewer note** ({result['mode']}):", "", "```text", result["note_text"], "```", ""]
            if result["guardrail"]["fallback_reason"]:
                out += [f"Template note used because: {result['guardrail']['fallback_reason']}", ""]
        else:
            out += ["Auto-labelled: no evidence gathering and no note needed. The decision is still logged.", ""]

    out += [
        "## Guardrails in this run",
        "",
        "- Label and route are copied from `predict_and_route` after the note is written; any label or route the "
        "LLM returns is overwritten, and the attempt is logged (`llm_label_overridden`).",
        "- A note naming a category that is not in the evidence is discarded and replaced by the template note.",
        "- Invoice fields reach the LLM only inside `<invoice_data>` / `<evidence>` blocks that the system prompt "
        "declares untrusted; the LLM has no tool that can change anything.",
        f"- Every decision above was appended to `{cfg['agent']['audit_log']}` (input hash, label, route, reasons, note, mode).",
        "",
    ]
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out))
    shown = sum(idx is not None for idx in examples.values())
    print(f"wrote {path} ({agent.note_writer.mode} mode, {shown} lines)")
    return path


if __name__ == "__main__":
    main()
