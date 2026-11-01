# Model card: invoice-line category classifier (synthetic demo model)

| | |
|---|---|
| **Model** | L2 logistic regression, balanced class weights, C = 10, temperature-calibrated (`ml/train.py`, `ml/calibration.py`) |
| **Artefact** | `models/model.joblib`, built by `python -m ml.train` (not committed). The bundle holds the model, the threshold, the known vendors, the explanation baseline, the training lines and metadata. `models/training_summary.json` sits next to it. |
| **Version** | `model_version` in `training_summary.json`: the training timestamp plus a hash of the data file |
| **Training data** | **Synthetic.** `data/synthetic/sample.csv`, 6,000 generated lines (see [`data/synthetic/README.md`](../data/synthetic/README.md)) |
| **Classes** | Cat1, Cat2, Cat3, Cat4, Cat6. **Cat5 is not learned** (see Limits) |
| **Owner** | Muhammad Farhan Fadhilah. In real use, each category would need a named business owner. |

This card describes the model as trained on the synthetic demo data. The same model on the private data is described in [`docs/historical_results.md`](../docs/historical_results.md).

## Intended use

The model suggests a category for each accounts-payable line. `ml/policy.py` then decides whether the line is **auto-labelled** or goes to a **human reviewer**. Only Cat1 predictions can be auto-labelled, and only when the calibrated confidence is at or above the threshold, the vendor is known, the line is not a credit note and nothing in the line is new to the model. Everything else goes to review, with reasons.

**Not intended for:** auto-labelling any Cat2 to Cat6 line, or lines from another organisation, chart of accounts or labelling scheme without retraining. The demo model is trained on synthetic data and must not be used on real invoices.

## Training data

- 6,000 synthetic lines from seed 42. The 10% hold-out (602 rows) is set aside first by `ml/split.py` and never used for training.
- Cleaning (`ml/clean.py`) on the other 90% (5,398 rows): 11 label spellings mapped to 6 classes, 558 exact-duplicate copies removed (425 groups) and 1 conflicting-label group resolved to the majority. That leaves 4,839 lines.
- The shipped model trains on all development months except Cat5: **4,836 lines** (Cat1 4,315 · Cat2 467 · Cat3 22 · Cat4 15 · Cat6 17).
- Features (`ml/features.py`): vendor tokens and the vendor itself, reference-ID shape and length, log amount, amount bands, credit-note and round-amount flags, TF-IDF on description words and word pairs, account and account group, and document type. The posting period is not a feature. It is used only for the time split.

## Evaluation (synthetic data)

All numbers come from `python -m ml.evaluate`. See [`docs/split_strategy.md`](../docs/split_strategy.md) for the split. The README also reports a 60,000-row set with bootstrap intervals and seen versus unseen vendors.

**Validation months (2023-09 to 2023-12, 795 lines the evaluation model never saw).** The evaluation model is fitted on the training months only, with its own threshold of 0.780.

| Class | Lines | Precision | Recall | F1 | Vendor-lookup F1 |
|---|---|---|---|---|---|
| Cat1 | 713 | 0.966 | 0.958 | 0.962 | 0.971 |
| Cat2 | 72 | 0.646 | 0.736 | 0.688 | 0.718 |
| Cat3 | 4 | 1.000 | 1.000 | 1.000 | 1.000 |
| Cat4 | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| Cat6 | 2 | 0.500 | 0.500 | 0.500 | 0.667 |
| **Macro-F1** | | | | **0.630** | **0.671** |

The validation months also hold 3 Cat5 lines. The model cannot predict Cat5. As credit notes they go to review, and they count in accuracy and routing, not in macro-F1.

- **Routing:** 74.6% of lines auto-labelled at 98.31% Cat1 precision (target 98%). 202 lines go to review. 10 non-Cat1 lines were auto-labelled.
- **Hold-out (shipped model, 537 lines after removing duplicate copies):** 96.1% accuracy, macro-F1 0.664, 78.8% auto-labelled at 99.53% Cat1 precision, 2 non-Cat1 lines auto-labelled. The hold-out has 9 rare-class lines in total, so its macro-F1 says little.
- **Shipped threshold: 0.771**, set out-of-fold on all development months. Out-of-fold, that gives 79.8% automation at 98.00% precision.

**The vendor lookup beats the model on macro-F1.** On the sample the difference is within noise (95% interval −0.111 to +0.005). On the 60,000-row set it is clear (−0.180 to −0.082). The model finds more rare-class lines than the lookup, but with many false alarms, and on the larger set the lookup is never wrong when it names a rare class. See the README for the full comparison.

## Explanations and the review agent

- **Per-line reasons** (`ml/explain.py`): for the predicted class against the runner-up, (weight difference) × (feature value − training mean), summed per field. For a linear model this is the same formula SHAP's linear explainer uses.
- **Agent** (`ml/agent/`): a fixed flow that calls seven read-only tools (validation, prediction and route, reasons, similar past lines, vendor history, a suspicious-text check, value frequencies) and writes a short note. With `ANTHROPIC_API_KEY` set, Claude writes the note. Otherwise a template does. The model and the policy make every decision. The LLM only writes the text.
- **Guardrails:** the label and route are copied from the model and policy after the LLM writes, so the LLM cannot change them. Instruction-like text in a field is flagged by a fixed regex and treated as data. A category the evidence does not support makes the agent fall back to the template, and any failed check does the same with the reason logged.

## Limits

- **Rare classes are tiny.** 15 to 22 training lines each, and 1 to 4 in the validation months. Per-class scores rest on a few lines.
- **Cat5 is not learned.** The sample has 3 lines, all credit notes. Every credit note goes to review instead.
- **Label drift.** The fixed-fee vendor's identical lines change label twice. The model learns past labels and cannot foresee a change in the labelling scheme.
- **Synthetic data.** The generator's rules define what is learnable. A good score here says the pipeline works, not that it would score the same on real data.

## Monitoring signals

Reviewer override rate on sampled auto-labels (must stay at or above 98% correct) · review-queue size and its mix of reasons · share of lines from unseen vendors and accounts (6.3% in the validation months) · per-vendor category share (drift) · share of auto-labels just above the threshold · novelty-flag rate from the agent · validation errors, warnings and duplicate flags.

## Retraining

- **Scheduled:** monthly or quarterly, adding reviewer decisions as new labels.
- **On a trigger:** the override rate on sampled auto-labels falls below 98%, a vendor's category share changes, the unseen-vendor share rises above about 10%, or a new kind of validation error appears.
- **Promotion rule:** `python -m ml.train` re-sets the threshold out-of-fold and writes a new versioned bundle. Replace the live model only if, on the latest months it was not trained on, per-class F1 is at least as good and automation precision stays at or above 98%. Keep the previous bundle for rollback.
