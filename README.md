# AP Review Copilot

An invoice-line classifier that **auto-labels only what is safe and sends everything else to a person, with reasons**.

Accounts-payable teams label thousands of invoice lines by hand. Most lines are routine, and a few are rare and costly to get wrong. This project lets a model take the routine ones and leaves the rest to people:

- A **model** suggests a category for every line.
- A **routing policy** (`ml/policy.py`) auto-labels a line only when the model predicts the common class with high calibrated confidence, the vendor is known and nothing about the line is new. Everything else goes to review.
- A **review agent** gathers evidence for each flagged line (reasons, similar past lines, vendor history) and writes a short note for the reviewer. The model and the policy make every decision. The LLM only writes the note, and it works offline with a template when no API key is set.

> **The demo runs on synthetic data.** The project was first built on a private, anonymised file that cannot be published. Every file in this repo was generated from hand-set parameters (`data/synthetic/`). Results on the private data are in [`docs/historical_results.md`](docs/historical_results.md), as aggregate numbers only.

## Run it (Python 3.11+)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt      # exact versions (requirements.txt has the minimums)

python -m ml.split                        # set aside the 10% hold-out (data/holdout/)
python -m ml.train                        # fit the model (models/model.joblib), about 10 s
python -m ml.evaluate                     # score the later months (models/evaluation/)
python -m data.synthetic.generate --seed 43 --rows 60000 --out data/synthetic/out/eval_60k.csv
python -m ml.evaluate --data data/synthetic/out/eval_60k.csv   # the larger set, about 70 s

streamlit run app/dashboard.py            # dashboard on http://localhost:8501
uvicorn api.main:app --reload             # API: GET /health, POST /predict
python -m ml.predict --input data/holdout/holdout_features.xlsx --output predictions.csv
python -m ml.agent.demo                   # agent on five lines -> docs/agent_demo.md
python -m pytest                          # 142 tests
```

**LLM notes (optional).** Put `ANTHROPIC_API_KEY=...` in a `.env` file (git-ignored), or paste a key into the dashboard sidebar. Without a key everything still works, and notes come from a template labelled "Offline template explanation".

**Synthetic data.** `data/synthetic/sample.csv` (6,000 rows) is committed. To make more:

```bash
python -m data.synthetic.generate --seed 42 --rows 6000 --out data/synthetic/sample.csv
python -m data.synthetic.generate --rows 10000000 --format parquet --out data/synthetic/out/big.parquet
python -m data.synthetic.measure data/synthetic/sample.csv     # each quirk, configured vs measured
```

## Results on synthetic demo data

These numbers come from `python -m ml.evaluate`. They show the pipeline working end to end. They say nothing about real data, because the generator's own rules decide how hard the task is. The generator's vendor structure was set to match three aggregate facts about real data, without looking at any model result ([ADR 0004](docs/adr/0004-generator-calibration.md)).

Both sets use a time split. A model fitted on the months up to 2023-08 scores the four later months (2023-09 to 2023-12), as it would in use. The baseline is a vendor-lookup rule: each vendor's most common past label, and Cat1 for a vendor never seen. Brackets show 95% bootstrap intervals (2,000 paired resamples of the validation lines).

| | 6,000-row sample (seed 42) | 60,000-row set (seed 43) |
|---|---|---|
| Validation lines | 795 | 9,091 |
| **Macro-F1, model** | 0.630 [0.512, 0.736] | 0.678 [0.639, 0.715] |
| **Macro-F1, vendor lookup** | 0.671 [0.519, 0.750] | 0.811 [0.766, 0.850] |
| Difference, model − lookup | [−0.111, +0.005] | [−0.180, −0.082] |
| Cat2 F1, model | 0.688 [0.601, 0.767] | 0.739 [0.716, 0.760] |
| Cat2 F1, vendor lookup | 0.718 [0.627, 0.800] | 0.722 [0.696, 0.749] |
| Auto-labelled at Cat1 precision | 74.6% at 98.31% | 76.4% at 98.24% |
| Non-Cat1 lines auto-labelled | 10 | 122 |

Lines from vendors seen and not seen in the training months (60,000-row set):

| | Lines | Accuracy, model / lookup | Macro-F1, model / lookup |
|---|---|---|---|
| Seen vendors | 8,441 | 92.7% / 95.2% | 0.680 / 0.817 |
| Unseen vendors | 650 | 94.8% / 91.2% | 0.546 / 0.318 |

Rare classes on the 60,000-row set (44, 34 and 31 lines):

| | Cat3 precision / recall | Cat4 precision / recall | Cat6 precision / recall |
|---|---|---|---|
| Model | 0.529 / 0.841 | 0.431 / 0.824 | 0.333 / 0.839 |
| Vendor lookup | 1.000 / 0.682 | 1.000 / 0.588 | 1.000 / 0.677 |

**What this means.** On this synthetic data the vendor lookup is the better labeller overall, and on the larger set the gap is clear: the whole interval for the difference is below zero. The reason shows in the rare classes. Each rare class comes from a few vendors, so when the lookup names a rare class it is always right, but it misses every rare line from a vendor that mostly sends Cat1. The model, trained with balanced class weights, finds about 83% of rare lines but raises one to two false alarms for each real one. The model is clearly ahead only on vendors it has never seen, where the lookup can only guess Cat1, and roughly level on Cat2. In this system every rare-class prediction goes to a reviewer, so the model's false alarms cost review time rather than wrong labels, while a rare line labelled Cat1 can be auto-labelled. That trade-off may suit routing, but it is an argument, not a measured win. On these numbers, a lookup-first hybrid or a different class weighting is worth testing next.

The hold-out of the 6,000-row sample (537 lines after removing duplicate copies, 9 rare-class lines) gives 96.1% accuracy and 78.8% auto-labelled at 99.53% Cat1 precision for the shipped model. Full per-class numbers are in the [model card](models/MODEL_CARD.md).

## How it works

| Step | Where | What it does |
|---|---|---|
| Split | `ml/split.py` | Sets aside 10% of the raw file as a hold-out, stratified by label, with duplicate copies kept together |
| Clean | `ml/clean.py` | Maps label spellings to classes, removes exact duplicates (and reports them as possible duplicate payments), resolves conflicting labels, and validates single records |
| Features | `ml/features.py` | Vendor tokens, reference-ID shape, amount bands, TF-IDF on descriptions, account and account group |
| Train | `ml/train.py`, `ml/calibration.py` | Logistic regression with balanced class weights and temperature scaling. The auto-label threshold is set out-of-fold. |
| Route | `ml/policy.py` | The only place a route is decided |
| Explain | `ml/explain.py` | Per-line reasons: weight difference × (value − training average), summed per field |
| Agent | `ml/agent/` | Fixed flow of read-only tools, then a note. Guardrails stop the LLM from changing a label or route, and an audit log records every decision. |
| Serve | `ml/predict.py`, `api/main.py`, `app/dashboard.py` | CLI, FastAPI and a Streamlit dashboard, all calling `predict_records` |

Design notes: [`docs/adr/`](docs/adr/) · split: [`docs/split_strategy.md`](docs/split_strategy.md) · generator: [`data/synthetic/README.md`](data/synthetic/README.md).

## Public-safety check

`tools/check_public_safe.py` fails if the repo holds a file over 5 MB, an `.xls` file, a token from the old anonymisation scheme, a word from a local git-ignored `.denylist`, or (when given the private file by path) any multi-token string from it. CI runs the parts that need no private data.

## Limits

- **Synthetic data.** A good score here means the code works, not that the model would do as well on real invoices.
- **Rare classes are tiny in the sample.** Its validation months hold 1 to 4 lines per rare class, so one line moves macro-F1 a lot. That is why the 60,000-row set is reported too.
- **Cat5 is not learned.** The sample has 3 lines, all credit notes. Every credit note goes to review instead.
- **Knob sensitivity.** The learnability knobs move these results a lot. A sweep on an earlier version of the generator showed the auto-label share ranging from 23% to 78% across settings. That sweep has not been rerun on the current config.
- **Scale changes structure.** The vendor structure was matched to real aggregates at 6,000 rows. At 60,000 rows the same config gives 0.664 of rows from mixed-label vendors (target 0.45), 0.857 single-class vendors (target 0.89) and 0.045 damaged IDs (target 0.06), because each vendor sends more lines.
- **Not ported yet:** the model-comparison ladder and its notebooks (the dashboard sections that showed them were removed), a React front end, Postgres, retraining and deployment.

## License

MIT. See [LICENSE](LICENSE).
