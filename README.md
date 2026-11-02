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

**Synthetic data (generator v1.0).** `data/synthetic/sample.csv` (6,000 rows) is committed. To make more:

```bash
python -m data.synthetic.generate --seed 42 --rows 6000 --out data/synthetic/sample.csv
python -m data.synthetic.generate --rows 10000000 --format parquet --out data/synthetic/out/big.parquet
python -m data.synthetic.measure data/synthetic/sample.csv     # each quirk, configured vs measured
```

## Results on synthetic demo data

These numbers come from `python -m ml.evaluate` on data from **generator v1.0** ([ADR 0004](docs/adr/0004-generator-calibration.md)). They show the pipeline working end to end. They say nothing about real data, because the generator's own rules decide how hard the task is. The generator's vendor structure was tuned toward five aggregate facts about real data, using structure measures only. No model result was looked at while choosing it.

Both sets use a time split. A model fitted on the months up to 2023-08 scores the four later months (2023-09 to 2023-12), as it would in use. The baseline is a vendor-lookup rule: each vendor's most common past label, and Cat1 for a vendor never seen. Brackets show 95% bootstrap intervals (2,000 paired resamples of the validation lines).

**Labelling**

| | 6,000-row sample (seed 42) | 60,000-row set (seed 43) |
|---|---|---|
| Validation lines | 788 | 9,074 |
| **Macro-F1, model** | 0.667 [0.466, 0.740] | 0.711 [0.662, 0.753] |
| **Macro-F1, vendor lookup** | 0.709 [0.496, 0.762] | 0.712 [0.653, 0.764] |
| Difference, model − lookup | [−0.106, 0.000] | [−0.057, +0.057] |
| Cat2 F1, model | 0.707 [0.619, 0.788] | 0.714 [0.691, 0.737] |
| Cat2 F1, vendor lookup | 0.766 [0.667, 0.847] | 0.743 [0.717, 0.768] |

**Seen and unseen vendors** (60,000-row set)

| | Lines | Accuracy, model / lookup | Macro-F1, model / lookup |
|---|---|---|---|
| Vendors seen in training | 8,319 | 93.7% / 96.5% | 0.708 / 0.834 |
| Vendors not seen in training | 755 | 89.3% / 85.4% | 0.587 / 0.184 |

**Routing.** The vendor lookup used as a router auto-labels a line only if every training line of its vendor was Cat1.

| | Sample: auto-labelled | Cat1 precision | Non-Cat1 auto | 60k: auto-labelled | Cat1 precision | Non-Cat1 auto |
|---|---|---|---|---|---|---|
| Model + policy | 68.8% | 99.08% | 5 | 71.9% | 98.39% | 105 |
| Vendor-lookup router | 48.0% | 100.00% | 0 | 43.8% | 99.87% | 5 |

**Rare classes** (60,000-row set, precision / recall)

| | Cat3 (46 lines) | Cat4 (23 lines) | Cat6 (39 lines) |
|---|---|---|---|
| Model | 0.872 / 0.891 | 0.387 / 0.522 | 0.473 / 0.667 |
| Vendor lookup | 1.000 / 0.870 | 1.000 / 0.217 | 1.000 / 0.385 |

**What this means.** On the larger set the model and the vendor lookup are level on macro-F1, and the interval for the difference sits evenly around zero. They get there in different ways. On vendors it has seen, the lookup labels better, because most vendors send one class and their history says which. On vendors it has never seen, the lookup can only guess Cat1, and the model is far better. For routing, the two sit at different points of one trade-off. The model auto-labels about 72% of lines and lets about 1.6% of its auto-labels be wrong (105 lines). The lookup router auto-labels about 44% and is almost never wrong. Neither result is a clear win, and that is the honest finding on this data.

**Next experiment: a hybrid.** Use the vendor lookup where a vendor has a clean single-class history, and the model everywhere else, with the same routing policy on top. It is not built yet.

The hold-out of the 6,000-row sample (537 lines after removing duplicate copies, 6 rare-class lines) gives 94.4% accuracy and 77.5% auto-labelled at 99.04% Cat1 precision for the shipped model. Full per-class numbers are in the [model card](models/MODEL_CARD.md).

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
- **Generator v1.0 is frozen.** Its three vendor-structure measures hold within ±0.03 of their targets from 6,000 to 600,000 rows. The two rare-class measures are near their targets on large files, but noisy on small ones. Changing the generator needs a new ADR (see ADR 0004).
- **Not ported yet:** the model-comparison ladder and its notebooks (the dashboard sections that showed them were removed), a React front end, Postgres, retraining and deployment.

## License

MIT. See [LICENSE](LICENSE).
