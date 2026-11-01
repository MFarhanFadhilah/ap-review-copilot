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

These numbers come from `python -m ml.evaluate` on the committed sample. They show the pipeline working end to end. They say nothing about real data, because the generator's own rules decide how hard the task is.

**Validation months (2023-09 to 2023-12, 797 lines).** A model fitted on the earlier months scores the later ones, as it would in use.

| | Model | Vendor-lookup rule |
|---|---|---|
| Macro-F1 | 0.604 | 0.646 |
| Cat2 F1 | 0.571 | 0.476 |
| Accuracy | 90.6% | |
| Auto-labelled | **61.0%** of lines, at **98.35%** Cat1 precision (target 98%) | |
| Sent to review | 311 lines | |
| Non-Cat1 lines auto-labelled | 8 | |

Recall by class: Cat1 0.928 · Cat2 0.687 · Cat3 1.000 (4 lines) · Cat6 0.667 (3 lines). The validation months have no Cat4 lines.

**On macro-F1 the vendor lookup beats the model.** The gap comes from 4 lines, all Cat1 lines the model flagged as rare classes. In the synthetic data each rare class comes from a few pure vendors, which a lookup gets right. On Cat2, the class with volume, the model is clearly better. The false alarms go to review, so they cost reviewer time, not wrong postings. A small sweep of the learnability knobs (see [Limits](#limits)) shows the lookup wins on macro-F1 in most settings, and loses when more non-Cat1 lines come from mixed vendors.

**Hold-out (shipped model, 537 lines after removing duplicate copies):** 90.5% accuracy, 72.6% auto-labelled at 97.95% Cat1 precision. It holds only 6 rare-class lines, so its macro-F1 (0.486) says little.

Full per-class numbers are in the [model card](models/MODEL_CARD.md).

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
- **Rare classes are tiny.** The validation months hold 0 to 4 lines per rare class, so one line moves macro-F1 a lot.
- **Cat5 is not learned.** It has 3 lines, all credit notes. Every credit note goes to review instead.
- **Knob sensitivity.** Each run below regenerates the whole data set, so the rare-class numbers also move with the random draw. Validation months, one knob changed at a time:

  | Setting | Macro-F1 (lookup) | Cat2 F1 (lookup) | Auto-labelled at precision |
  |---|---|---|---|
  | defaults | 0.604 (0.646) | 0.571 (0.476) | 61.0% at 98.35% |
  | `description_signal` 0.2 | 0.655 (0.730) | 0.656 (0.685) | 49.0% at 99.21% |
  | `description_signal` 0.6 | 0.626 (0.649) | 0.671 (0.487) | 70.8% at 98.56% |
  | `label_noise` 0.0 | 0.673 (0.675) | 0.734 (0.611) | 77.5% at 98.53% |
  | `label_noise` 0.05 | 0.727 (0.784) | 0.410 (0.385) | 22.9% at 97.19% |
  | `vendor_class_overlap` 0.1 | 0.886 (0.880) | 0.662 (0.633) | 78.4% at 98.87% |
  | `vendor_class_overlap` 0.6 | 0.676 (0.600) | 0.627 (0.051) | 65.5% at 99.04% |

- **Not ported yet:** the model-comparison ladder and its notebooks (the dashboard sections that showed them were removed), a React front end, Postgres, retraining and deployment.

## License

MIT. See [LICENSE](LICENSE).
