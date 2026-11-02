# Synthetic invoice lines

This folder makes the demo data. Everything in this repo runs on it.

**The generator never reads real data.** Every parameter is a number typed by hand into `config.yaml`. The real data this project was first built on is private and is not in this repo. Some shares are loosely shaped like published aggregate facts about that kind of data, such as heavy class imbalance and many duplicate postings. No vendor, account, description or token comes from it.

## Files

| File | What it is |
|---|---|
| `config.yaml` | every parameter, one knob per quirk |
| `generate.py` | the generator |
| `measure.py` | measures each quirk in a generated file, next to its configured value |
| `sample.csv` | 6,000 rows from seed 42. Committed, so the repo works without running anything |
| `sample_near_duplicates.csv` | ground truth for the near-duplicate postings planted in the sample |

## Run it

```bash
python -m data.synthetic.generate --seed 42 --rows 6000 --out data/synthetic/sample.csv
python -m data.synthetic.measure data/synthetic/sample.csv
python -m data.synthetic.generate --rows 10000000 --format parquet --out data/synthetic/out/big.parquet
```

The same seed and config always give byte-identical files. CI checks this by regenerating the sample and comparing it with the committed one. The check depends on the pinned NumPy version, because NumPy does not promise the same random stream across versions.

## Output format

The raw file has columns `Col1` to `Col7` plus `ClassificationLabel`. That is the format the pipeline expects.

| Column | Meaning | Example |
|---|---|---|
| Col1 | vendor, as anonymised tokens | `Ven107` |
| Col2 | reference ID | `26805430`, `JB66818`, `95807-10`, `3.17Q+11` |
| Col3 | amount, negative for a credit note | `760.0` |
| Col4 | description tokens | `Tok614 Tok253` |
| Col5 | posting period, as a date | `2023-10-31` |
| Col6 | account code. The first token is the account group | `Acc2 Acc45` |
| Col7 | document type | `NoDoc` |
| ClassificationLabel | class, in one of several spellings | `Category_2` |

Tokens use the `Ven`, `Tok` and `Acc` prefixes on purpose. The private data used a different scheme, so a leak check can flag any token from it without false alarms.

The near-duplicate ground truth is a separate file because a later stage adds a duplicate-payment check. Keeping the answers out of the raw file means that check can be measured honestly.

## What it copies, and what it does not

It reproduces these quirks of real accounts-payable data. Each one has a knob in `config.yaml`.

- **Class imbalance.** About 89% Cat1 and 10% Cat2, plus four rare classes. Cat5 has only 3 rows, and all of them are credit notes.
- **Label spellings.** 11 spellings for 6 classes, such as `Category_1`, `category 1` and `Categry_6`.
- **Exact duplicates.** About 10% extra copies. That includes one large line posted 50 times in one month across 30 accounts, and one group of identical lines with conflicting labels.
- **Near duplicates.** The same vendor, amount, account and description posted again in the same or the next period, with a slightly different reference ID.
- **Excel-damaged IDs** like `3.17Q+11`, nearly all from one vendor, among several normal ID shapes.
- **A date convention change.** Month-start dates early on, month-end dates later, and a few legacy rows well before the main block.
- **Labelling drift.** Fixed-fee vendors (one per 6,000 lines) bill the same amount to one account every month, and their label changes twice.
- **Structure.** Most vendors send one class and some are mixed. Account codes are hierarchical, and the doc type almost always follows the account group.
- **New vendors** in the last four months.
- **Description signal.** Some tokens are strongly linked to a class. Most are shared noise.

It does **not** copy:

- Any real vendor, account, description, token, amount or date.
- The real number of vendors or accounts, the real timeline, or the real model scores.
- Real-world messiness beyond the list above, such as free-text descriptions, multiple currencies, tax lines or partial payments.

## Generator v1.0: calibrated vendor structure

The vendor structure was tuned toward five aggregate facts about real accounts-payable data, in two rounds, using structure measures only. No model result was looked at while choosing it. The full procedure, the knobs and every value tried are in [ADR 0004](../../docs/adr/0004-generator-calibration.md). That config is frozen as **generator v1.0** (`generator_version` in `config.yaml`). Any further change to the generator needs a new ADR.

| Structure | Target | 6,000 rows (seed 42) | 60,000 rows (seed 43) | 600,000 rows (seed 44) |
|---|---|---|---|---|
| Rows from vendors with more than one label | 0.45 ± 0.03 | 0.431 | 0.467 | 0.468 |
| Vendors with one label | 0.89 ± 0.03 | 0.893 | 0.886 | 0.889 |
| Rows with an Excel-damaged ID | 0.06 ± 0.03 | 0.060 | 0.0595 | 0.0588 |
| Rare-class lines from multi-label vendors | about 0.78 | 0.724 | 0.767 | 0.761 |
| Rare-class validation lines from a vendor seen earlier | about 0.81 | 0.875 (8 lines) | 0.685 (108 lines) | 0.810 (964 lines) |

Three things keep the structure the same at every size. Vendor counts grow in proportion to rows. Vendor sizes are fixed quantiles of one lognormal curve, not random draws. And there is one fixed-fee vendor per 6,000 lines.

## Learnability knobs

Three knobs set how hard the task is.

| Knob | Default | Raising it |
|---|---|---|
| `label_noise` | 0.02 | Swaps more Cat1 and Cat2 labels (only on mixed vendors, see `label_noise_scope`). Lowers the precision any model can reach, so fewer lines can be auto-labelled. |
| `vendor_class_overlap` | 0.35 | Sends more non-Cat1 lines through vendors that mostly send Cat1. A vendor lookup gets worse and the description matters more. |
| `description_signal` | 0.35 | Makes description tokens more class-specific. The model gets more confident, so more lines are auto-labelled. |

`account_class_signal` (0.35) works the same way for the account field.

The defaults were picked so routing matters: a clear share of lines goes to review, Cat1 auto-label precision is high but not perfect, and rare classes are imperfect. They were not tuned to match any real-data score, or to make the model look good.

## Scale

Rows are NumPy index arrays into vendors, accounts and tokens that are built once. There is no Python loop over rows. Rows are made and written in chunks of `chunk_rows` (1,000,000), so memory depends on the chunk size, not the total.

Measured on an Apple-silicon laptop:

| Rows | Format | Wall time | Peak memory (RSS) |
|---|---|---|---|
| 1,000,000 | CSV | 4.4 s | 1.3 GB |
| 1,000,000 | Parquet | 2.4 s | 1.3 GB |
| 10,000,000 | Parquet | 21.6 s | 1.8 GB |

In a multi-chunk run, exact and near duplicates stay within a chunk. Fixed-fee vendors are spread over the chunks. The other one-off cases (the large repeated line, the conflict group and the Cat5 rows) appear once, in the first chunk.
