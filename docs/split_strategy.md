# Split strategy: train, validation and test

The data is used in three parts. Each part answers a different question, so they are kept apart.

| Set | Rows (synthetic sample) | Used for |
|---|---|---|
| **Test (hold-out)** | 10% of the raw file: 602 rows, 537 after removing duplicate copies | Scored once, by the shipped model. Never used to fit, calibrate or set a threshold. |
| **Train** | development lines posted up to 2023-08: 4,044 cleaned lines | Fitting, calibration and the auto-label threshold of the evaluation model |
| **Validation** | development lines posted 2023-09 to 2023-12: 795 cleaned lines | The time check. A model fitted on the training months scores the later months, as it would in use. |

"Development" is the other 90% of the raw file: 5,398 rows, or 4,839 after cleaning. The shipped model (`models/model.joblib`) is fitted on all development months, so it learns from the newest lines too.

## How the hold-out is carved (`ml/split.py`)

- **From the raw file, before cleaning.** The test rows are then exactly what a new file would look like (`data/holdout/holdout_features.xlsx`, Col1 to Col7 in raw format, with labels in `data/holdout/holdout_labels.csv`).
- **Stratified by the normalised label**, so each class keeps its share. In the sample that gives Cat1 534, Cat2 59, Cat4 5, Cat3 2 and Cat6 2. Cat5 has 3 rows in total, and all of them stay in development.
- **Grouped by exact-duplicate group**, so every copy of a line lands on the same side. Otherwise a copy in development would leak the answer to its twin in the test set.
- **Seeded** from `config.yaml` (`split.holdout_seed` 2026, `split.holdout_fraction` 0.10). `python -m ml.split` rewrites the two files. Every development step reads the raw file through `load_development_raw`, which applies the same split and drops the hold-out.

Mechanically it is one fold of a 10-fold `StratifiedGroupKFold`.

## Why a time split as well

A random hold-out mixes every month, so most of its vendors and wording also appear in training. That flatters a model. In use, the model always scores a month it has never seen, with some vendors it has never seen. The time split copies that. In the sample, 6.3% of validation lines come from vendors with no training history.

## What the numbers can and cannot show

- The validation months hold only a handful of rare-class lines (Cat3 4, Cat5 3, Cat6 2, Cat4 1). One wrong rare-class line moves macro-F1 a lot, so read the rare-class scores as rough. The README adds a 60,000-row set for steadier numbers.
- The hold-out has a similar handful (9 rare-class lines). It mostly confirms the Cat1 and Cat2 behaviour.
- The model type and its settings were chosen on the private data before this repo existed (see [`historical_results.md`](historical_results.md)). They are fixed in `config.yaml` and were not re-tuned on the synthetic data. So neither the validation months nor the hold-out were used to choose anything here.
