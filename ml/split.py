"""The 10% hold-out test set, and the 90% development set everything else uses.

    python -m ml.split        # writes data/holdout/holdout_features.xlsx and holdout_labels.csv

* Rows are split **before cleaning**, straight from the raw file.
* **Stratified** by the normalised label, so each class keeps its share.
* **Grouped** by exact-duplicate group (identical Col1-Col7 after the same
  standardisation the cleaning uses), so every copy of a line lands on the
  same side. Otherwise a copy in development would leak into the test.
* **Seeded** from ``config.yaml → split.holdout_seed``: the same rows every run.

Development code never reads the raw file directly; it calls
``load_development_raw``, which applies the same split and drops the hold-out.
See docs/split_strategy.md.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

from ml.clean import FEATURE_COLS, LABEL_COL, TARGET_COL, load_raw, normalize_label, standardize_records

ROOT = Path(__file__).resolve().parent.parent
HOLDOUT_DIR = ROOT / "data" / "holdout"


def duplicate_groups(raw: pd.DataFrame) -> pd.Series:
    """One group id per raw row; exact duplicates (same Col1-Col7) share an id."""
    std = standardize_records(raw[FEATURE_COLS])
    return std.groupby(FEATURE_COLS, dropna=False, sort=False).ngroup()


def split_raw(raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(development_rows, holdout_rows)``, both in raw format with the original index."""
    fraction, seed = cfg["split"]["holdout_fraction"], cfg["split"]["holdout_seed"]
    labels = raw[LABEL_COL].map(normalize_label)
    folds = StratifiedGroupKFold(n_splits=round(1 / fraction), shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        # Cat5 has fewer rows than the 10 folds; its rows simply land wherever their fold puts them.
        warnings.filterwarnings("ignore", message="The least populated class")
        _, holdout_idx = next(folds.split(raw, labels, groups=duplicate_groups(raw)))
    is_holdout = raw.index.isin(raw.index[holdout_idx])
    return raw[~is_holdout], raw[is_holdout]


def load_development_raw(cfg: dict) -> pd.DataFrame:
    """The raw file minus the hold-out. Every development step starts here."""
    raw = load_raw(ROOT / cfg["paths"]["raw_data"])
    development, holdout = split_raw(raw, cfg)
    saved = HOLDOUT_DIR / "holdout_labels.csv"
    if saved.exists() and pd.read_csv(saved)["row_id"].tolist() != holdout.index.tolist():
        raise RuntimeError("data/holdout no longer matches the split in config.yaml; re-run `python -m ml.split`")
    return development


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    raw = load_raw(ROOT / cfg["paths"]["raw_data"])
    development, holdout = split_raw(raw, cfg)

    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
    holdout[FEATURE_COLS].to_excel(HOLDOUT_DIR / "holdout_features.xlsx", index=False)
    labels = pd.DataFrame({"row_id": holdout.index, LABEL_COL: holdout[LABEL_COL],
                           TARGET_COL: holdout[LABEL_COL].map(normalize_label)})
    labels.to_csv(HOLDOUT_DIR / "holdout_labels.csv", index=False)

    groups = duplicate_groups(raw)
    shared = set(groups[development.index]) & set(groups[holdout.index])
    counts = pd.DataFrame({
        "development": development[LABEL_COL].map(normalize_label).value_counts(),
        "holdout": labels[TARGET_COL].value_counts(),
    }).fillna(0).astype(int).sort_index()
    print(f"raw rows {len(raw):,} → development {len(development):,} ({len(development) / len(raw):.1%}), "
          f"hold-out {len(holdout):,} ({len(holdout) / len(raw):.1%})")
    print(f"duplicate groups on both sides: {len(shared)} (must be 0)")
    print(counts.to_string())
    print(f"wrote {HOLDOUT_DIR.relative_to(ROOT)}/holdout_features.xlsx and holdout_labels.csv "
          "(same row order; row_id = row number in the raw file)")


if __name__ == "__main__":
    main()
