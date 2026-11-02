"""Measure every quirk in a generated file, next to its configured value.

    python -m data.synthetic.measure data/synthetic/sample.csv

The generator tests use ``measure()`` to check each quirk is present at
roughly its configured rate. The table it prints is the "configured vs
measured" comparison in the README. Everything is computed from the output
files alone, never from the generator's internals.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data.synthetic.generate import DEFAULT_CONFIG, load_config, near_duplicates_path

FEATURES = ["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7"]
CORRUPTED = re.compile(r"^\d\.\d+[A-Z]\+\d+$")


def read(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype={"Col2": str}, keep_default_na=False, na_values={"Col3": [""]})


def measure(path: str | Path, cfg: dict | None = None) -> dict:
    """Return ``{quirk: {"configured": ..., "measured": ...}}`` for one generated file."""
    cfg = cfg or load_config(DEFAULT_CONFIG)
    df = read(path)
    truth = pd.read_csv(near_duplicates_path(path))
    n = len(df)
    label = df["ClassificationLabel"].str.replace(r"\D", "", regex=True).map(lambda d: f"Cat{int(d)}")
    date = pd.to_datetime(df["Col5"])
    period = date.dt.to_period("M")
    t = cfg["timeline"]
    start, switch = pd.Period(t["start"], "M"), pd.Period(t["convention_switch"], "M")
    main = period >= start
    out: dict[str, dict] = {}

    def put(name, configured, measured):
        out[name] = {"configured": configured, "measured": measured}

    # Classes (on unique rows, so duplicate copies do not distort rare classes)
    key = df[FEATURES].astype(str).agg("|".join, axis=1)
    unique = ~key.duplicated()
    shares = label[unique].value_counts(normalize=True)
    for c, s in cfg["classes"]["shares"].items():
        put(f"class share {c} (unique rows)", s, round(float(shares.get(c, 0)), 4))
    cat5 = label == "Cat5"
    put("Cat5 rows", cfg["classes"]["cat5_rows"], int(cat5.sum()))
    put("Cat5 rows that are credit notes", 1.0, float((df.loc[cat5, "Col3"] < 0).mean()) if cat5.any() else None)

    put("label spellings", sum(len(v) for v in cfg["labels"]["spellings"].values()),
        int(df["ClassificationLabel"].nunique()))

    # Exact duplicates (same Col1-Col7 and class), the way cleaning counts them
    full = key + "|" + label
    put("extra exact copies, share of rows (random + big line)",
        round(cfg["duplicates"]["extra_copy_rate"] + (cfg["duplicates"]["big_line"]["copies"]
                                                      - cfg["duplicates"]["big_line"]["accounts"]) / n, 4),
        round(float(full.duplicated().mean()), 4))
    big = df.groupby(["Col1", "Col2", "Col3", "Col4", "Col5"]).agg(rows=("Col6", "size"), accounts=("Col6", "nunique"))
    top = big.sort_values("rows", ascending=False).iloc[0]
    b = cfg["duplicates"]["big_line"]
    put("big line: copies / accounts", f"{b['copies']} / {b['accounts']}", f"{int(top['rows'])} / {int(top['accounts'])}")
    labels_per_key = label.groupby(key).nunique()
    put("groups of identical lines with conflicting labels", 1, int((labels_per_key > 1).sum()))

    # Near duplicates
    o, d = df.iloc[truth["original_row"]].reset_index(drop=True), df.iloc[truth["duplicate_row"]].reset_index(drop=True)
    same = (o[["Col1", "Col3", "Col4", "Col6"]].to_numpy() == d[["Col1", "Col3", "Col4", "Col6"]].to_numpy()).all(axis=1)
    gap = (pd.to_datetime(d["Col5"]).dt.to_period("M") - pd.to_datetime(o["Col5"]).dt.to_period("M")).map(lambda x: x.n)
    differs = (o["Col2"].to_numpy() != d["Col2"].to_numpy()) | (o["Col5"].to_numpy() != d["Col5"].to_numpy())
    put("near-duplicate pairs, share of rows", cfg["near_duplicates"]["rate"], round(len(truth) / n, 4))
    put("near-duplicate pairs valid (same vendor/amount/account/description, differ, <= 1 period apart)", 1.0,
        float((same & differs & gap.between(0, 1).to_numpy()).mean()))
    put("near-duplicate copies in the next period", cfg["near_duplicates"]["next_period_share"],
        round(float((gap == 1).mean()), 3))
    put("near-duplicate copies that are also exact duplicates", 0, int(full.iloc[truth["duplicate_row"]].isin(
        full.drop(index=truth["duplicate_row"])).sum()))

    # Reference IDs
    corrupted = df["Col2"].fillna("").str.match(CORRUPTED)
    top_vendor = df.loc[corrupted, "Col1"].value_counts()
    put("corrupted IDs, share of rows", 0.06, round(float(corrupted.mean()), 4))
    put("corrupted IDs from one vendor", None, round(float(top_vendor.iloc[0] / corrupted.sum()), 3))
    put("that vendor's lines corrupted", cfg["ref_ids"]["corrupted_vendor_share"],
        round(float(corrupted[df["Col1"] == top_vendor.index[0]].mean()), 3))
    shape = np.select([df["Col2"].fillna("").eq(""), corrupted, df["Col2"].str.contains(r"[\s\-/_.]", na=False),
                       df["Col2"].str.isdigit().fillna(False)], ["missing", "corrupted", "separator", "numeric"],
                      "alphanumeric")
    put("ID shapes present", 5, int(len(set(shape))))

    # Dates
    month_end = date.dt.is_month_end
    put("month-start dates before the switch", 1.0, float((date.dt.day[main & (period < switch)] == 1).mean()))
    put("month-end dates from the switch on", 1.0, float(month_end[period >= switch].mean()))
    put("legacy rows (before the main block)", round(t["legacy_share"], 4), round(float((~main).mean()), 4))

    # Fixed-fee vendor: one amount, one account, several labels
    by_vendor = df.assign(label=label, period=period).groupby("Col1")
    stats = by_vendor.agg(amounts=("Col3", "nunique"), accounts=("Col6", "nunique"), labels=("label", "nunique"),
                          rows=("Col3", "size"))
    fixed = stats[(stats.amounts == 1) & (stats.accounts == 1) & (stats.labels >= 2) & (stats.rows >= 6)]
    if len(fixed):
        seq = df.assign(label=label, period=period)[df["Col1"] == fixed.index[0]].sort_values("period")["label"]
        put("fixed-fee vendor label changes", len(cfg["fixed_fee"]["schedule"]) - 1,
            int((seq != seq.shift()).sum() - 1))
    else:
        put("fixed-fee vendor label changes", len(cfg["fixed_fee"]["schedule"]) - 1, 0)

    # Vendor and account structure
    vendor_labels = stats["labels"]
    put("vendors with one class", 0.89, round(float((vendor_labels == 1).mean()), 3))
    put("rows from vendors with more than one class", 0.45,
        round(float(df["Col1"].map(vendor_labels).gt(1).mean()), 3))
    group = df["Col6"].str.split().str[0]
    modal = df.groupby(group)["Col7"].agg(lambda s: s.value_counts().iloc[0]).sum() / n
    put("doc type matches the account group's usual type", cfg["accounts"]["doc_type_purity"], round(float(modal), 4))

    # New vendors in the last months
    k = cfg["new_vendors"]["months"]
    window = period >= start + (t["months"] - k)
    seen_before = set(df.loc[~window, "Col1"])
    put(f"lines in the last {k} months from vendors new in that window", cfg["new_vendors"]["late_share"],
        round(float((~df.loc[window, "Col1"].isin(seen_before)).mean()), 4))

    # Rare-class structure (ADR 0004, round 2), on unique rows
    rare = label.isin(["Cat3", "Cat4", "Cat6"]) & unique
    put("rare-class lines from vendors with more than one class", 0.78,
        round(float(df.loc[rare, "Col1"].map(vendor_labels).gt(1).mean()), 3))
    rare_late = rare & window
    put(f"rare-class lines in the last {k} months whose vendor appears earlier", 0.81,
        round(float(df.loc[rare_late, "Col1"].isin(seen_before).mean()), 3) if rare_late.any() else None)
    put("rare-class lines in those months", None, int(rare_late.sum()))

    # Description signal: tokens whose lines are mostly Cat2 (base rate about 10%)
    tokens = df.assign(label=label)[["Col4", "label"]].assign(tok=lambda f: f["Col4"].str.split()).explode("tok")
    per_tok = tokens.dropna(subset=["tok"]).groupby("tok")["label"].agg(rows="size", cat2=lambda s: (s == "Cat2").mean())
    common = per_tok[per_tok.rows >= 10]
    put("common description tokens (>= 10 rows)", None, int(len(common)))
    put("of which >= 80% Cat2", None, int((common.cat2 >= 0.8).sum()))
    put("empty descriptions, share of rows", cfg["description"]["missing_share"], round(float(df["Col4"].eq("").mean()), 4))
    put("credit notes, share of rows", cfg["amounts"]["credit_note_share"], round(float((df["Col3"] < 0).mean()), 4))
    return out


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description="Measure the quirks in a generated file.")
    parser.add_argument("path")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args(argv)
    result = measure(args.path, load_config(args.config))
    width = max(len(k) for k in result)
    print(f"{'quirk':<{width}}  {'configured':>12}  {'measured':>12}")
    for name, v in result.items():
        conf = "-" if v["configured"] is None else str(v["configured"])
        print(f"{name:<{width}}  {conf:>12}  {str(v['measured']):>12}")
    sys.stdout.flush()
    return result


if __name__ == "__main__":
    main()
