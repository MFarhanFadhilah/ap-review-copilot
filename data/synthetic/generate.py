"""Seeded generator for synthetic accounts-payable invoice lines.

    python -m data.synthetic.generate --seed 42 --rows 6000 --out data/synthetic/sample.csv
    python -m data.synthetic.generate --rows 10000000 --format parquet --out data/synthetic/out/big.parquet

It writes the raw format the pipeline reads (Col1 to Col7 plus
ClassificationLabel) and, next to it, ``<name>_near_duplicates.csv``: the
ground truth for the near-duplicate postings it planted. The ground truth is
kept out of the raw file so a later duplicate-payment check can be measured
against it honestly.

Every parameter comes from ``data/synthetic/config.yaml``. Nothing is read
from real data.

How a row is made, in order:

1. **Class first.** Each row's class is drawn from the configured shares, so
   the class balance is exact. Everything else is then drawn given the class.
2. **Vendor**, from the vendors whose main class is that class, or (with the
   ``vendor_class_overlap`` chance) from a "mixed" vendor of another class.
   Lines late in the timeline can come from vendors held back as "new".
3. **Account**: the vendor's home account, or (with ``account_class_signal``)
   an account from the groups the class tends to use. Doc type follows the
   account group.
4. **Description**: 1 to 4 tokens. Each is a class signal token with chance
   ``description_signal``, otherwise common noise shared by all classes.
5. **Amount, reference ID, date** from per-class and per-vendor rules.
6. **Label noise and spelling**: a few Cat1 and Cat2 labels are swapped, then each label is
   written in one of the configured spellings.

Then the special cases are added (the fixed-fee vendor, the large repeated
line, the conflicting-label group, the Cat5 credit notes), then exact and near
duplicates, and the rows are shuffled.

Why it is fast: entities (vendors, accounts, tokens) are built once. Rows are
NumPy index arrays into them, so a row costs a few array operations, never a
Python loop. Rows are made and written in chunks of ``chunk_rows``, so memory
stays flat however many rows are asked for.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.yaml"
COLUMNS = ["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "ClassificationLabel"]
CLASSES = ["Cat1", "Cat2", "Cat3", "Cat4", "Cat5", "Cat6"]
IDX = {c: i for i, c in enumerate(CLASSES)}
CAT1, CAT2, CAT5 = IDX["Cat1"], IDX["Cat2"], IDX["Cat5"]
DOC_TYPES = np.array(["NoDoc", "Doc1", "Doc2", "Doc3"], dtype=object)
# No "E": a corrupted ID must never show the real letter, as in the damage it imitates.
LETTERS = np.array(list("ABCDFGHIJKLMNOPQRSTUVWXYZ"), dtype=object)


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text())


def near_duplicates_path(out: str | Path) -> Path:
    out = Path(out)
    return out.with_name(f"{out.stem}_near_duplicates.csv")


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _scaled(base: int, rows: int, minimum: int) -> int:
    """Entity counts grow with sqrt(rows): more lines bring more vendors, but far fewer than lines."""
    return max(minimum, round(base * np.sqrt(rows / 6000)))


def _zipf(n: int, exponent: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Zipf weights summing to 1, in random order if ``rng`` is given."""
    w = 1.0 / np.arange(1, n + 1) ** exponent
    w = w[rng.permutation(n)] if rng is not None else w
    return w / w.sum()


def _split_counts(total: int, shares: dict) -> dict:
    """Largest-remainder rounding, so the counts add up to ``total`` exactly."""
    names = list(shares)
    p = np.array([shares[n] for n in names], float)
    raw = p / p.sum() * total
    counts = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - counts), kind="stable")[: total - counts.sum()]:
        counts[i] += 1
    return dict(zip(names, counts.tolist()))


def _month(text: str) -> np.datetime64:
    return np.datetime64(text, "M")


def _months(n: int, rng: np.random.Generator, start: np.datetime64, span: int) -> np.ndarray:
    return start + rng.integers(0, span, n).astype("timedelta64[M]")


def _text(a) -> np.ndarray:
    """Any array as an object array of Python strings, so ``+`` concatenates element-wise."""
    return np.asarray(a).astype(str).astype(object)


def _zfill(a, width: int) -> np.ndarray:
    """Integers as zero-padded text. (np.char.zfill fails on an empty array.)"""
    a = np.asarray(a)
    return np.char.zfill(a.astype(str), width).astype(object) if a.size else np.array([], dtype=object)


# --------------------------------------------------------------------------
# The world: vendors, accounts and tokens, built once per run
# --------------------------------------------------------------------------
@dataclass
class World:
    vendor_names: np.ndarray
    vendor_class: np.ndarray
    vendor_weight: np.ndarray
    vendor_new: np.ndarray
    vendor_mixed: np.ndarray
    vendor_shape: np.ndarray       # 0 numeric, 1 alphanumeric, 2 separator, 3 missing
    vendor_home: np.ndarray        # (vendors, 2) home accounts
    vendor_offset: np.ndarray      # log-amount offset per vendor
    corrupt_vendor: int
    big_vendor: int
    conflict_vendor: int
    account_names: np.ndarray
    account_group: np.ndarray
    group_doc: np.ndarray
    class_accounts: dict
    token_names: np.ndarray
    class_signal: dict
    noise_tokens: np.ndarray
    noise_p: np.ndarray
    fixed_fee_name: str
    fixed_fee_account: int
    fixed_fee_description: str


def build_world(cfg: dict, rows: int, rng: np.random.Generator) -> World:
    v, a, d = cfg["vendors"], cfg["accounts"], cfg["description"]

    # Vendors and their main class. Rare classes get at least 2 vendors each.
    n_v = _scaled(v["count"], rows, 50)
    counts = _split_counts(n_v, v["class_share"])
    for c in ("Cat3", "Cat4", "Cat6"):
        short = max(0, 2 - counts[c])
        counts[c] += short
        counts["Cat1"] -= short
    vendor_class = np.concatenate([np.full(k, IDX[c]) for c, k in counts.items()])
    rng.shuffle(vendor_class)
    weight = _zipf(n_v, v["zipf_exponent"], rng)

    # "New" vendors only post in the last months. Taken from Cat1 and Cat2 vendors.
    common = np.flatnonzero(vendor_class <= CAT2)
    new = np.zeros(n_v, bool)
    new[rng.choice(common, round(cfg["new_vendors"]["vendor_share"] * n_v), replace=False)] = True

    # Mixed vendors: the largest established Cat1 vendors, which also send other classes.
    est_cat1 = np.flatnonzero((vendor_class == CAT1) & ~new)
    by_size = est_cat1[np.argsort(-weight[est_cat1], kind="stable")]
    # The largest one is kept pure Cat1: it is the vendor whose IDs Excel damaged.
    n_mixed = max(1, round(v["mixed_share"] * len(est_cat1)))
    mixed = np.zeros(n_v, bool)
    mixed[by_size[1:n_mixed + 1]] = True
    corrupt_vendor = int(by_size[0])
    # Two more pure Cat1 vendors carry the big repeated line and the conflicting-label group.
    big_vendor, conflict_vendor = (int(x) for x in by_size[n_mixed + 1:n_mixed + 3])

    # Names: one unique token plus 0-3 tokens shared across vendors.
    n_shared = v["shared_name_tokens"]
    shared = np.array([f"Ven{n_v + 1 + j}" for j in range(n_shared)], dtype=object)
    shared_p = _zipf(n_shared, 1.0, rng)
    uid = rng.permutation(n_v) + 1
    n_extra = rng.choice(4, n_v, p=v["name_extra_tokens_p"])
    names = np.array([
        " ".join([f"Ven{uid[i]}", *rng.choice(shared, n_extra[i], replace=False, p=shared_p)])
        for i in range(n_v)   # a loop over vendors (hundreds), not over rows
    ], dtype=object)

    shapes = cfg["ref_ids"]["shape_p"]
    shape = rng.choice(4, n_v, p=[shapes[k] for k in ("numeric", "alphanumeric", "separator", "missing")])
    shape[corrupt_vendor] = 0     # its IDs were long numbers before Excel damaged them

    # Accounts: the first token is the group, then a unique token, sometimes a shared one.
    n_a = _scaled(a["count"], rows, 40)
    n_groups = a["groups"]
    gw = np.array(a["group_weights"], float)
    account_group = rng.choice(n_groups, n_a, p=gw / gw.sum())
    account_group[:n_groups] = np.arange(n_groups)   # every group has at least one account
    unique_tok = rng.permutation(n_a) + n_groups + 1
    shared_tok = n_groups + n_a + 1 + rng.integers(0, 30, n_a)
    account_names = _text([f"Acc{g + 1}" for g in account_group]) + " " + _text([f"Acc{t}" for t in unique_tok])
    has_third = rng.random(n_a) < 0.5
    account_names[has_third] = account_names[has_third] + " " + _text([f"Acc{t}" for t in shared_tok[has_third]])
    class_accounts = {IDX[c]: np.flatnonzero(np.isin(account_group, np.array(g) - 1))
                      for c, g in a["class_groups"].items()}

    # Each vendor has two home accounts in the groups its main class uses.
    home = np.empty((n_v, 2), int)
    for c in np.unique(vendor_class):
        members = np.flatnonzero(vendor_class == c)
        pool = class_accounts[c]
        home[members] = pool[rng.integers(0, len(pool), (len(members), 2))]

    # Description tokens: a signal set per class, the rest is shared Zipf noise.
    n_tok = _scaled(d["vocabulary"], rows, 300)
    token_names = np.array([f"Tok{i}" for i in rng.permutation(n_tok) + 1], dtype=object)
    order = rng.permutation(n_tok)
    class_signal, pos = {}, 0
    for c, k in d["signal_tokens"].items():
        class_signal[IDX[c]] = order[pos:pos + k]
        pos += k
    noise = order[pos:]

    # The fixed-fee vendor is its own entity: it posts nothing but its monthly fee.
    fixed_account_pool = class_accounts[IDX["Cat3"]]
    return World(
        vendor_names=names, vendor_class=vendor_class, vendor_weight=weight, vendor_new=new,
        vendor_mixed=mixed, vendor_shape=shape, vendor_home=home,
        vendor_offset=rng.normal(0, v["amount_spread"], n_v),
        corrupt_vendor=corrupt_vendor, big_vendor=big_vendor, conflict_vendor=conflict_vendor,
        account_names=account_names, account_group=account_group,
        group_doc=np.array(a["group_doc_types"], dtype=object), class_accounts=class_accounts,
        token_names=token_names, class_signal=class_signal, noise_tokens=noise,
        noise_p=_zipf(len(noise), d["noise_zipf_exponent"]),
        fixed_fee_name=f"Ven{n_v + n_shared + 1} {shared[0]}",
        fixed_fee_account=int(fixed_account_pool[rng.integers(len(fixed_account_pool))]),
        fixed_fee_description=" ".join(token_names[noise[rng.choice(len(noise), 2, replace=False)]]),
    )


# --------------------------------------------------------------------------
# Row attributes, all vectorised
# --------------------------------------------------------------------------
def _vendors(w: World, cfg: dict, cls: np.ndarray, from_new: np.ndarray, rng) -> np.ndarray:
    learn = cfg["learnability"]
    out = np.empty(len(cls), int)
    established = ~w.vendor_new
    fallback = np.flatnonzero(established & (w.vendor_class == CAT1))
    for c in np.unique(cls):
        rows = np.flatnonzero(cls == c)
        chance = learn["vendor_class_overlap"] * (learn["cat1_overlap_factor"] if c == CAT1 else 1.0)
        new = from_new[rows]
        borrowed = (rng.random(len(rows)) < chance) & ~new
        pools = [
            (~borrowed & ~new, np.flatnonzero(established & (w.vendor_class == c))),
            (borrowed, np.flatnonzero(established & (w.vendor_class == CAT2)) if c == CAT1
             else np.flatnonzero(w.vendor_mixed)),
            (new, np.flatnonzero(w.vendor_new & (w.vendor_class == c))),
        ]
        for mask, pool in pools:
            if not mask.any():
                continue
            pool = pool if len(pool) else fallback   # e.g. Cat5 has no vendors of its own
            p = w.vendor_weight[pool]
            out[rows[mask]] = rng.choice(pool, mask.sum(), p=p / p.sum())
    return out


def _accounts(w: World, cfg: dict, cls: np.ndarray, vendor: np.ndarray, rng) -> np.ndarray:
    second = rng.random(len(cls)) >= cfg["accounts"]["home_account_p"]
    acc = w.vendor_home[vendor, second.astype(int)]
    by_class = rng.random(len(cls)) < cfg["learnability"]["account_class_signal"]
    for c in np.unique(cls[by_class]):
        rows = np.flatnonzero(by_class & (cls == c))
        pool = w.class_accounts[c]
        acc[rows] = pool[rng.integers(0, len(pool), len(rows))]
    return acc


def _doc_types(w: World, cfg: dict, account: np.ndarray, rng) -> np.ndarray:
    doc = w.group_doc[w.account_group[account]].copy()
    off = rng.random(len(account)) >= cfg["accounts"]["doc_type_purity"]
    doc[off] = DOC_TYPES[rng.integers(0, len(DOC_TYPES), off.sum())]
    return doc


def _descriptions(w: World, cfg: dict, cls: np.ndarray, rng) -> np.ndarray:
    d = cfg["description"]
    n = len(cls)
    length = rng.choice(4, n, p=d["length_p"]) + 1
    ids = w.noise_tokens[rng.choice(len(w.noise_tokens), (n, 4), p=w.noise_p)]
    signal = rng.random((n, 4)) < cfg["learnability"]["description_signal"]
    for c in np.unique(cls):
        slots = signal & (cls == c)[:, None]
        pool = w.class_signal[c]
        ids[slots] = pool[rng.integers(0, len(pool), slots.sum())]
    words = w.token_names[ids]
    text = words[:, 0].copy()
    for j in range(1, 4):
        longer = length > j
        text[longer] = text[longer] + " " + words[longer, j]
    text[rng.random(n) < d["missing_share"]] = ""
    return text


def _amounts(w: World, cfg: dict, cls: np.ndarray, vendor: np.ndarray, rng) -> np.ndarray:
    am = cfg["amounts"]
    median = np.array([am["median"][c] for c in CLASSES], float)
    sigma = np.array([am["sigma"][c] for c in CLASSES], float)
    log_amount = np.log(median[cls]) + sigma[cls] * rng.standard_normal(len(cls)) + w.vendor_offset[vendor]
    amount = np.round(np.exp(log_amount), 2)
    whole = rng.random(len(cls)) < am["round_share"]
    amount[whole] = np.maximum(100.0, np.round(amount[whole] / 100) * 100)
    credit = (cls <= CAT2) & (rng.random(len(cls)) < am["credit_note_share"])
    amount[credit] *= -1
    amount[cls == CAT5] = -np.abs(amount[cls == CAT5])   # Cat5 lines are always credit notes
    return amount


def _ref_ids(w: World, cfg: dict, vendor: np.ndarray, rng) -> np.ndarray:
    r = cfg["ref_ids"]
    n = len(vendor)
    shape = w.vendor_shape[vendor]
    digits = rng.integers(10_000, 100_000_000, n)   # 5 to 8 digits
    out = _text(digits)
    alnum = shape == 1
    out[alnum] = (LETTERS[rng.integers(0, len(LETTERS), alnum.sum())]
                  + LETTERS[rng.integers(0, len(LETTERS), alnum.sum())]
                  + _zfill((digits[alnum] % 100_000), 5))
    sep = shape == 2
    out[sep] = (_zfill((digits[sep] % 100_000), 5) + "-"
                + _text(rng.integers(10, 30, sep.sum())))
    out[shape == 3] = ""
    corrupted = (((vendor == w.corrupt_vendor) & (rng.random(n) < r["corrupted_vendor_share"]))
                 | (rng.random(n) < r["corrupted_stray_share"]))
    k = corrupted.sum()
    out[corrupted] = (_text(rng.integers(1, 10, k)) + "."
                      + _zfill(rng.integers(0, 100, k), 2)
                      + LETTERS[rng.integers(0, len(LETTERS), k)] + "+11")
    return out


def _dates(cfg: dict, month: np.ndarray) -> np.ndarray:
    """Month-start dates before the convention switch, month-end dates from it on."""
    switch = _month(cfg["timeline"]["convention_switch"])
    start = month.astype("datetime64[D]")
    end = (month + np.timedelta64(1, "M")).astype("datetime64[D]") - np.timedelta64(1, "D")
    return _text(np.datetime_as_string(np.where(month < switch, start, end), unit="D"))


def _spellings(cfg: dict, cls: np.ndarray, rng) -> np.ndarray:
    out = np.empty(len(cls), dtype=object)
    for c in np.unique(cls):
        options = cfg["labels"]["spellings"][CLASSES[c]]
        names = np.array(list(options), dtype=object)
        p = np.array(list(options.values()), float)
        rows = cls == c
        out[rows] = names[rng.choice(len(names), rows.sum(), p=p / p.sum())]
    return out


def _make_rows(w: World, cfg: dict, cls: np.ndarray, month: np.ndarray, rng,
               from_new: np.ndarray | None = None, vendor: np.ndarray | None = None) -> dict:
    """All attributes for rows of the given (true) classes and months."""
    from_new = np.zeros(len(cls), bool) if from_new is None else from_new
    vendor = _vendors(w, cfg, cls, from_new, rng) if vendor is None else vendor
    account = _accounts(w, cfg, cls, vendor, rng)
    return {
        "cls": cls.copy(), "month": month, "vendor": vendor,
        "Col1": w.vendor_names[vendor], "Col2": _ref_ids(w, cfg, vendor, rng),
        "Col3": _amounts(w, cfg, cls, vendor, rng), "Col4": _descriptions(w, cfg, cls, rng),
        "Col6": w.account_names[account], "Col7": _doc_types(w, cfg, account, rng),
    }


def _concat(parts: list[dict]) -> dict:
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _take(rows: dict, idx: np.ndarray) -> dict:
    return {k: v[idx] for k, v in rows.items()}


# --------------------------------------------------------------------------
# Special cases (first chunk only)
# --------------------------------------------------------------------------
def _fixed_fee(w: World, cfg: dict, rng) -> dict:
    t, f = cfg["timeline"], cfg["fixed_fee"]
    month = _month(t["start"]) + np.arange(t["months"]).astype("timedelta64[M]")
    changes = sorted((_month(m), IDX[c]) for m, c in f["schedule"].items())
    cls = np.array([[c for m0, c in changes if m0 <= m][-1] for m in month])
    n = len(month)
    return {
        "cls": cls, "month": month, "vendor": np.full(n, -1),
        "Col1": np.full(n, w.fixed_fee_name, dtype=object),
        "Col2": _text([f"{40_000 + 37 * i}-{22 + i // 12}" for i in range(n)]),   # one invoice number a month
        "Col3": np.full(n, float(f["amount"])),
        "Col4": np.full(n, w.fixed_fee_description, dtype=object),
        "Col6": np.full(n, w.account_names[w.fixed_fee_account], dtype=object),
        "Col7": np.full(n, w.group_doc[w.account_group[w.fixed_fee_account]], dtype=object),
    }


def _big_line(w: World, cfg: dict, rng) -> dict:
    """One line posted many times in one month across many accounts."""
    b = cfg["duplicates"]["big_line"]
    accounts = rng.choice(w.class_accounts[CAT1], b["accounts"], replace=False)
    # Every account gets one copy, the rest land on random accounts of the same set.
    per_copy = np.concatenate([accounts, rng.choice(accounts, b["copies"] - b["accounts"])])
    n = b["copies"]
    one = _make_rows(w, cfg, np.array([CAT1]), np.array([_month(b["month"])]), rng,
                     vendor=np.array([w.big_vendor]))
    rows = _take(one, np.zeros(n, int))
    rows["Col3"] = np.full(n, float(b["amount"]))
    rows["Col6"] = w.account_names[per_copy]
    rows["Col7"] = w.group_doc[w.account_group[per_copy]]
    return rows


def _conflict_group(w: World, cfg: dict, rng) -> dict:
    """Identical lines that were labelled differently."""
    labels = np.concatenate([np.full(k, IDX[c]) for c, k in cfg["duplicates"]["conflict_group"].items()])
    t = cfg["timeline"]
    one = _make_rows(w, cfg, np.array([CAT1]), _months(1, rng, _month(t["start"]), t["months"]), rng,
                     vendor=np.array([w.conflict_vendor]))
    rows = _take(one, np.zeros(len(labels), int))
    rows["cls"] = labels
    return rows


def _cat5(w: World, cfg: dict, rng) -> dict:
    t = cfg["timeline"]
    n = cfg["classes"]["cat5_rows"]
    return _make_rows(w, cfg, np.full(n, CAT5), _months(n, rng, _month(t["start"]), t["months"]), rng)


# --------------------------------------------------------------------------
# One chunk
# --------------------------------------------------------------------------
def generate_chunk(cfg: dict, w: World, n_rows: int, rng, first: bool, row_offset: int = 0
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(raw rows, near-duplicate ground truth)`` for one chunk."""
    t, dup, near = cfg["timeline"], cfg["duplicates"], cfg["near_duplicates"]
    start, span = _month(t["start"]), t["months"]

    specials = [_fixed_fee(w, cfg, rng), _big_line(w, cfg, rng), _conflict_group(w, cfg, rng), _cat5(w, cfg, rng)] \
        if first else []
    n_special = sum(len(s["cls"]) for s in specials)
    n_dup = round(n_rows * dup["extra_copy_rate"])
    n_near = round(n_rows * near["rate"])
    n_base = n_rows - n_dup - n_near - n_special
    if n_base <= 0:
        raise ValueError(f"--rows {n_rows} is too small for the configured special cases")

    # Class counts. The fixed-fee lines count towards their classes, so rare classes are not inflated.
    target = _split_counts(n_base + (len(specials[0]["cls"]) if first else 0), cfg["classes"]["shares"])
    if first:
        for c in specials[0]["cls"]:
            target[CLASSES[c]] -= 1
    deficit = n_base - sum(max(0, k) for k in target.values())
    target = {c: max(0, k) for c, k in target.items()}
    target["Cat1"] += deficit
    cls = np.concatenate([np.full(k, IDX[c]) for c, k in target.items()])
    rng.shuffle(cls)

    # Months: the main block, a few legacy Cat1 rows, and "new vendor" lines in the last months.
    month = _months(n_base, rng, start, span)
    n_legacy = round(t["legacy_share"] * n_base)
    legacy = rng.choice(np.flatnonzero(cls == CAT1), n_legacy, replace=False)
    legacy_start = _month(t["legacy_start"])
    legacy_span = int((_month(t["legacy_end"]) - legacy_start).astype(int)) + 1
    month[legacy] = _months(n_legacy, rng, legacy_start, legacy_span)
    late = month >= start + np.timedelta64(span - cfg["new_vendors"]["months"], "M")
    from_new = late & (cls <= CAT2) & (rng.random(n_base) < cfg["new_vendors"]["late_share"])
    base = _make_rows(w, cfg, cls, month, rng, from_new=from_new)

    # Label noise on base rows only: the special cases keep their planned labels, legacy rows stay Cat1.
    # Half the flips go Cat1 -> Cat2 and half Cat2 -> Cat1, so the class shares stay as configured.
    k = round(cfg["learnability"]["label_noise"] * n_base / 2)
    in_main = month >= start
    to_cat2 = rng.choice(np.flatnonzero(in_main & (cls == CAT1)), k, replace=False)
    to_cat1 = rng.choice(np.flatnonzero(in_main & (cls == CAT2)), k, replace=False)
    base["cls"][to_cat2], base["cls"][to_cat1] = CAT2, CAT1

    rows = _concat([base, *specials])

    # Exact duplicates: whole groups of identical copies of base rows; most groups have 2 copies.
    sources_used = np.zeros(n_base, bool)
    if n_dup:
        k = rng.geometric(dup["copies_geometric_p"], n_dup)
        total = np.cumsum(k)
        groups = int(np.searchsorted(total, n_dup)) + 1
        k = k[:groups]
        k[-1] -= total[groups - 1] - n_dup
        sources = rng.choice(n_base, groups, replace=False)
        sources_used[sources] = True
        copies = _take(rows, np.repeat(sources, k))
    else:
        copies = _take(rows, np.array([], int))

    # Near duplicates: same vendor, amount, account and description; a slightly different
    # reference ID, a later period (within a few weeks), or both.
    in_block = (base["month"] >= start) & (base["Col3"] > 0) & ~sources_used
    originals = rng.choice(np.flatnonzero(in_block), n_near, replace=False)
    nd = _take(rows, originals)
    # "Within a few weeks" of a monthly posting period means the same period or the next one.
    can_move = nd["month"] < start + np.timedelta64(span - 1, "M")
    move = can_move & (rng.random(n_near) < near["next_period_share"])
    # A copy that stays in the same period must differ in its ID; a moved copy keeps its ID half the time.
    change_ref = ~move | (rng.random(n_near) < 0.5)
    nd["month"] = np.where(move, nd["month"] + np.timedelta64(1, "M"), nd["month"])
    nd["Col2"] = np.where(change_ref, _nudge_ids(nd["Col2"], rng), nd["Col2"])
    changed = np.where(change_ref & move, "ref_id+period", np.where(move, "period", "ref_id"))

    allrows = _concat([rows, copies, nd])
    allrows["Col5"] = _dates(cfg, allrows["month"])
    allrows["ClassificationLabel"] = _spellings(cfg, allrows["cls"], rng)

    # Shuffle, and translate the near-duplicate pairs to final row numbers.
    n_total = len(allrows["cls"])
    order = rng.permutation(n_total)
    position = np.empty(n_total, int)
    position[order] = np.arange(n_total)
    first_copy = len(rows["cls"]) + len(copies["cls"])
    truth = pd.DataFrame({
        "original_row": position[originals] + row_offset,
        "duplicate_row": position[first_copy + np.arange(n_near)] + row_offset,
        "changed": changed,
    }).sort_values("duplicate_row", kind="stable").reset_index(drop=True)
    frame = pd.DataFrame({c: allrows[c][order] for c in COLUMNS})
    return frame, truth


def _nudge_ids(ids: np.ndarray, rng) -> np.ndarray:
    """Change one character of each ID: the last digit if it is a digit, else append "1"."""
    s = pd.Series(ids, dtype=object).astype(str)
    last = s.str[-1].fillna("")
    is_digit = last.str.isdigit()
    bump = pd.Series(rng.integers(1, 10, len(s)), index=s.index)
    new_digit = ((pd.to_numeric(last.where(is_digit), errors="coerce").fillna(0).astype(int) + bump) % 10).astype(str)
    out = np.where(is_digit, s.str[:-1] + new_digit, s + "1")
    out = np.where(s.eq(""), _text(rng.integers(100_000, 1_000_000, len(s))), out)   # a missing ID gets one
    return out.astype(object)


# --------------------------------------------------------------------------
# Whole run
# --------------------------------------------------------------------------
def generate(cfg: dict, rows: int, seed: int, out: str | Path, fmt: str = "csv") -> dict:
    """Generate ``rows`` rows in chunks and write them to ``out``. Returns a short summary."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    chunk = int(cfg["chunk_rows"])
    sizes = [chunk] * (rows // chunk) + ([rows % chunk] if rows % chunk else [])
    # One independent stream per chunk: chunk i always gets the same stream for a given seed.
    world_seed, *chunk_seeds = np.random.SeedSequence(seed).spawn(1 + len(sizes))
    world = build_world(cfg, rows, np.random.default_rng(world_seed))

    started = time.perf_counter()
    writer, truths, offset = None, [], 0
    for i, (size, s) in enumerate(zip(sizes, chunk_seeds)):
        frame, truth = generate_chunk(cfg, world, size, np.random.default_rng(s), first=i == 0, row_offset=offset)
        if fmt == "csv":
            frame.to_csv(out, mode="w" if i == 0 else "a", header=i == 0, index=False)
        else:
            import pyarrow as pa
            import pyarrow.parquet as pq
            table = pa.Table.from_pandas(frame, preserve_index=False)
            writer = writer or pq.ParquetWriter(out, table.schema)
            writer.write_table(table)
        truths.append(truth)
        offset += len(frame)
    if writer is not None:
        writer.close()
    truth_path = near_duplicates_path(out)
    pd.concat(truths, ignore_index=True).to_csv(truth_path, index=False)
    return {"rows": offset, "chunks": len(sizes), "seconds": round(time.perf_counter() - started, 2),
            "out": str(out), "near_duplicates": str(truth_path)}


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description="Generate synthetic invoice lines (never reads real data).")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--seed", type=int, help="default: seed in the config file")
    parser.add_argument("--rows", type=int, help="default: rows in the config file")
    parser.add_argument("--out", required=True, help="output file (.csv or .parquet)")
    parser.add_argument("--format", choices=["csv", "parquet"], default="csv")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    summary = generate(cfg, args.rows or cfg["rows"], cfg["seed"] if args.seed is None else args.seed,
                       args.out, args.format)
    print(f"wrote {summary['rows']:,} rows in {summary['chunks']} chunk(s), {summary['seconds']} s -> {summary['out']}")
    print(f"near-duplicate ground truth -> {summary['near_duplicates']}")
    return summary


if __name__ == "__main__":
    main()
