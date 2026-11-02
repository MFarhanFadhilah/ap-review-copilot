# 4. Calibrate the generator to structural targets

**Status:** accepted

## Context

The first version of the generator got its class shares and quirk rates right, but its vendor structure was far from real accounts-payable data. In the 6,000-row sample, 75% of vendors sent one class (real data: 89%), lines from vendors with more than one label were 75% of rows (real data: 45%), and damaged IDs were 10.5% of rows (real data: 6.1%).

Vendor structure decides how much a vendor lookup can do, and so how hard the task is. A demo should be hard in a realistic way, not in whatever way the first guesses produced.

The risk is tuning the data until the model wins. So the targets and the knobs are fixed here, before any value is tried, and model results play no part in the choice.

## Targets

Taken from aggregate facts about the private data (see [`historical_results.md`](../historical_results.md)). They are measured on the generated file by `python -m data.synthetic.measure`, on unique rows after normalising labels, the same way the private-data figures were counted.

| Structure | Target | Measure |
|---|---|---|
| Rows from vendors with more than one label | about 0.45 | `rows from vendors with more than one class` |
| Vendors with one label | about 0.89 | `vendors with one class` |
| Rows with an Excel-damaged ID | about 0.06 | `corrupted IDs, share of rows` |

## Knobs that may change

Only knobs that shape these structures:

- `vendors.mixed_share`: how many vendors send more than one class.
- `vendors.mixed_pick`: which vendors are mixed (new option, see below).
- `vendors.zipf_exponent`: how concentrated lines are in a few vendors.
- `learnability.cat1_overlap_factor`: how often Cat1 lines go through Cat2 vendors, which makes those vendors mixed.
- `learnability.label_noise_scope`: where label noise lands (new option, see below). The noise **rate** stays at 0.02.
- `ref_ids.corrupted_vendor_share`: the share of the damaged vendor's lines with a damaged ID.

## Knobs that stay fixed

`label_noise` (0.02), `vendor_class_overlap` (0.35), `description_signal` (0.35) and `account_class_signal` (0.35) set learnability, not structure, so they stay as they are. Class shares, quirk rates and the timeline stay too.

## New options

- **`label_noise_scope`.** With `all`, noise flips labels on any vendor. A big vendor then almost always gets one flipped line and counts as mixed, so the single-class share cannot reach 0.89 while the noise rate stays at 2%. With `mixed_vendors`, flips land only on lines from vendors that already send more than one class. That also matches how labelling disagreement tends to arise: on vendors whose lines are genuinely ambiguous, not on a vendor that always sends the same thing.
- **`mixed_pick`.** With `largest`, the mixed vendors are the biggest Cat1 vendors. With `random`, they are drawn from all established Cat1 vendors with chance proportional to size. The expectation was that the biggest 11% of vendors hold far more than 45% of rows under a Zipf size curve, so `largest` might not hit both vendor targets at once. The search showed otherwise (see Result).

## Procedure

1. Generate the 6,000-row sample for each candidate setting (same seed 42).
2. Read only the three measures above from `data/synthetic/measure.py`. Do not train or evaluate.
3. Pick the setting closest to all three targets. Prefer changing fewer knobs.
4. Freeze the config, regenerate the committed sample, and only then train and evaluate.

## Result

80 settings were generated and measured: noise scope × mixed pick × `mixed_share` (0.06 to 0.15) × `zipf_exponent` (0.8, 1.0) × `cat1_overlap_factor` (0.1, 0.03). Then `corrupted_vendor_share` was set on its own, since it moves only the damaged-ID share. No model was trained during the search.

The closest setting changes four knobs and keeps `mixed_pick: largest` and `zipf_exponent: 1.0` as they were:

| Knob | Before | After |
|---|---|---|
| `label_noise_scope` | all | mixed_vendors |
| `vendors.mixed_share` | 0.12 | 0.06 |
| `learnability.cat1_overlap_factor` | 0.1 | 0.03 |
| `ref_ids.corrupted_vendor_share` | 0.6 | 0.32 |

Measured on the committed sample (seed 42, 6,000 rows):

| Structure | Target | Before | After |
|---|---|---|---|
| Rows from vendors with more than one label | about 0.45 | 0.745 | 0.454 |
| Vendors with one label | about 0.89 | 0.750 | 0.889 |
| Rows with an Excel-damaged ID | about 0.06 | 0.105 | 0.060 |

`mixed_pick: random` did not get closer than `largest`, so the option was removed again rather than left as dead code.

## Consequences

- The vendor structure now matches the private data on these three measures. It does not match on others that were never targets, for example how many lines the largest vendor sends.
- Model results changed with the data. They were measured only after this config was frozen, and are reported as they came out.
- Any future change to these knobs should go through the same procedure: targets first, structure-only measures, then evaluation.

## Round 2: size invariance and rare-class structure

Written before any change in this round, and before any model was trained on the result.

### Problem

At 60,000 rows the round-1 config drifted to 0.664 / 0.857 / 0.045 on the three targets above. Vendor counts grew with the square root of the row count, so each vendor sent more lines as the file grew, and more vendors picked up a second class. The vendor sizes also followed a Zipf curve over all vendors, so the share of rows held by the largest vendors changed with the vendor count.

### Size requirement

The three round-1 measures must stay within ±0.03 of their targets at 6,000, 60,000 and 600,000 rows. To get there:

- Vendor counts grow in proportion to rows, so lines per vendor stay stable.
- Vendor sizes are drawn independently from one fixed distribution, so the share of rows held by the top few percent of vendors does not depend on how many vendors there are.
- The damaged-ID vendor keeps a fixed share of rows (`vendors.largest_share`), because the real data has one such vendor whatever the file size.

### New targets

Measured read-only on the private file (full file after cleaning, time split at the end of the training months, aggregates only):

| Structure | Target | Real count |
|---|---|---|
| Rare-class lines (Cat3, Cat4, Cat6) from vendors with more than one label | about 0.78 | 0.784 of 51 lines |
| Rare-class validation lines whose vendor appears in the training months | about 0.81 | 13 of 16 lines |

On the synthetic side, "validation" is the last `new_vendors.months` months of the main block, the same months the time split scores.

### Knobs that may change in this round

- `vendors.size_sigma` (new): spread of the lognormal vendor-size distribution.
- `vendors.largest_share` (new): share of rows from the damaged-ID vendor.
- `vendors.mixed_share`, `learnability.cat1_overlap_factor`, `ref_ids.corrupted_vendor_share`.
- `learnability.rare_class_overlap` (new): chance a rare-class line comes from a mixed vendor. Cat2 keeps `vendor_class_overlap`.
- `new_vendors.rare_share` (new): chance a rare-class line in the last months comes from a vendor new in that window.

`label_noise`, `vendor_class_overlap`, `description_signal` and `account_class_signal` stay fixed.

### Procedure

The same as round 1. Generate at 6,000 rows (seed 42), read only the five structure measures, pick the closest setting, then check the three size-sensitive measures at 60,000 and 600,000 rows. No model is trained during the search.

### Amendments made during the round-2 search (still before any model run)

Structure-only runs showed two size effects that the knobs above cannot fix. Both changes were made before any model was trained.

- **Cat5 lines borrowed the biggest pure vendor.** Cat5 has no vendors of its own, and its 3 lines fell back to a size-weighted pick of Cat1 vendors. That usually landed on the damaged-ID vendor (18% of rows) and made it count as mixed, which moved the mixed-vendor row share by about 0.19 from one seed to the next. Cat5 lines now come from the mixed vendors, which already send several classes.
- **One fixed-fee vendor at every size.** At 6,000 rows its 24 lines are about a third of all rare-class lines, and at 60,000 rows a few percent. So the rare-class mixed-vendor share fell from about 0.62 to about 0.37 as the file grew. Fixed-fee vendors now scale with rows, one per `fixed_fee.rows_per_vendor` (6,000) lines. They share the same label schedule, as if one labelling-scheme change hit every fixed-fee contract. `fixed_fee.rows_per_vendor` is added to the knobs that may change.
- **Rare-class targets are judged at 60,000 rows.** The 6,000-row sample has only about 8 rare-class lines in its last four months, too few to measure a share. The two rare-class measures are read on a 60,000-row file (seed 43), the three round-1 measures on both sizes.
- **Vendor sizes from quantiles, not random draws.** With a few hundred vendors, random lognormal sizes made the mixed-vendor row share swing by about ±0.06 between seeds at 6,000 rows, so no setting could promise ±0.03 for one particular sample. Sizes are now the lognormal's quantiles at evenly spaced probabilities, assigned to vendors in random order. The size distribution then has the same shape at every file size and every seed. `size_sigma` keeps its meaning.

### Round-2 result

Structure-only searches over `size_sigma` (1.0 to 2.25), `mixed_share` (0.04 to 0.12), `cat1_overlap_factor` (0 to 0.03), `rare_class_overlap` (0.5 to 0.8) and `new_vendors.rare_share` (0.15 to 0.3). No model was trained. The chosen values:

| Knob | Round 1 | Round 2 |
|---|---|---|
| Vendor count | grows with √rows | grows with rows |
| Vendor sizes | Zipf, exponent 1.0 | lognormal quantiles, `size_sigma` 1.5 |
| `vendors.largest_share` (new) | — | 0.18 |
| `vendors.mixed_share` | 0.06 | 0.11 |
| `learnability.cat1_overlap_factor` | 0.03 | 0.0 |
| `learnability.rare_class_overlap` (new) | 0.35, shared with Cat2 | 0.60 |
| `new_vendors.rare_share` (new) | 0 | 0.30 |
| `fixed_fee.rows_per_vendor` (new) | one vendor at any size | 6,000 |
| `ref_ids.corrupted_vendor_share` | 0.32 | 0.32 |

Measured with the final config:

| Structure | Target | 6,000 rows (seed 42) | 60,000 rows (seed 43) | 600,000 rows (seed 44) |
|---|---|---|---|---|
| Rows from vendors with more than one label | 0.45 ± 0.03 | 0.431 | 0.467 | 0.468 |
| Vendors with one label | 0.89 ± 0.03 | 0.893 | 0.886 | 0.889 |
| Rows with an Excel-damaged ID | 0.06 ± 0.03 | 0.060 | 0.0595 | 0.0588 |
| Rare-class lines from multi-label vendors | about 0.78 | 0.724 | 0.767 | 0.761 |
| Rare-class validation lines from a vendor seen earlier | about 0.81 | 0.875 (8 lines) | 0.685 (108 lines) | 0.810 (964 lines) |

The last measure was 0.813 at 60,000 rows before `mixed_share` moved from 0.12 to 0.11 (that move kept the 600,000-row mixed share inside ±0.03). It reads 0.685 now. With about 100 lines it moves a lot with the random draw, and the 600,000-row file, with ten times the lines, sits on target. The search stopped there rather than chase one file's number.

## Generator v1.0 (frozen)

The config in `data/synthetic/config.yaml` at this point is **generator v1.0** (`generator_version: "1.0"` in that file). All published synthetic results use it.

Any future change to the generator code or its config, beyond fixing a bug that leaves the committed sample byte-identical, needs a new ADR. That ADR should give the reason, the targets and the procedure before any value is tried, as this one does, and it should bump the version.
