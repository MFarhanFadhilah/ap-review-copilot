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
