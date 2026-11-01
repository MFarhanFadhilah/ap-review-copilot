# Historical results on the private data

This project was first built and measured on a real, anonymised file of accounts-payable lines. **That file is not public, and it is not in this repo.** It belongs to someone else, and even anonymised tokens and amounts can identify a business. So this page gives aggregate numbers only: counts, shares and scores. It names no vendor, account, description, token or amount.

The code that produced these numbers is the same pipeline as in this repo, before it was ported to synthetic data. The model, features and routing rules are unchanged. Every number below was re-checked against the private file by running that code read-only.

The demo in this repo runs on synthetic data, and its numbers are in the [README](../README.md). Don't compare the two sets of numbers as if they measured the same thing.

## The data

| | |
|---|---|
| Lines | 5,899 |
| Columns | 7 anonymised fields plus a label |
| Classes | 6 (Cat1 to Cat6) |
| Lines after cleaning | 5,219 |
| Class counts after cleaning | Cat1 4,655 (89.2%) · Cat2 511 · Cat3 23 · Cat4 16 · Cat6 12 · Cat5 2 |

## Data-quality findings

| Finding | Count |
|---|---|
| Label spellings for 6 classes | 11 |
| Exact duplicate copies removed | 679, in 536 groups |
| The largest repeated line | 59 copies across 32 accounts in one month, 91% of the value of all extra copies |
| Groups of identical lines with conflicting labels | 1 |
| Reference IDs damaged by Excel (scientific notation, letter scrambled) | 317, 97.8% from one vendor, all Cat1 |
| Legacy lines well before the main block | 38 |
| Cat5 lines | 2, both credit notes |
| Fixed-fee vendor whose label changed over time | 1 vendor, 24 lines, label changed twice |
| Vendors | 313, of which 34 carry more than one label. Those 34 send 45% of lines |
| Document type matching the account group's usual type | 99.7% of lines |
| Date convention | month-start dates in the first part of the timeline, month-end dates after |

The duplicates mattered beyond modelling. Most of their value sat in one line repeated many times, which is worth checking as a possible duplicate payment.

## Model results

The data was split three ways. A 10% hold-out test set was set aside first and scored once. The other 90% was split in time, into training months and four later validation months.

**Validation months (1,299 lines the model never saw)**

| | |
|---|---|
| Macro-F1 | **0.749**, against 0.610 for a vendor-lookup rule |
| Accuracy | 93.5% |
| Auto-labelled | **79.2%** of lines, at **99.13%** Cat1 precision (target 98%) |
| Review queue | 270 lines, about one line in five |
| Non-Cat1 lines auto-labelled by mistake | 9 |

**Hold-out test set (522 lines after removing duplicate copies, 590 as delivered)**

| | |
|---|---|
| Accuracy | **95.6%** |
| Macro-F1 | 0.887 |
| Auto-labelled | 84.9% of lines, at 98.87% Cat1 precision |
| Non-Cat1 lines auto-labelled by mistake | 5 |

The hold-out is a random 10% of all months, with only 2 lines per rare class. So it mostly confirms the Cat1 and Cat2 behaviour. The validation months are the better guide to a future month.

## What carries over to the synthetic demo

The code, the model, the features, the routing policy and the review agent all carry over unchanged. The data quirks above are reproduced on purpose by the synthetic generator ([`data/synthetic/README.md`](../data/synthetic/README.md)), with hand-set rates.

The scores do not carry over. The synthetic data has its own difficulty, set by its learnability knobs.
