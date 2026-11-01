# 1. Run the public repo on synthetic data

**Status:** accepted

## Context

The project was first built on a real, anonymised file of invoice lines. That file is not ours to publish. Anonymised tokens and amounts can still identify a business, and a model trained on the file carries its rows (the bundle stores training lines for the agent). The repo still needs data that runs every part of the pipeline: cleaning, the split, training, routing, the agent and the dashboard.

## Decision

A seeded generator (`data/synthetic/generate.py`) makes all the data. Every parameter is set by hand in `data/synthetic/config.yaml`, and the generator never reads the real file. It reproduces the quirks the pipeline was built to handle (duplicates, label spellings, damaged IDs, a date convention change, labelling drift, new vendors) at configured rates. A 6,000-row sample is committed. Real-data results appear only as aggregates in `docs/historical_results.md`.

Tokens use a new scheme (`Ven`, `Tok`, `Acc`), so a leak check can flag any token from the old scheme without false alarms.

## Alternatives

- **Publish the real file.** Not ours to publish.
- **Perturb the real file** (shuffle, add noise, rename tokens). It is still derived from real rows, and hard to prove safe.
- **A public dataset.** None has this shape (7 anonymised fields, 6 imbalanced classes, these quirks), and the code would need changes to fit.
- **No data, code only.** Nobody could run it, and the tests would need a private file.

## Consequences

- Anyone can run everything, and CI can test it.
- Scores on synthetic data say the pipeline works. They do not say how well it would do on real data. The README keeps the two sets of numbers apart.
- The generator's rules decide what is learnable. Three knobs (`label_noise`, `vendor_class_overlap`, `description_signal`) set the difficulty, and the defaults were not tuned to make the model look good.
- The same generator can make millions of rows, which a later stage uses for benchmarks.
