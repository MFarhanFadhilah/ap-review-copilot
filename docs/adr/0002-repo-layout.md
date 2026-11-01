# 2. Repo layout

**Status:** accepted

## Context

The code started as one `src/` package with a dashboard, an API and notebooks beside it. Later stages add a React front end, Postgres and a retraining job. The layout should make clear what is the model, what serves it and what is data, and it should leave room for those stages.

## Decision

```
ml/              model core: cleaning, features, training, calibration, policy, explanations, agent, evaluation
api/             FastAPI service (becomes backend/ when the React app arrives)
app/             Streamlit dashboard (kept until the React app replaces it)
data/synthetic/  generator, config, measurement and the committed sample
docs/            ADRs, split strategy, historical results
models/          model card. Trained bundles are built, not committed
tests/           pytest suite and fixture files
tools/           leak check, fixture builder, CLI smoke test
```

`src/` became `ml/`, with every import, every `python -m` command and the pickled class path (`ml.calibration.RareClassCalibratedCV`) renamed. `data/` is a package so the generator runs as `python -m data.synthetic.generate`.

The trained model is not committed. Training on the sample takes about 10 seconds, so CI and a new user can rebuild it. A committed pickle would be tied to one scikit-learn version and would change on every retrain.

The executed notebooks were not ported. They were executed on real data, and their builder scripts contain real tokens and a narrative written around real findings.

## Alternatives

- **Keep `src/`.** The name says nothing, and `src/` next to `api/` and `app/` hides that it is the model.
- **One installable package** (`ap_review/ml`, `ap_review/api`, ...). Cleaner for a library, but heavier than this repo needs today.
- **Commit the model.** Faster first run, but a binary that goes stale and can't be reviewed.

## Consequences

- The ML core keeps its behaviour. Only import paths and comments changed, plus the few edits listed in the README.
- A first run needs `python -m ml.split && python -m ml.train` before the API, the dashboard or the model tests can run.
- The dashboard lost two sections (model comparison and tuning), which read files only the notebooks wrote. Porting the model ladder is a later task.
