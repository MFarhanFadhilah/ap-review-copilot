"""End-to-end check that the command line gives the same answers as the library.

    python tools/smoke_test_cli.py

1. Fits the evaluation bundle on the training months only (``ml.train --until-split``).
2. Writes the cleaned validation months to a CSV, as a new file of invoices would arrive.
3. Runs ``python -m ml.predict`` on that file in a separate process.
4. Checks that every label and route in the CLI output equals ``predict_records``
   called in this process on the same lines. The CLI adds file reading and
   writing on top of the library; this proves it adds nothing else.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ml.clean import FEATURE_COLS, clean_training_data  # noqa: E402
from ml.features import time_split  # noqa: E402
from ml.predict import load_bundle, predict_records  # noqa: E402
from ml.split import load_development_raw  # noqa: E402

cfg = yaml.safe_load(open(ROOT / "config.yaml"))
eval_model = ROOT / cfg["paths"]["model_dir"] / "eval_model.joblib"
subprocess.run([sys.executable, "-m", "ml.train", "--until-split", "--output", str(eval_model)],
               cwd=ROOT, check=True, stdout=subprocess.DEVNULL)

df, _, _ = clean_training_data(load_development_raw(cfg))
_, valid = time_split(df, cfg["split"]["train_end_period"])
with tempfile.TemporaryDirectory() as tmp:
    src, out = Path(tmp) / "validation_months.csv", Path(tmp) / "predictions.csv"
    valid[FEATURE_COLS].to_csv(src, index=False)
    subprocess.run([sys.executable, "-m", "ml.predict", "--input", str(src), "--output", str(out),
                    "--model", str(eval_model)], cwd=ROOT, check=True)
    cli = pd.read_csv(out)

library = pd.DataFrame(predict_records(valid[FEATURE_COLS], load_bundle(str(eval_model))))
same = (cli["predicted_label"].to_numpy() == library["predicted_label"].to_numpy()) & \
       (cli["route"].to_numpy() == library["route"].to_numpy())
print(f"\n{len(cli)} lines: CLI and library agree on {int(same.sum())}")
print(cli["route"].value_counts().to_string())
sys.exit(0 if same.all() else "MISMATCH between the CLI and predict_records")
