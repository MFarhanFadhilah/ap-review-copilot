"""Transaction-line classifier dashboard.

    streamlit run app/dashboard.py

Tab 1 shows the exploration, cleaning, split, model comparison and hold-out
results; Tab 2 scores an uploaded file and explains any row on request.
All logic comes from ``ml/``: this file only loads, caches and draws.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
import yaml  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402

from ml.agent import from_config  # noqa: E402
from ml.agent.notes import make_note_writer, plain_feature, strength  # noqa: E402
from ml.batch import process_file, process_upload, upload_error  # noqa: E402
from ml.clean import clean_training_data, normalize_label  # noqa: E402
from ml.eda import add_eda_columns, amount_outliers, association_matrix  # noqa: E402
from ml.explain import global_top_features, linear_pipeline  # noqa: E402
from ml.features import add_derived_columns, time_split  # noqa: E402
from ml.predict import format_confidence, load_bundle, read_table  # noqa: E402
from ml.split import HOLDOUT_DIR, load_development_raw  # noqa: E402

CFG = yaml.safe_load(open(ROOT / "config.yaml"))
MODEL_PATH = ROOT / CFG["paths"]["model_dir"] / "model.joblib"
EVAL_DIR = ROOT / CFG["paths"]["model_dir"] / "evaluation"
BLUE, GREY, INK = "#2a78d6", "#a3a29c", "#52514e"
CLASS_COLORS = {"Cat1": GREY, "Cat2": BLUE, "Cat3": "#eb6834", "Cat4": "#2a9d8f", "Cat5": "#9b5de5", "Cat6": "#e9c46a"}
PAGE_SIZE = 25

load_dotenv(ROOT / ".env")


# ---------------------------------------------------------------------------
# Cached loading (all heavy work happens once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the model…")
def get_bundle() -> dict:
    return load_bundle(str(MODEL_PATH))


@st.cache_resource(show_spinner="Starting the review agent…")
def get_agent(api_key: str | None):
    return from_config(CFG, get_bundle(), note_writer=make_note_writer(CFG, api_key))


@st.cache_data(show_spinner="Cleaning the development data…")
def development_data():
    raw = load_development_raw(CFG)
    clean, log, duplicates = clean_training_data(raw)
    return raw, clean, log, duplicates


@st.cache_data(show_spinner="Computing associations…")
def eda_data():
    _, clean, _, _ = development_data()
    eda = add_eda_columns(clean)
    return eda, association_matrix(eda)


@st.cache_data(show_spinner="Scoring the hold-out test set…")
def holdout_result():
    features = read_table(HOLDOUT_DIR / "holdout_features.xlsx")
    labels = pd.read_csv(HOLDOUT_DIR / "holdout_labels.csv")
    return process_upload(features.assign(ClassificationLabel=labels["ClassificationLabel"].to_numpy()), get_bundle())


@st.cache_data
def validation_summary() -> dict | None:
    """Written by `python -m ml.evaluate`; None until that has been run."""
    path = EVAL_DIR / "validation_summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def fixed_fee_vendor(clean: pd.DataFrame) -> str | None:
    """The vendor that bills one amount to one account but whose label changes: labelling drift.

    Found from the data, not hard-coded, so the chart works on any file.
    """
    stats = clean.groupby("Col1").agg(amounts=("Col3", "nunique"), accounts=("Col6", "nunique"),
                                      labels=("label", "nunique"), lines=("label", "size"))
    found = stats[(stats.amounts == 1) & (stats.accounts == 1) & (stats.labels >= 2) & (stats.lines >= 6)]
    return found.sort_values("lines", ascending=False).index[0] if len(found) else None


@st.cache_data(show_spinner="Scoring the file…")
def score_upload(data: bytes, name: str):
    return process_file(io.BytesIO(data), name, get_bundle())


# ---------------------------------------------------------------------------
# Small drawing helpers
# ---------------------------------------------------------------------------
def chart(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), font=dict(size=12),
                      title_font=dict(size=15), plot_bgcolor="rgba(0,0,0,0)", legend_title_text="")
    st.plotly_chart(fig, width="stretch")


def table_height(n_rows: int) -> int:
    """A dataframe height that shows every row, so no fixed height hides any."""
    return 38 + 35 * n_rows


def share_text(share: float) -> str:
    """Shares under 0.1% keep two decimals (e.g. Cat5's 0.04%) instead of rounding to 0.0%."""
    return f"{share:.2%}" if 0 < share < 0.001 else f"{share:.1%}"


def dates_only(frame: pd.DataFrame) -> pd.DataFrame:
    """Show date columns as dates, without a time of day."""
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    return out


def show_messages(messages):
    for level, text in messages:
        {"error": st.error, "warning": st.warning}.get(level, st.info)(text)


def confusion_heatmap(actual, predicted, title):
    labels = sorted(set(actual) | set(predicted))
    cm = confusion_matrix(actual, predicted, labels=labels)
    share = cm / cm.sum(axis=1, keepdims=True)
    fig = go.Figure(go.Heatmap(z=share, x=labels, y=labels, colorscale="Blues", zmin=0, zmax=1, text=cm,
                               texttemplate="%{text}", hovertemplate="actual %{y}, predicted %{x}: %{text}<extra></extra>",
                               showscale=False))
    fig.update_layout(title=title, xaxis_title="predicted", yaxis_title="actual", yaxis_autorange="reversed")
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Transaction line classifier", page_icon="📊", layout="wide")
bundle = get_bundle()
meta = bundle["metadata"]
with st.sidebar:
    st.header("Model")
    st.caption(f"Version `{meta['model_version']}` · trained on {meta['rows_used']:,} development lines "
               f"({meta['periods'][0]} to {meta['periods'][1]})")
    st.metric("Auto-label threshold", f"{bundle['threshold']:.3f}", help=meta.get("threshold_note"))
    st.header("Explanations")
    env_key = os.environ.get("ANTHROPIC_API_KEY")
    typed_key = st.text_input("Anthropic API key", type="password",
                              help="Optional. Read from the environment or a .env file if set there.",
                              placeholder="set in environment" if env_key else "sk-ant-…")
    api_key = typed_key or env_key
    if api_key:
        st.success(f"LLM explanations on ({CFG['agent']['model']}).", icon="✍️")
    else:
        st.info("No API key: explanations use the **offline template**.", icon="📄")
    st.caption("The model and `ml/policy.py` set every label and route. The explanation never changes them.")

st.title("Transaction line classifier")
st.caption("Demo data is **synthetic** (`data/synthetic/`): generated from hand-set parameters, not real invoices.")
tab1, tab2 = st.tabs(["1 · Exploration, preprocessing and model results", "2 · Score a file (hold-out pipeline)"])

# ===========================================================================
# Tab 1
# ===========================================================================
with tab1:
    raw, clean, log, duplicates = development_data()
    eda, V = eda_data()
    st.caption("Everything in this tab uses the **90% development set**, except the hold-out section, "
               "which scores the 10% test set exactly once.")

    st.header("Data overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Development rows (raw)", f"{len(raw):,}")
    c2.metric("Columns", f"{raw.shape[1]}")
    c3.metric("Posting months", f"{pd.to_datetime(raw['Col5']).dt.to_period('M').nunique()}")
    c4.metric("Hold-out rows (set aside)", f"{len(pd.read_csv(HOLDOUT_DIR / 'holdout_labels.csv')):,}")
    profile = pd.DataFrame({"type": raw.dtypes.astype(str), "missing": raw.isna().sum(),
                            "unique values": raw.nunique(), "example": dates_only(raw.head(1)).iloc[0].astype(str)})
    left, right = st.columns([3, 2])
    left.dataframe(profile, width="stretch")
    missing = profile["missing"][profile["missing"] > 0]
    with right:
        if missing.empty:
            st.success("No missing values.")
        else:
            chart(px.bar(missing.rename("missing rows").rename_axis("column").reset_index(), x="column", y="missing rows",
                         title=f"Missing values: {', '.join(missing.index)}", color_discrete_sequence=[BLUE]), 260)

    st.header("Distributions")
    left, right = st.columns(2)
    with left:
        counts = clean["label"].value_counts().rename_axis("class").reset_index(name="lines")
        counts["share"] = (counts["lines"] / counts["lines"].sum()).map(share_text)
        fig = px.bar(counts, x="class", y="lines", text="share", color="class", color_discrete_map=CLASS_COLORS,
                     log_y=True, title=f"Class balance: Cat1 is {counts.iloc[0]['share']} of lines (log scale)")
        fig.update_layout(showlegend=False)
        chart(fig)
    with right:
        amounts = clean["Col3"]
        outlier = amount_outliers(amounts)
        frame = pd.DataFrame({"log10 |amount|": np.log10(amounts.abs().where(amounts != 0)),
                              "point": np.where(outlier, "outlier (beyond 1.5 IQR on the log scale)", "typical")}).dropna()
        fig = px.histogram(frame, x="log10 |amount|", color="point", nbins=60, barmode="overlay",
                           color_discrete_map={"typical": GREY, "outlier (beyond 1.5 IQR on the log scale)": BLUE},
                           title=f"Amounts span {int(np.ceil(frame['log10 |amount|'].max() - frame['log10 |amount|'].min()))} "
                                 f"orders of magnitude; {int(outlier.sum())} outliers marked")
        chart(fig)
        st.caption(f"{int((amounts == 0).sum())} zero amounts are left out (no logarithm); "
                   f"{int((amounts < 0).sum())} credit notes are plotted by absolute value.")

    st.header("Associations and drift")
    left, right = st.columns(2)
    with left:
        fig = px.imshow(V.round(2), text_auto=True, color_continuous_scale="Blues", zmin=0, zmax=1,
                        title="Cramér's V between every pair of fields, including the label")
        fig.update_layout(coloraxis_showscale=False)
        chart(fig, 460)
    with right:
        vendor = fixed_fee_vendor(clean)
        if vendor is not None:
            ff = clean[clean["Col1"] == vendor].sort_values("period")
            drift = ff.groupby([ff["period"].dt.strftime("%Y-%m"), "label"]).size().rename("lines").reset_index()
            sequence = " → ".join(ff["label"][ff["label"] != ff["label"].shift()])
            fig = px.bar(drift, x="period", y="lines", color="label", color_discrete_map=CLASS_COLORS,
                         title=f"Fixed-fee vendor ({ff['Col3'].iloc[0]:,.2f} a month): same invoice, label drifts {sequence}")
            fig.update_xaxes(type="category")
            chart(fig, 460)

    st.header("Cleaning: before and after")
    steps = pd.DataFrame(log.steps)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw rows", f"{log.rows_in:,}")
    c2.metric("Clean rows", f"{log.rows_out:,}", delta=f"{log.rows_out - log.rows_in:,}")
    c3.metric("Duplicate groups", f"{len(duplicates):,}")
    c4.metric("Value of extra copies", f"{((duplicates.n_copies - 1) * duplicates.Col3).sum() / 1e6:,.1f}M")
    left, mid, right = st.columns([2, 2, 3])
    with left:
        st.markdown(f"**Label mapping** ({raw['ClassificationLabel'].nunique()} spellings → "
                    f"{raw['ClassificationLabel'].map(normalize_label).nunique()} classes)")
        spellings = raw["ClassificationLabel"].value_counts().rename_axis("raw label").reset_index(name="rows")
        spellings["clean label"] = spellings["raw label"].map(normalize_label)
        st.dataframe(spellings, hide_index=True, width="stretch")
    with mid:
        st.markdown("**Class counts, raw vs clean**")
        before_after = pd.DataFrame({"raw": raw["ClassificationLabel"].map(normalize_label).value_counts(),
                                     "clean": clean["label"].value_counts()})
        st.dataframe(before_after, width="stretch")
    with right:
        st.markdown("**Cleaning log**")
        st.dataframe(steps, hide_index=True, width="stretch")
    with st.expander("Possible duplicate postings (largest groups)"):
        st.dataframe(dates_only(duplicates.head(20)), hide_index=True, width="stretch")

    st.header("Feature engineering")
    features = linear_pipeline(bundle["model"]).named_steps["features"].named_steps["columns"]
    blocks = pd.Series([n.split("__")[0] for n in features.get_feature_names_out()]).value_counts()
    engineering = pd.DataFrame([
        ("Col1 vendor", "word tokens + the vendor itself", "vendor_tokens, categorical"),
        ("Col2 reference ID", "shape (numeric / alphanumeric / separator / corrupted) and length", "categorical, numeric"),
        ("Col3 amount", "log size, 10 amount bands, credit-note and round-amount flags", "numeric, amount_band, flags"),
        ("Col4 description", "TF-IDF on words and word pairs", "description"),
        ("Col5 period", "not a feature: used for the time split and drift monitoring", "-"),
        ("Col6 account", "full code + account group (first token)", "categorical"),
        ("Col7 document type", "one-hot", "categorical"),
    ], columns=["column", "becomes", "model input block"])
    left, right = st.columns([3, 2])
    left.dataframe(engineering, hide_index=True, width="stretch")
    right.dataframe(blocks.rename("inputs").rename_axis("block").to_frame(), width="stretch")
    right.caption(f"{blocks.sum():,} model inputs in total (`ml/features.py`).")
    with st.expander("One line before and after the derived columns"):
        sample = clean.head(1)
        st.dataframe(add_derived_columns(sample)[["Col2", "ref_shape", "ref_len", "Col3", "log_amount", "is_credit_note",
                                                  "is_round_amount", "Col6", "account_group"]], hide_index=True)

    st.header("Split strategy: train / validation / 10% test")
    train, valid = time_split(clean, CFG["split"]["train_end_period"])
    test_labels = pd.read_csv(HOLDOUT_DIR / "holdout_labels.csv")["label"]
    split_counts = pd.DataFrame({
        f"train (≤ {train['period'].max():%b-%Y}, clean lines)": train["label"].value_counts(),
        f"validation ({valid['period'].min():%b-%Y} to {valid['period'].max():%b-%Y}, clean lines)":
            valid["label"].value_counts(),
        "test (10% hold-out, raw rows)": test_labels.value_counts(),
    }).fillna(0).astype(int)
    split_counts.loc["total"] = split_counts.sum()
    st.dataframe(split_counts, width="stretch", height=table_height(len(split_counts)))
    st.markdown(
        "- **Test (hold-out):** 10% of the raw file, carved out before cleaning, stratified by label, with every "
        "duplicate copy kept on the same side. Scored once, by the final model.\n"
        "- **Train / validation:** the other 90%, split in time. The model learns from the past and is checked on "
        "the four later months, the way it would be used.\n"
        "- **The model and its settings** are fixed in `config.yaml`. Validation and hold-out scores are reported, "
        "never used to choose (see `docs/split_strategy.md`).")
    with st.expander("Full split strategy (docs/split_strategy.md)"):
        st.markdown((ROOT / "docs" / "split_strategy.md").read_text())

    st.header("Hold-out test set (scored once) and time-based validation")
    holdout = holdout_result()
    table = holdout.table
    dedup = table[table["duplicate_of_row"].isna() & table["predicted_label"].notna()]
    m = holdout.metrics["duplicate copies removed"]
    st.markdown("**The random hold-out measures familiar vendors and months; the time-based validation is the "
                "realistic forward-looking estimate.**")
    left, right = st.columns(2, border=True)
    with left:
        st.markdown(f"**Random 10% hold-out** · {m['rows']} lines (duplicate copies removed; {len(table)} as delivered)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{m['accuracy']:.1%}")
        c2.metric("Macro-F1", f"{m['macro_f1']:.3f}")
        c3.metric("Auto-labelled", f"{m['automation_rate']:.1%}", help=f"Cat1 precision of auto-labels: {m['auto_precision']:.2%}")
        c4.metric("Rare lines auto", f"{m['non_cat1_auto']}")
    with right:
        validation = validation_summary()
        if validation is None:
            st.info("Run `python -m ml.evaluate` to add the time-based validation here.")
        else:
            first, last = validation["validation_months"]
            st.markdown(f"**Time-based validation, {first} to {last}** · {validation['lines']:,} lines the model never saw")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{validation['accuracy']:.1%}")
            c2.metric("Macro-F1", f"{validation['macro_f1']:.3f}")
            c3.metric("Auto-labelled", f"{validation['automation_rate']:.1%}",
                      help=f"Cat1 precision of auto-labels: {validation['auto_precision']:.2%}")
            c4.metric("Rare lines auto", f"{validation['non_cat1_auto']}")
    rare = dedup["true_label"].value_counts().drop("Cat1", errors="ignore")
    st.caption("Hold-out: a random 10% of all months, scored by the shipped model, with "
               + ", ".join(f"{n} {c}" for c, n in rare.sort_index().items()) + " lines. "
               f"Validation: a model fitted on the months up to {CFG['split']['train_end_period'][:7]} scores the later months "
               "(threshold set on the training months).")
    left, right = st.columns(2)
    with left:
        chart(confusion_heatmap(dedup["true_label"], dedup["predicted_label"],
                                "Confusion matrix: most errors are Cat1 ↔ Cat2"), 420)
    with right:
        report = pd.DataFrame(classification_report(dedup["true_label"], dedup["predicted_label"],
                                                    output_dict=True, zero_division=0)).T
        st.markdown("**Classification report**")
        st.dataframe(report.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}",
                                          "support": "{:.0f}"}), width="stretch")

    st.header("Global top features per class")
    top = global_top_features(bundle["model"], n=10)
    chosen_class = st.radio("Class", sorted(top["class"].unique()), horizontal=True, key="top_class")
    rows = top[top["class"] == chosen_class].sort_values("weight")
    fig = px.bar(rows, x="weight", y="feature", orientation="h", color_discrete_sequence=[BLUE],
                 title=f"Inputs that push hardest towards {chosen_class} (weights of the shipped model)")
    chart(fig, 420)

# ===========================================================================
# Tab 2
# ===========================================================================
def _on_upload_change():
    """A new, removed or rejected upload always clears the previous results."""
    upload = st.session_state.get("uploader")
    st.session_state.update(source=None, upload_error=None, page=1)
    if upload is None:
        return
    problem = upload_error(upload.name)
    if problem:
        st.session_state["upload_error"] = problem
        return
    st.session_state["source"] = (upload.getvalue(), upload.name)


def _amount_text(value) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "invalid"


@st.dialog("Explanation", width="large")
def explanation_dialog(index: int, table: pd.DataFrame, records: list, missing_columns: list, scored: str):
    row = table.loc[index]
    st.markdown(f"**Row {int(row['row'])}** · vendor `{row['Col1']}` · amount {_amount_text(row['Col3'])}")
    if pd.isna(row["predicted_label"]) or row["predicted_label"] is None:
        st.error("This row was not scored, so there is nothing to explain: " + (row["validation_errors"] or "invalid values") + ".")
        return
    cache = st.session_state.setdefault("explanations", {})
    key = (scored, index, bool(api_key))
    if key not in cache:
        with st.spinner("Gathering evidence and writing the explanation…"):
            cache[key] = get_agent(api_key).explain_line(records[index], missing_columns=tuple(missing_columns))
    explanation = cache[key]
    if explanation["mode"] == "llm":
        st.success(f"Written by {CFG['agent']['model']} from the evidence below. The label and route are the model's.")
    elif explanation["mode"] == "template_fallback":
        st.warning(f"LLM explanation not used ({explanation['guardrail']['fallback_reason']}); "
                   "showing the **offline template** instead.")
    else:
        st.info("**Offline template explanation** (no API key). Built from the same evidence an LLM would get.")
    st.code(explanation["text"], language=None, wrap_lines=True)
    evidence = explanation.get("evidence")
    if evidence:
        reasons = evidence["explanation"]["reasons"] + evidence["explanation"]["against"]
        if reasons:
            frame = pd.DataFrame({
                "feature": [plain_feature(r["feature"], evidence) for r in reasons],
                "size": [r["contribution"] for r in reasons],
                "strength": [strength(r["contribution"]) for r in reasons],
                "direction": ["towards " + explanation["label"] if r["contribution"] > 0 else "towards runner-up" for r in reasons],
            }).sort_values("size")
            fig = px.bar(frame, x="size", y="feature", orientation="h", color="direction", text="strength",
                         color_discrete_map={"towards " + explanation["label"]: BLUE, "towards runner-up": GREY},
                         title="What moved the decision (bar length = strength)")
            fig.update_xaxes(showticklabels=False, title_text="")
            fig.update_traces(hovertemplate="%{y}: %{text}<extra></extra>")
            chart(fig, 320)
        left, right = st.columns(2)
        history = evidence["vendor_history"]
        left.markdown("**Vendor history**")
        left.write("New vendor: no labelled history." if history["status"] == "new vendor" else
                   f"{history['lines']} past lines ({history['first_period']} to {history['last_period']}): "
                   + ", ".join(f"{k} {v}" for k, v in history["label_counts"].items()))
        right.markdown("**Most similar past lines**")
        right.dataframe(pd.DataFrame(evidence["similar"]["similar"]).drop(columns=["similarity"]), hide_index=True,
                        width="stretch")


with tab2:
    st.markdown("Upload a CSV, XLSX or XLS file with columns **Col1 to Col7**. The pipeline runs live: "
                "validate the schema → standardise → predict → route. A label column, if present, is ignored "
                "for prediction and used to report accuracy.")
    left, right = st.columns([3, 1])
    left.file_uploader("File to score", key="uploader", on_change=_on_upload_change,
                       help="CSV, XLSX or XLS with columns Col1 to Col7.")
    if right.button("Use the hold-out sample", key="use_sample", help="Loads data/holdout with its labels."):
        sample = read_table(HOLDOUT_DIR / "holdout_features.xlsx")
        sample["ClassificationLabel"] = pd.read_csv(HOLDOUT_DIR / "holdout_labels.csv")["ClassificationLabel"].to_numpy()
        buffer = io.BytesIO()
        sample.to_excel(buffer, index=False)
        st.session_state.update(source=(buffer.getvalue(), "holdout_sample.xlsx"), upload_error=None, page=1)

    source = st.session_state.get("source")
    result = None
    if st.session_state.get("upload_error"):
        st.error(st.session_state["upload_error"])
    elif source is None:
        st.info("Upload a file, or use the hold-out sample, to start.")
    else:
        data, name = source
        result = score_upload(data, name)
        digest = hashlib.sha256(data).hexdigest()
        if st.session_state.get("scored") != digest:
            st.session_state.update(scored=digest, page=1)
        st.subheader(f"Results for `{name}`")
        show_messages(result.messages)

    if result is not None and result.ok:
        table = result.table
        if result.metrics:
            m = result.metrics["all scored rows"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy (labelled rows)", f"{m['accuracy']:.1%}")
            c2.metric("Macro-F1", f"{m['macro_f1']:.3f}")
            c3.metric("Auto-labelled", f"{m['automation_rate']:.1%}",
                      help=None if m["auto_precision"] is None else f"Cat1 precision of auto-labels: {m['auto_precision']:.2%}")
            c4.metric("Rare lines auto-labelled", f"{m['non_cat1_auto']}")
        copies = int(table["duplicate_of_row"].notna().sum())
        if copies:
            unique_holdout = holdout_result().metrics["duplicate copies removed"]["rows"]
            st.caption(f"This file includes {copies} duplicate copies; Tab 1 reports the hold-out on "
                       f"{unique_holdout} unique lines.")

        route_filter = st.radio("Show", ["all rows", "review only", "auto only", "duplicates only"], horizontal=True,
                                key="route_filter")
        view = {"review only": table[table["route"] == "review"], "auto only": table[table["route"] == "auto"],
                "duplicates only": table[table["duplicate_of_row"].notna()]}.get(route_filter, table)
        pages = max(1, -(-len(view) // PAGE_SIZE))
        st.session_state["page"] = min(max(st.session_state.get("page", 1), 1), pages)
        page = st.number_input(f"Page (of {pages}, {PAGE_SIZE} rows each)", 1, pages, key="page")
        chunk = view.iloc[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]

        widths = [0.5, 2.1, 1.3, 0.7, 1.0, 1.0, 1.0, 2.3, 0.7, 1.0]
        header = st.columns(widths)
        for col, title in zip(header, ["row", "vendor (Col1)", "amount", "label", "confidence", "band", "route",
                                       "review reasons", "dup. of", ""]):
            col.markdown(f"**{title}**")
        with st.container(height=900, border=True):
            for i, r in chunk.iterrows():
                cells = st.columns(widths, vertical_alignment="center")
                cells[0].write(int(r["row"]))
                cells[1].write(str(r["Col1"]))
                cells[2].markdown(f"<div style='white-space:nowrap;text-align:right'>{_amount_text(r['Col3'])}</div>",
                                  unsafe_allow_html=True)
                cells[3].write(r["predicted_label"] or "—")
                cells[4].write(format_confidence(r["confidence"]))
                cells[5].write(r["confidence_band"] or "—")
                cells[6].write(("🟢\u00a0" if r["route"] == "auto" else "🟠\u00a0") + r["route"])
                cells[7].write(r["review_reasons"] or r["validation_errors"] or "")
                cells[8].write("" if pd.isna(r["duplicate_of_row"]) else int(r["duplicate_of_row"]))
                if cells[9].button("Explain", key=f"explain_{i}"):
                    explanation_dialog(i, table, result.records, result.missing_columns, st.session_state["scored"])

        st.download_button("Download predictions (CSV)", table.to_csv(index=False).encode(),
                           file_name=f"predictions_{Path(source[1]).stem}.csv", mime="text/csv", key="download")
