import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
)

# =========================
# Page config (light theme)
# =========================
st.set_page_config(page_title="Customer Booking Prediction Dashboard", layout="wide")

BG = "#F6F8FC"
CARD = "#FFFFFF"
TEXT = "#111827"
MUTED = "rgba(17,24,39,0.65)"
BORDER = "rgba(15,23,42,0.08)"

st.markdown(
    f"""
<style>
html, body, [data-testid="stAppViewContainer"] {{
    background: {BG};
}}
.block-container {{
    padding-top: 2.6rem;  /* avoids title clipping */
    padding-bottom: 1.5rem;
}}
#MainMenu {{visibility:hidden;}}
footer {{visibility:hidden;}}

section[data-testid="stSidebar"] > div {{
    border-right: 1px solid {BORDER};
}}

.card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 16px 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
}}
.small {{
    color: {MUTED};
    font-size: 12px;
}}
</style>
""",
    unsafe_allow_html=True,
)

def kpi_card(title, value, subtitle=""):
    st.markdown(
        f"""
<div class="card">
  <div style="color:{MUTED}; font-weight:600; font-size:14px;">{title}</div>
  <div style="color:{TEXT}; font-weight:800; font-size:28px; margin-top:6px;">{value}</div>
  <div style="color:{MUTED}; font-size:12px; margin-top:6px;">{subtitle}</div>
</div>
""",
        unsafe_allow_html=True,
    )

# =========================
# Data loading (single-folder project)
# =========================
APP_DIR = Path(__file__).resolve().parent

CANDIDATES = [
    APP_DIR / "customer_booking.csv",
    APP_DIR / "data.csv",
    APP_DIR / "customer_booking_prediction.csv",
]

@st.cache_data(show_spinner=False)
def load_csv_with_fallback(path: Path) -> pd.DataFrame:
    # Common encodings for this dataset
    for enc in ("utf-8", "ISO-8859-1", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)

def resolve_dataset_path() -> Path | None:
    for p in CANDIDATES:
        if p.exists():
            return p
    return None

# =========================
# Domain cleanup helpers
# =========================
DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DAY_MAP = {d: i + 1 for i, d in enumerate(DAY_ORDER)}

def standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # If flight_day is Mon/Tue/etc, keep it as ordered categorical for readability in insights
    if "flight_day" in df.columns:
        if df["flight_day"].dtype == object:
            df["flight_day"] = df["flight_day"].astype(str).str.strip()
            df["flight_day"] = pd.Categorical(df["flight_day"], categories=DAY_ORDER, ordered=True)

    # Ensure target numeric
    if "booking_complete" in df.columns:
        df["booking_complete"] = pd.to_numeric(df["booking_complete"], errors="coerce")

    return df

def detect_target(df: pd.DataFrame) -> str:
    if "booking_complete" in df.columns:
        return "booking_complete"
    for c in df.columns:
        if c.strip().lower() == "booking_complete":
            return c
    return df.columns[-1]

# =========================
# Modeling
# =========================
def build_pipeline(n_estimators: int, max_depth: int | None, random_state: int) -> Pipeline:
    num_sel = selector(dtype_include=np.number)
    cat_sel = selector(dtype_exclude=np.number)

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_sel),
            ("cat", cat_pipe, cat_sel),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    clf = RandomForestClassifier(
        n_estimators=int(n_estimators),
        random_state=int(random_state),
        n_jobs=-1,
        max_depth=max_depth,
    )

    return Pipeline(steps=[("prep", pre), ("model", clf)])

@st.cache_resource(show_spinner=False)
def train_eval_cached(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    random_state: int,
    n_estimators: int,
    max_depth: int | None,
):
    df = df.copy().dropna(axis=0, how="all")

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])

    X = df.drop(columns=[target_col], errors="ignore")
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=y if y.nunique() == 2 else None,
    )

    pipe = build_pipeline(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    proba_test = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe.named_steps["model"], "predict_proba") else None

    acc = accuracy_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, proba_test) if proba_test is not None and y_test.nunique() == 2 else np.nan
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, pred)

    roc = (None, None, None)
    pr = (None, None, None)
    ap = np.nan

    if proba_test is not None and y_test.nunique() == 2:
        fpr, tpr, thr = roc_curve(y_test, proba_test)
        roc = (fpr, tpr, thr)

        prec, rec, thr2 = precision_recall_curve(y_test, proba_test)
        pr = (prec, rec, thr2)
        ap = average_precision_score(y_test, proba_test)

    # Feature importance (post-encoding)
    fi = pd.DataFrame({"feature": [], "importance": []})
    try:
        feature_names = pipe.named_steps["prep"].get_feature_names_out()
        importances = pipe.named_steps["model"].feature_importances_
        fi = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    except Exception:
        pass

    # Probability on full dataset for insight bands (fast enough for 50k)
    proba_all = None
    try:
        if hasattr(pipe.named_steps["model"], "predict_proba"):
            proba_all = pipe.predict_proba(X)[:, 1]
    except Exception:
        proba_all = None

    return pipe, X, y, X_test, y_test, pred, proba_test, proba_all, acc, roc_auc, report, cm, fi, roc, pr, ap

# =========================
# Insights helpers
# =========================
def completion_rate(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    return float((s > 0).mean())

def rate_table_by_category(df: pd.DataFrame, target: str, col: str, min_n: int, top_n: int = 10) -> pd.DataFrame:
    base = completion_rate(df[target])
    g = (
        df.groupby(col, dropna=False)[target]
        .agg(n="size", completion_rate=lambda x: float((pd.to_numeric(x, errors="coerce").fillna(0) > 0).mean()))
        .reset_index()
    )
    g["lift_vs_overall"] = g["completion_rate"] - base
    g = g[g["n"] >= min_n].copy()

    # Tidy category values for display
    g[col] = g[col].astype(str)

    g = g.sort_values(["lift_vs_overall", "n"], ascending=[False, False]).head(top_n)
    return g, base

def numeric_bins_rate(df: pd.DataFrame, target: str, col: str, bins: int = 8) -> pd.DataFrame:
    d = df[[col, target]].copy()
    d[col] = pd.to_numeric(d[col], errors="coerce")
    d[target] = pd.to_numeric(d[target], errors="coerce")
    d = d.dropna(subset=[col, target])
    if len(d) == 0:
        return pd.DataFrame()

    # Quantile bins are stable for skewed distributions
    try:
        d["bin"] = pd.qcut(d[col], q=bins, duplicates="drop")
    except ValueError:
        d["bin"] = pd.cut(d[col], bins=bins)

    out = (
        d.groupby("bin", observed=True)
        .agg(
            n=(target, "size"),
            completion_rate=(target, lambda x: float((x > 0).mean())),
            value_min=(col, "min"),
            value_max=(col, "max"),
            value_median=(col, "median"),
        )
        .reset_index(drop=True)
    )
    return out

def probability_band(p: float, low: float = 0.40, high: float = 0.70) -> str:
    if not np.isfinite(p):
        return "Unknown"
    if p < low:
        return "Low"
    if p <= high:
        return "Medium"
    return "High"

# =========================
# Load dataset automatically
# =========================
path = resolve_dataset_path()
if path is None:
    st.error(
        "Dataset file not found. Place customer_booking.csv (recommended) or data.csv in the same folder as app.py."
    )
    st.stop()

raw = load_csv_with_fallback(path)
df = standardize_df(raw)
target_default = detect_target(df)

# =========================
# Header
# =========================
st.markdown("## Customer Booking Prediction Dashboard")
st.caption("An interactive dashboard for understanding booking completion drivers and evaluating a Random Forest classifier.")
st.write("")

# =========================
# Sidebar
# =========================
st.sidebar.title("Controls")

page = st.sidebar.radio(
    "Navigate",
    ["Dashboard Summary", "Data Overview", "Exploratory Analysis", "Model Performance", "Insights", "Prediction Tool"],
    index=0,
)

st.sidebar.divider()
st.sidebar.subheader("Target")
target_col = st.sidebar.selectbox(
    "Target column",
    options=df.columns.tolist(),
    index=df.columns.tolist().index(target_default),
)

st.sidebar.divider()
st.sidebar.subheader("Model settings")
test_size = st.sidebar.slider("Test split", 0.1, 0.4, 0.2, step=0.05)
random_state = st.sidebar.number_input("Random state", min_value=0, max_value=10_000, value=42)
n_estimators = st.sidebar.slider("Trees (n_estimators)", 50, 600, 200, step=50)
depth_choice = st.sidebar.selectbox("Max depth", ["None", "10", "20", "30"], index=0)
max_depth = None if depth_choice == "None" else int(depth_choice)

threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.50, step=0.05)

st.sidebar.divider()
st.sidebar.subheader("Insights settings")
low_band = st.sidebar.slider("Low probability cutoff", 0.10, 0.60, 0.40, step=0.05)
high_band = st.sidebar.slider("High probability cutoff", 0.50, 0.95, 0.70, step=0.05)
min_group_n = st.sidebar.slider("Minimum group size for category insights", 50, 1000, 200, step=50)

run_train = st.sidebar.button("Train / Refresh", type="primary")
st.sidebar.caption(f"Data source: {path.name}")

# =========================
# KPIs
# =========================
n_rows, n_cols = df.shape
base_rate = completion_rate(df[target_col]) * 100 if pd.to_numeric(df[target_col], errors="coerce").notna().any() else np.nan

k1, k2, k3, k4 = st.columns(4, gap="large")
with k1:
    kpi_card("Rows", f"{n_rows:,}", "Records")
with k2:
    kpi_card("Columns", f"{n_cols:,}", "Features + target")
with k3:
    kpi_card("Booking completion rate", f"{base_rate:.2f}%" if np.isfinite(base_rate) else "—", "Share of completed bookings")
with k4:
    kpi_card("Target", target_col, "Prediction target")

st.write("")

# =========================
# Pages
# =========================
if page == "Dashboard Summary":
    left, right = st.columns([1.25, 1.0], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Project Summary")
        st.write(
            """
This project analyses customer booking behaviour and predicts whether a booking is completed.
The dashboard provides:
- Dataset profile and booking completion rate
- Exploratory analysis of patterns and distributions
- Model evaluation (confusion matrix, classification report, ROC and precision-recall)
- An insights section with actionable patterns grounded in the data and model outputs
- A prediction tool to test scenarios by adjusting inputs
            """.strip()
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Data Overview")
        st.dataframe(df.head(30), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Data Overview":
    st.markdown("### Data Overview")

    a, b = st.columns([1.0, 1.0], gap="large")

    with a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Columns and types")
        schema = pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes]})
        st.dataframe(schema, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Missing values (top columns)")
        miss = df.isna().sum().sort_values(ascending=False)
        miss = miss[miss > 0].head(15)
        if len(miss) == 0:
            st.success("No missing values detected.")
        else:
            mdf = miss.reset_index()
            mdf.columns = ["column", "missing_count"]
            figm = px.bar(mdf, x="missing_count", y="column", orientation="h")
            figm.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(figm, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Full preview")
    st.dataframe(df.head(200), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Exploratory Analysis":
    st.markdown("### Exploratory Analysis")

    a, b = st.columns([1.1, 1.0], gap="large")

    with a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Booking completion distribution")
        y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)
        dist = pd.DataFrame({"Booking completion": y.map({0: "Not completed", 1: "Completed"})})
        fig = px.histogram(dist, x="Booking completion")
        fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Numeric feature explorer")
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != target_col]
        if not num_cols:
            st.info("No numeric features detected.")
        else:
            col = st.selectbox("Select a numeric feature", num_cols)
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            figd = px.histogram(s, nbins=40, labels={"value": col})
            figd.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(figd, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Correlation heatmap (numeric features)")
    num_df = df.select_dtypes(include=[np.number]).copy()
    if num_df.shape[1] < 2:
        st.info("Not enough numeric columns for correlation.")
    else:
        corr = num_df.corr(numeric_only=True)
        figc = px.imshow(corr, aspect="auto")
        figc.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(figc, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Model Performance":
    st.markdown("### Model Performance")

    if not run_train:
        st.info("Use the Train / Refresh button in the sidebar to compute model metrics.")
        st.stop()

    with st.spinner("Training model..."):
        pipe, X_all, y_all, X_test, y_test, pred, proba_test, proba_all, acc, roc_auc, report, cm, fi, roc, pr, ap = train_eval_cached(
            df=df,
            target_col=target_col,
            test_size=float(test_size),
            random_state=int(random_state),
            n_estimators=int(n_estimators),
            max_depth=max_depth,
        )

    rep_df = pd.DataFrame(report).T

    m1, m2, m3, m4 = st.columns(4, gap="large")
    with m1:
        kpi_card("Accuracy", f"{acc:.3f}" if np.isfinite(acc) else "—", "Overall correctness")
    with m2:
        kpi_card("ROC-AUC", f"{roc_auc:.3f}" if np.isfinite(roc_auc) else "—", "Ranking quality")
    with m3:
        kpi_card("Average precision", f"{ap:.3f}" if np.isfinite(ap) else "—", "Precision-recall summary")
    with m4:
        kpi_card("Test rows", f"{len(y_test):,}", "Holdout size")

    st.write("")
    left, right = st.columns([1.2, 1.0], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Confusion matrix")
        cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
        fig_cm = px.imshow(cm_df, text_auto=True, aspect="auto")
        fig_cm.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_cm, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Classification report")
        show_cols = ["precision", "recall", "f1-score", "support"]
        rep_show = rep_df[[c for c in show_cols if c in rep_df.columns]].copy()
        st.dataframe(rep_show, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    a, b = st.columns([1.0, 1.0], gap="large")

    with a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("ROC curve")
        if roc[0] is None:
            st.info("ROC curve not available (requires binary target and probabilities).")
        else:
            fpr, tpr, _ = roc
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name="ROC"))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Baseline", mode="lines"))
            fig_roc.update_layout(
                height=360,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="False positive rate",
                yaxis_title="True positive rate",
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Feature importance")
        if fi.empty:
            st.info("Feature importance could not be extracted.")
        else:
            topn = st.slider("Show top N", 5, 30, 12)
            fi_top = fi.head(topn).iloc[::-1]
            fig_fi = px.bar(fi_top, x="importance", y="feature", orientation="h")
            fig_fi.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_fi, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Insights":
    st.markdown("### Insights")

    if not run_train:
        st.info("Use the Train / Refresh button in the sidebar to generate model-driven insights.")
        st.stop()

    with st.spinner("Preparing insights..."):
        pipe, X_all, y_all, X_test, y_test, pred, proba_test, proba_all, acc, roc_auc, report, cm, fi, roc, pr, ap = train_eval_cached(
            df=df,
            target_col=target_col,
            test_size=float(test_size),
            random_state=int(random_state),
            n_estimators=int(n_estimators),
            max_depth=max_depth,
        )

    overall = completion_rate(df[target_col])
    overall_pct = overall * 100 if np.isfinite(overall) else np.nan

    # Probability bands (using full-dataset probabilities)
    band_df = None
    if proba_all is not None and len(proba_all) == len(df):
        band_df = df[[target_col]].copy()
        band_df["predicted_probability"] = proba_all
        band_df["probability_band"] = [probability_band(p, low=low_band, high=high_band) for p in proba_all]

    # Top drivers (model importance)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Summary metrics")
    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        kpi_card("Overall completion rate", f"{overall_pct:.2f}%" if np.isfinite(overall_pct) else "—", "Across the full dataset")
    with c2:
        kpi_card("Model ROC-AUC", f"{roc_auc:.3f}" if np.isfinite(roc_auc) else "—", "Model ranking quality on the test split")
    with c3:
        kpi_card("Model accuracy", f"{acc:.3f}" if np.isfinite(acc) else "—", "Correct classification on the test split")
    with c4:
        kpi_card("Insights group threshold", f"{min_group_n:,}", "Minimum group size used for categorical insights")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # Probability band distribution
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Predicted probability distribution")
    st.caption("Probability bands are computed from the model output for each record.")
    if band_df is None:
        st.info("Probability distribution is not available (model probabilities could not be computed).")
    else:
        band_counts = (
            band_df["probability_band"].value_counts()
            .reindex(["Low", "Medium", "High", "Unknown"])
            .dropna()
            .reset_index()
        )
        band_counts.columns = ["band", "count"]
        band_counts["share"] = band_counts["count"] / band_counts["count"].sum()

        figb = px.bar(band_counts, x="band", y="share", labels={"band": "Probability band", "share": "Share of records"})
        figb.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(figb, use_container_width=True)

        st.markdown(
            f"<div class='small'>Band cutoffs: Low &lt; {low_band:.2f}, Medium {low_band:.2f}–{high_band:.2f}, High &gt; {high_band:.2f}.</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # Categorical actionable segments
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("High- and low-performing segments (completion rate vs overall)")
    st.caption("Completion rate differences are associations observed in the data. They do not imply causation.")

    cat_cols = [c for c in ["sales_channel", "trip_type", "flight_day"] if c in df.columns]
    hi_cols = st.columns(len(cat_cols)) if len(cat_cols) > 0 else []

    for i, col in enumerate(cat_cols):
        tab, base = rate_table_by_category(df, target_col, col, min_n=min_group_n, top_n=8)
        with hi_cols[i]:
            st.markdown(f"**{col}**")
            if tab.empty:
                st.write("No groups meet the minimum sample size.")
            else:
                show = tab.copy()
                show["completion_rate"] = (show["completion_rate"] * 100).round(2)
                show["lift_vs_overall"] = (show["lift_vs_overall"] * 100).round(2)
                show = show.rename(columns={"completion_rate": "completion_rate_pct", "lift_vs_overall": "lift_pct_points"})
                st.dataframe(show[[col, "n", "completion_rate_pct", "lift_pct_points"]], use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # High-cardinality categories (route, booking_origin)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Route and origin patterns (filtered for reliable sample sizes)")
    st.caption("These columns can contain many categories. Results below apply the minimum group size threshold.")

    c1, c2 = st.columns(2, gap="large")

    if "route" in df.columns:
        with c1:
            st.markdown("**route**")
            rt, _ = rate_table_by_category(df, target_col, "route", min_n=min_group_n, top_n=10)
            if rt.empty:
                st.write("No routes meet the minimum sample size.")
            else:
                show = rt.copy()
                show["completion_rate"] = (show["completion_rate"] * 100).round(2)
                show["lift_vs_overall"] = (show["lift_vs_overall"] * 100).round(2)
                show = show.rename(columns={"completion_rate": "completion_rate_pct", "lift_vs_overall": "lift_pct_points"})
                st.dataframe(show[["route", "n", "completion_rate_pct", "lift_pct_points"]], use_container_width=True, hide_index=True)

    if "booking_origin" in df.columns:
        with c2:
            st.markdown("**booking_origin**")
            og, _ = rate_table_by_category(df, target_col, "booking_origin", min_n=min_group_n, top_n=10)
            if og.empty:
                st.write("No origins meet the minimum sample size.")
            else:
                show = og.copy()
                show["completion_rate"] = (show["completion_rate"] * 100).round(2)
                show["lift_vs_overall"] = (show["lift_vs_overall"] * 100).round(2)
                show = show.rename(columns={"completion_rate": "completion_rate_pct", "lift_vs_overall": "lift_pct_points"})
                st.dataframe(show[["booking_origin", "n", "completion_rate_pct", "lift_pct_points"]], use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # Numeric drivers (binned completion rates)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Numeric drivers (completion rate by value range)")
    st.caption("Numeric features are grouped into bins and evaluated by observed completion rate.")

    num_candidates = [c for c in ["purchase_lead", "length_of_stay", "flight_duration", "flight_hour", "num_passengers"] if c in df.columns]
    num_col = st.selectbox("Select a numeric feature", num_candidates, index=0 if num_candidates else 0)

    if not num_candidates:
        st.info("No expected numeric features were found.")
    else:
        nbins = st.slider("Number of bins", 5, 12, 8)
        bt = numeric_bins_rate(df, target_col, num_col, bins=nbins)
        if bt.empty:
            st.write("Not enough valid values to compute binned rates.")
        else:
            bt_show = bt.copy()
            bt_show["completion_rate_pct"] = (bt_show["completion_rate"] * 100).round(2)
            bt_show["range"] = bt_show.apply(lambda r: f"{r['value_min']:.0f} to {r['value_max']:.0f}", axis=1)

            fig = px.line(
                bt_show,
                x="value_median",
                y="completion_rate_pct",
                markers=True,
                labels={"value_median": "Bin median value", "completion_rate_pct": "Completion rate (%)"},
            )
            fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

            top = bt_show.sort_values("completion_rate_pct", ascending=False).head(1).iloc[0]
            low = bt_show.sort_values("completion_rate_pct", ascending=True).head(1).iloc[0]

            st.write(
                f"Highest observed bin: {top['range']} (n={int(top['n'])}, completion rate={top['completion_rate_pct']:.2f}%). "
                f"Lowest observed bin: {low['range']} (n={int(low['n'])}, completion rate={low['completion_rate_pct']:.2f}%)."
            )

            st.dataframe(
                bt_show[["range", "n", "completion_rate_pct"]].rename(columns={"range": "value_range"}),
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # Model top drivers + recommendations (grounded)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Top model drivers and recommended focus areas")
    st.caption("Recommendations below translate measurable patterns into practical focus areas. They are phrased as associations, not causation.")

    recs = []

    # Feature importance summary
    if fi is not None and not fi.empty:
        top_fi = fi.head(12).copy()
        fig_fi = px.bar(top_fi.iloc[::-1], x="importance", y="feature", orientation="h")
        fig_fi.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_fi, use_container_width=True)

    # Simple, data-grounded recommendation rules
    # sales_channel
    if "sales_channel" in df.columns:
        tab, base = rate_table_by_category(df, target_col, "sales_channel", min_n=min_group_n, top_n=20)
        if not tab.empty:
            best = tab.iloc[0]
            recs.append(
                f"Channel performance: {best['sales_channel']} has completion rate {(best['completion_rate']*100):.2f}% "
                f"(lift {(best['lift_vs_overall']*100):+.2f} percentage points vs overall). "
                f"Focus on reducing friction in lower-performing channels."
            )

    # trip_type
    if "trip_type" in df.columns:
        tab, base = rate_table_by_category(df, target_col, "trip_type", min_n=min_group_n, top_n=20)
        if not tab.empty:
            best = tab.iloc[0]
            worst = tab.sort_values("lift_vs_overall", ascending=True).iloc[0]
            recs.append(
                f"Trip type gap: best={best['trip_type']} at {(best['completion_rate']*100):.2f}% vs "
                f"worst={worst['trip_type']} at {(worst['completion_rate']*100):.2f}%. "
                f"Review messaging and checkout steps for lower-performing trip types."
            )

    # wants_* flags
    wants_cols = [c for c in ["wants_extra_baggage", "wants_preferred_seat", "wants_in_flight_meals"] if c in df.columns]
    for wc in wants_cols:
        tmp = df[[wc, target_col]].copy()
        tmp[wc] = pd.to_numeric(tmp[wc], errors="coerce")
        tmp[target_col] = pd.to_numeric(tmp[target_col], errors="coerce")
        tmp = tmp.dropna()
        if tmp.empty:
            continue
        g = tmp.groupby(wc)[target_col].agg(n="size", rate=lambda x: float((x > 0).mean())).reset_index()
        if set(g[wc].unique().tolist()) >= {0, 1}:
            r0 = float(g.loc[g[wc] == 0, "rate"].values[0])
            r1 = float(g.loc[g[wc] == 1, "rate"].values[0])
            lift_pp = (r1 - r0) * 100
            recs.append(
                f"Add-on indicator ({wc}): rate when 1 is {r1*100:.2f}% vs {r0*100:.2f}% when 0 "
                f"(difference {lift_pp:+.2f} percentage points). Consider how add-ons are presented in the booking flow."
            )

    if len(recs) == 0:
        st.write("No recommendations could be generated for the available columns.")
    else:
        for r in recs[:6]:
            st.write(f"- {r}")

    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Prediction Tool":
    st.markdown("### Prediction Tool")
    st.caption("Adjust inputs and view the predicted probability of booking completion.")

    if not run_train:
        st.info("Use the Train / Refresh button in the sidebar before using the prediction tool.")
        st.stop()

    with st.spinner("Training model..."):
        pipe, X_all, y_all, X_test, y_test, pred, proba_test, proba_all, acc, roc_auc, report, cm, fi, roc, pr, ap = train_eval_cached(
            df=df,
            target_col=target_col,
            test_size=float(test_size),
            random_state=int(random_state),
            n_estimators=int(n_estimators),
            max_depth=max_depth,
        )

    X = df.drop(columns=[target_col], errors="ignore").copy()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Input selection")

    input_row = {}
    cols = st.columns(3, gap="large")

    for i, col in enumerate(X.columns):
        box = cols[i % 3]
        s = X[col]

        if pd.api.types.is_numeric_dtype(s):
            v = pd.to_numeric(s, errors="coerce").dropna()
            if len(v) == 0:
                input_row[col] = 0.0
                continue
            vmin = float(v.quantile(0.01))
            vmax = float(v.quantile(0.99))
            vmed = float(v.median())
            input_row[col] = box.slider(col, min_value=vmin, max_value=vmax, value=vmed)
        else:
            vals = s.dropna().astype(str).unique().tolist()
            vals = sorted(vals)[:250] if len(vals) > 250 else sorted(vals)
            input_row[col] = box.selectbox(col, vals) if vals else ""

    st.markdown("</div>", unsafe_allow_html=True)

    X_new = pd.DataFrame([input_row])

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Prediction result")

    if hasattr(pipe.named_steps["model"], "predict_proba"):
        p = float(pipe.predict_proba(X_new)[:, 1][0])
        decision = 1 if p >= threshold else 0

        band = probability_band(p, low=low_band, high=high_band)

        st.success(f"Predicted booking completion probability: {p:.2%}")
        st.write(f"Probability band: {band}")
        st.write(f"Decision threshold: {threshold:.2f}")
        st.write(f"Predicted class: {'Completed' if decision == 1 else 'Not completed'}")
        st.markdown(
            "<div class='small'>The probability is model-based. Use it to compare scenarios, not as a guarantee of outcomes.</div>",
            unsafe_allow_html=True,
        )
    else:
        pred_cls = int(pipe.predict(X_new)[0])
        st.success(f"Predicted class: {pred_cls}")

    st.markdown("</div>", unsafe_allow_html=True)
