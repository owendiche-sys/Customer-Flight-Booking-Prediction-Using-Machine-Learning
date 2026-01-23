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


def _dedupe_columns(cols) -> list[str]:
    """
    Fixes the TypeError you saw when selecting num_passengers.
    If a column name appears multiple times, df[col] returns a DataFrame (not a Series).
    This makes names unique by appending __dup1, __dup2, ...
    """
    seen = {}
    out = []
    for c in cols:
        c = str(c).strip()
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}__dup{seen[c]}")
    return out


def standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Clean column names + ensure uniqueness (prevents df["num_passengers"] becoming a DataFrame)
    df.columns = _dedupe_columns(df.columns)

    # Tidy string columns
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    # flight_day as ordered categorical (if present)
    if "flight_day" in df.columns:
        df["flight_day"] = pd.Categorical(df["flight_day"], categories=DAY_ORDER, ordered=True)

    # Ensure target numeric if present
    if "booking_complete" in df.columns:
        df["booking_complete"] = pd.to_numeric(df["booking_complete"], errors="coerce")

    # Common binary flags: coerce where possible
    for c in ["wants_extra_baggage", "wants_preferred_seat", "wants_in_flight_meals"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

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

    num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
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

    # Probabilities for full dataset (used for probability bands)
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


def rate_table_by_category(
    df: pd.DataFrame, target: str, col: str, min_n: int, top_n: int = 10
) -> tuple[pd.DataFrame, float]:
    base = completion_rate(df[target])
    g = (
        df.groupby(col, dropna=False)[target]
        .agg(
            n="size",
            completion_rate=lambda x: float((pd.to_numeric(x, errors="coerce").fillna(0) > 0).mean()),
        )
        .reset_index()
    )
    g["lift_vs_overall"] = g["completion_rate"] - base
    g = g[g["n"] >= min_n].copy()
    g[col] = g[col].astype(str)
    g = g.sort_values(["lift_vs_overall", "n"], ascending=[False, False]).head(top_n)
    return g, base


def numeric_bins_rate(df: pd.DataFrame, target: str, col: str, bins: int = 8) -> pd.DataFrame:
    """
    Robust to duplicate column names:
    if df[col] returns a DataFrame, use the first column as the Series.
    """
    if col not in df.columns or target not in df.columns:
        return pd.DataFrame()

    x = df[col]
    y = df[target]

    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    d = pd.DataFrame({col: x, target: y}).copy()
    d[col] = pd.to_numeric(d[col], errors="coerce")
    d[target] = pd.to_numeric(d[target], errors="coerce")
    d = d.dropna(subset=[col, target])
    if len(d) == 0:
        return pd.DataFrame()

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


def safe_str(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return str(x)


# =========================
# Load dataset automatically
# =========================
path = resolve_dataset_path()
if path is None:
    st.error("Dataset file not found. Place customer_booking.csv (recommended) or data.csv in the same folder as app.py.")
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
    ["Dashboard Summary", "Data Overview", "Exploratory Analysis", "Insights", "Model Performance", "Prediction Tool"],
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
# KPIs (shown on all pages, intentionally)
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
- Insights that combine observed patterns and model outputs
- Model evaluation (confusion matrix, classification report, ROC and precision-recall)
- A prediction tool to test scenarios by adjusting inputs
            """.strip()
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Data preview")
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

elif page == "Insights":
    st.markdown("### Insights")
    st.caption("Insights are structured as: (1) data-driven insights, then (2) model-driven insights.")

    overall = completion_rate(df[target_col])
    overall_pct = overall * 100 if np.isfinite(overall) else np.nan

    # -------------------------
    # 1) Data-driven insights
    # -------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Data-driven insights")
    st.caption("These insights are computed directly from the dataset (no model required).")

    # Quick segment scans for common categorical columns
    cat_cols_primary = [c for c in ["sales_channel", "trip_type", "flight_day"] if c in df.columns]

    best_segment = "—"
    worst_segment = "—"
    best_lift_pp = np.nan
    worst_lift_pp = np.nan

    # Find strongest single segment across primary categorical cols
    for ccol in cat_cols_primary:
        tab, base = rate_table_by_category(df, target_col, ccol, min_n=min_group_n, top_n=50)
        if tab.empty:
            continue
        top = tab.iloc[0]
        bot = tab.sort_values("lift_vs_overall", ascending=True).iloc[0]
        if not np.isfinite(best_lift_pp) or float(top["lift_vs_overall"]) > best_lift_pp:
            best_lift_pp = float(top["lift_vs_overall"])
            best_segment = f"{ccol} = {safe_str(top[ccol])}"
        if not np.isfinite(worst_lift_pp) or float(bot["lift_vs_overall"]) < worst_lift_pp:
            worst_lift_pp = float(bot["lift_vs_overall"])
            worst_segment = f"{ccol} = {safe_str(bot[ccol])}"

    # Numeric: pick default numeric for a data-driven bin insight
    numeric_defaults = [c for c in ["purchase_lead", "length_of_stay", "flight_duration", "flight_hour", "num_passengers"] if c in df.columns]
    default_num = numeric_defaults[0] if numeric_defaults else None
    best_bin_text = "—"
    low_bin_text = "—"

    if default_num is not None:
        bt0 = numeric_bins_rate(df, target_col, default_num, bins=8)
        if not bt0.empty:
            bt_show0 = bt0.copy()
            bt_show0["completion_rate_pct"] = (bt_show0["completion_rate"] * 100)
            top0 = bt_show0.sort_values("completion_rate_pct", ascending=False).head(1).iloc[0]
            low0 = bt_show0.sort_values("completion_rate_pct", ascending=True).head(1).iloc[0]
            best_bin_text = f"{default_num}: {top0['value_min']:.0f} to {top0['value_max']:.0f} (rate {top0['completion_rate_pct']:.2f}%, n={int(top0['n'])})"
            low_bin_text = f"{default_num}: {low0['value_min']:.0f} to {low0['value_max']:.0f} (rate {low0['completion_rate_pct']:.2f}%, n={int(low0['n'])})"

    d1, d2, d3, d4 = st.columns(4, gap="large")
    with d1:
        kpi_card("Overall completion rate", f"{overall_pct:.2f}%" if np.isfinite(overall_pct) else "—", "Across the full dataset")
    with d2:
        kpi_card("Best observed segment", best_segment, f"Lift {best_lift_pp*100:+.2f} pp" if np.isfinite(best_lift_pp) else "No segment found")
    with d3:
        kpi_card("Lowest observed segment", worst_segment, f"Lift {worst_lift_pp*100:+.2f} pp" if np.isfinite(worst_lift_pp) else "No segment found")
    with d4:
        kpi_card("Numeric range signal", default_num if default_num else "—", "Range-based completion differences")

    st.write("")
    c1, c2 = st.columns([1.1, 1.0], gap="large")

    with c1:
        st.subheader("Category segments (completion rate vs overall)")
        if not cat_cols_primary:
            st.info("No expected categorical columns were found (sales_channel, trip_type, flight_day).")
        else:
            for ccol in cat_cols_primary:
                tab, base = rate_table_by_category(df, target_col, ccol, min_n=min_group_n, top_n=8)
                if tab.empty:
                    st.write(f"{ccol}: no groups meet the minimum sample size.")
                    continue
                show = tab.copy()
                show["completion_rate_pct"] = (show["completion_rate"] * 100).round(2)
                show["lift_pct_points"] = (show["lift_vs_overall"] * 100).round(2)
                show = show[[ccol, "n", "completion_rate_pct", "lift_pct_points"]]
                st.markdown(f"**{ccol}**")
                st.dataframe(show, use_container_width=True, hide_index=True)

    with c2:
        st.subheader("Numeric ranges (observed completion rate)")
        if default_num is None:
            st.info("No expected numeric columns were found.")
        else:
            st.write(f"Best range: {best_bin_text}")
            st.write(f"Lowest range: {low_bin_text}")

            nbins0 = st.slider("Bins for the default numeric view", 5, 12, 8)
            bt = numeric_bins_rate(df, target_col, default_num, bins=nbins0)
            if bt.empty:
                st.write("Not enough valid values to compute binned rates.")
            else:
                bt_show = bt.copy()
                bt_show["completion_rate_pct"] = (bt_show["completion_rate"] * 100).round(2)

                fig = px.line(
                    bt_show,
                    x="value_median",
                    y="completion_rate_pct",
                    markers=True,
                    labels={"value_median": "Bin median value", "completion_rate_pct": "Completion rate (%)"},
                )
                fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # -------------------------
    # 2) Model-driven insights
    # -------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Model-driven insights")
    st.caption("These insights require training the model. They help explain predicted probability bands and key drivers.")

    if not run_train:
        st.info("Use the Train / Refresh button in the sidebar to generate model-driven insights.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    with st.spinner("Preparing model-driven insights..."):
        pipe, X_all, y_all, X_test, y_test, pred, proba_test, proba_all, acc, roc_auc, report, cm, fi, roc, pr, ap = train_eval_cached(
            df=df,
            target_col=target_col,
            test_size=float(test_size),
            random_state=int(random_state),
            n_estimators=int(n_estimators),
            max_depth=max_depth,
        )

    m1, m2, m3, m4 = st.columns(4, gap="large")
    with m1:
        kpi_card("Model accuracy", f"{acc:.3f}" if np.isfinite(acc) else "—", "Test split")
    with m2:
        kpi_card("Model ROC-AUC", f"{roc_auc:.3f}" if np.isfinite(roc_auc) else "—", "Ranking quality")
    with m3:
        kpi_card("Average precision", f"{ap:.3f}" if np.isfinite(ap) else "—", "Precision-recall summary")
    with m4:
        kpi_card("Group threshold", f"{min_group_n:,}", "Minimum group size used")

    st.write("")

    # Probability bands
    st.subheader("Predicted probability distribution")
    if proba_all is None or len(proba_all) != len(df):
        st.info("Probability distribution is not available (model probabilities could not be computed).")
    else:
        band_df = df[[target_col]].copy()
        band_df["predicted_probability"] = proba_all
        band_df["probability_band"] = [probability_band(p, low=low_band, high=high_band) for p in proba_all]

        band_counts = (
            band_df["probability_band"]
            .value_counts()
            .reindex(["Low", "Medium", "High", "Unknown"])
            .dropna()
            .reset_index()
        )
        band_counts.columns = ["band", "count"]
        band_counts["share"] = band_counts["count"] / band_counts["count"].sum()

        figb = px.bar(
            band_counts,
            x="band",
            y="share",
            labels={"band": "Probability band", "share": "Share of records"},
        )
        figb.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(figb, use_container_width=True)

        st.markdown(
            f"<div class='small'>Band cutoffs: Low &lt; {low_band:.2f}, Medium {low_band:.2f}–{high_band:.2f}, High &gt; {high_band:.2f}.</div>",
            unsafe_allow_html=True,
        )

    st.write("")
    st.subheader("Top model drivers")
    if fi is None or fi.empty:
        st.info("Feature importance could not be extracted.")
    else:
        topn = st.slider("Show top N model drivers", 5, 30, 12)
        top_fi = fi.head(topn).copy()
        fig_fi = px.bar(top_fi.iloc[::-1], x="importance", y="feature", orientation="h")
        fig_fi.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_fi, use_container_width=True)

    st.write("")
    st.subheader("Recommended focus areas (grounded associations)")
    st.caption("These are associations observed in the dataset and model drivers. They do not imply causation.")

    recs = []

    # sales_channel
    if "sales_channel" in df.columns:
        tab, base = rate_table_by_category(df, target_col, "sales_channel", min_n=min_group_n, top_n=50)
        if not tab.empty:
            best = tab.iloc[0]
            worst = tab.sort_values("lift_vs_overall", ascending=True).iloc[0]
            recs.append(
                f"Channel gap: best={safe_str(best['sales_channel'])} at {(best['completion_rate']*100):.2f}% vs "
                f"worst={safe_str(worst['sales_channel'])} at {(worst['completion_rate']*100):.2f}%. "
                "Focus on reducing friction in lower-performing channels."
            )

    # trip_type
    if "trip_type" in df.columns:
        tab, base = rate_table_by_category(df, target_col, "trip_type", min_n=min_group_n, top_n=50)
        if not tab.empty:
            best = tab.iloc[0]
            worst = tab.sort_values("lift_vs_overall", ascending=True).iloc[0]
            recs.append(
                f"Trip type gap: best={safe_str(best['trip_type'])} at {(best['completion_rate']*100):.2f}% vs "
                f"worst={safe_str(worst['trip_type'])} at {(worst['completion_rate']*100):.2f}%. "
                "Review messaging and checkout steps for lower-performing trip types."
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
