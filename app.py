from __future__ import annotations

import io
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="Customer Booking Conversion Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
CANDIDATES = [
    APP_DIR / "customer_booking.csv",
    APP_DIR / "data.csv",
    APP_DIR / "customer_booking_prediction.csv",
]

BG = "#F6F8FC"
CARD = "#FFFFFF"
TEXT = "#111827"
MUTED = "rgba(17,24,39,0.68)"
BORDER = "rgba(15,23,42,0.10)"

DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
COMMON_BINARY_COLS = [
    "wants_extra_baggage",
    "wants_preferred_seat",
    "wants_in_flight_meals",
]
COMMON_NUMERIC_PRIORITY = [
    "purchase_lead",
    "length_of_stay",
    "flight_duration",
    "flight_hour",
    "num_passengers",
]
COMMON_CATEGORY_PRIORITY = [
    "sales_channel",
    "trip_type",
    "flight_day",
    "booking_origin",
    "route",
]


# =========================================================
# Styling
# =========================================================
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
          :root {{
            --bg: {BG};
            --panel: {CARD};
            --text: {TEXT};
            --muted: {MUTED};
            --border: {BORDER};
            --shadow: 0 10px 30px rgba(15,23,42,0.06);
            --radius: 18px;
            --accent: #1d4ed8;
            --accent-soft: rgba(29,78,216,0.08);
          }}

          html, body, [data-testid="stAppViewContainer"] {{
            background: var(--bg) !important;
            color: var(--text) !important;
          }}

          [data-testid="stHeader"] {{
            background: rgba(246,248,252,0.82);
          }}

          .block-container {{
            padding-top: 1.15rem;
            padding-bottom: 2.25rem;
            max-width: 1400px;
          }}

          #MainMenu {{ visibility: hidden; }}
          footer {{ visibility: hidden; }}

          section[data-testid="stSidebar"] > div {{
            background: #ffffff !important;
            border-right: 1px solid var(--border);
          }}

          .hero {{
            background: linear-gradient(135deg, #ffffff 0%, #f9fbff 100%);
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 24px 24px 18px 24px;
            box-shadow: var(--shadow);
            margin-bottom: 18px;
          }}

          .hero-title {{
            font-size: 30px;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin: 0 0 8px 0;
            color: var(--text);
          }}

          .hero-sub {{
            margin: 0;
            font-size: 15px;
            line-height: 1.6;
            color: var(--muted);
            max-width: 980px;
          }}

          .hero-strip {{
            margin-top: 14px;
            padding: 10px 12px;
            border-radius: 14px;
            background: var(--accent-soft);
            border: 1px solid rgba(29,78,216,0.12);
            color: #1e3a8a;
            font-size: 13px;
          }}

          .section-title {{
            font-size: 18px;
            font-weight: 800;
            color: var(--text);
            margin: 0 0 10px 0;
          }}

          .card {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            padding: 16px;
          }}

          .card-title {{
            margin: 0 0 6px 0;
            font-size: 16px;
            font-weight: 800;
            color: var(--text);
          }}

          .card-sub {{
            margin: 0 0 12px 0;
            color: var(--muted);
            font-size: 13px;
            line-height: 1.5;
          }}

          .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 12px;
            margin-bottom: 10px;
          }}

          @media (max-width: 1280px) {{
            .kpi-grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
          }}

          @media (max-width: 700px) {{
            .kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
          }}

          @media (max-width: 540px) {{
            .kpi-grid {{ grid-template-columns: repeat(1, minmax(0, 1fr)); }}
          }}

          .kpi-card {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 16px;
            box-shadow: var(--shadow);
            padding: 14px;
            min-height: 108px;
          }}

          .kpi-label {{
            font-size: 12px;
            color: var(--muted);
            margin-bottom: 8px;
          }}

          .kpi-value {{
            font-size: 24px;
            font-weight: 800;
            line-height: 1.1;
            color: var(--text);
            margin-bottom: 8px;
          }}

          .kpi-note {{
            font-size: 12px;
            color: var(--muted);
            line-height: 1.45;
          }}

          .insight-list {{
            margin: 0;
            padding-left: 18px;
          }}

          .insight-list li {{
            margin-bottom: 8px;
            line-height: 1.55;
            color: var(--text);
          }}

          .badge-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 6px;
          }}

          .badge {{
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            background: #f8fafc;
            border: 1px solid var(--border);
            color: var(--text);
            font-size: 12px;
          }}

          .small-note {{
            color: var(--muted);
            font-size: 12px;
          }}

          .js-plotly-plot .plotly .modebar {{
            opacity: 0.08;
          }}

          .js-plotly-plot .plotly:hover .modebar {{
            opacity: 1;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


# =========================================================
# Formatting helpers
# =========================================================
def fmt_num(x: Optional[float], digits: int = 2) -> str:
    if x is None or pd.isna(x) or not np.isfinite(x):
        return "N/A"
    return f"{x:,.{digits}f}"


def fmt_pct(x: Optional[float], digits: int = 1) -> str:
    if x is None or pd.isna(x) or not np.isfinite(x):
        return "N/A"
    return f"{x * 100:,.{digits}f}%"


def safe_divide(a: float, b: float) -> float:
    if b == 0 or pd.isna(b):
        return np.nan
    return a / b


def esc(text: str) -> str:
    return escape(str(text))


# =========================================================
# UI helpers
# =========================================================
def render_section_title(text: str) -> None:
    st.markdown(f"<div class='section-title'>{esc(text)}</div>", unsafe_allow_html=True)


def render_kpis(items: List[Tuple[str, str, str]]) -> None:
    blocks: List[str] = []
    for label, value, note in items:
        blocks.append(
            "<div class='kpi-card'>"
            f"<div class='kpi-label'>{esc(label)}</div>"
            f"<div class='kpi-value'>{esc(value)}</div>"
            f"<div class='kpi-note'>{esc(note)}</div>"
            "</div>"
        )
    st.markdown("<div class='kpi-grid'>" + "".join(blocks) + "</div>", unsafe_allow_html=True)


def render_list_card(title: str, items: List[str], subtitle: str = "") -> None:
    html = "<div class='card'>"
    html += f"<div class='card-title'>{esc(title)}</div>"
    if subtitle:
        html += f"<div class='card-sub'>{esc(subtitle)}</div>"
    html += "<ul class='insight-list'>"
    for item in items:
        html += f"<li>{esc(item)}</li>"
    html += "</ul></div>"
    st.markdown(html, unsafe_allow_html=True)


def render_badges(items: List[str]) -> None:
    html = "<div class='badge-row'>"
    for item in items:
        html += f"<span class='badge'>{esc(item)}</span>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# =========================================================
# Data loading
# =========================================================
@st.cache_data(show_spinner=False)
def load_csv_with_fallback_from_path(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "ISO-8859-1", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_csv_with_fallback_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    for enc in ("utf-8", "ISO-8859-1", "cp1252", "latin1"):
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(file_bytes))


def resolve_dataset_path() -> Optional[Path]:
    for p in CANDIDATES:
        if p.exists():
            return p
    return None


# =========================================================
# Data standardisation
# =========================================================
def dedupe_columns(cols) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for c in cols:
        c = str(c).strip()
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}__dup{seen[c]}")
    return out


def maybe_numeric_cast(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series
    temp = pd.to_numeric(series, errors="coerce")
    valid_ratio = temp.notna().mean()
    if valid_ratio >= 0.90:
        return temp
    return series


def standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = dedupe_columns(out.columns)

    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype(str).str.strip()

    for col in out.columns:
        out[col] = maybe_numeric_cast(out[col])

    if "flight_day" in out.columns:
        out["flight_day"] = pd.Categorical(out["flight_day"], categories=DAY_ORDER, ordered=True)

    if "booking_complete" in out.columns:
        out["booking_complete"] = pd.to_numeric(out["booking_complete"], errors="coerce")

    for c in COMMON_BINARY_COLS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def detect_target(df: pd.DataFrame) -> str:
    if "booking_complete" in df.columns:
        return "booking_complete"
    for c in df.columns:
        if c.strip().lower() == "booking_complete":
            return c
    return df.columns[-1]


def prepare_binary_target(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.dropna()
    if s.empty:
        return s
    return (s > 0).astype(int)


# =========================================================
# Filtering
# =========================================================
def apply_filters(
    df: pd.DataFrame,
    sales_channels: List[str],
    trip_types: List[str],
    flight_days: List[str],
) -> pd.DataFrame:
    out = df.copy()
    if sales_channels and "sales_channel" in out.columns:
        out = out[out["sales_channel"].astype(str).isin(sales_channels)]
    if trip_types and "trip_type" in out.columns:
        out = out[out["trip_type"].astype(str).isin(trip_types)]
    if flight_days and "flight_day" in out.columns:
        out = out[out["flight_day"].astype(str).isin(flight_days)]
    return out


# =========================================================
# Insight helpers
# =========================================================
def completion_rate(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.nan
    return float((s > 0).mean())


def rate_table_by_category(
    df: pd.DataFrame,
    target: str,
    col: str,
    min_n: int,
    top_n: int = 10,
) -> Tuple[pd.DataFrame, float]:
    if col not in df.columns or target not in df.columns:
        return pd.DataFrame(), np.nan

    base = completion_rate(df[target])
    grp = (
        df.groupby(col, dropna=False)[target]
        .agg(
            n="size",
            completion_rate=lambda x: float((pd.to_numeric(x, errors="coerce").fillna(0) > 0).mean()),
        )
        .reset_index()
    )
    grp["lift_vs_overall"] = grp["completion_rate"] - base
    grp = grp[grp["n"] >= min_n].copy()
    grp[col] = grp[col].astype(str)
    grp = grp.sort_values(["completion_rate", "n"], ascending=[False, False]).head(top_n)
    return grp, base


def numeric_bins_rate(df: pd.DataFrame, target: str, col: str, bins: int = 8) -> pd.DataFrame:
    if col not in df.columns or target not in df.columns:
        return pd.DataFrame()

    temp = pd.DataFrame(
        {
            col: pd.to_numeric(df[col], errors="coerce"),
            target: pd.to_numeric(df[target], errors="coerce"),
        }
    ).dropna()

    if temp.empty:
        return pd.DataFrame()

    try:
        temp["bin"] = pd.qcut(temp[col], q=bins, duplicates="drop")
    except ValueError:
        temp["bin"] = pd.cut(temp[col], bins=bins)

    out = (
        temp.groupby("bin", observed=True)
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


def addon_effects(df: pd.DataFrame, target: str) -> List[str]:
    notes: List[str] = []
    for col in COMMON_BINARY_COLS:
        if col not in df.columns or target not in df.columns:
            continue

        temp = pd.DataFrame(
            {
                col: pd.to_numeric(df[col], errors="coerce"),
                target: pd.to_numeric(df[target], errors="coerce"),
            }
        ).dropna()

        if temp.empty:
            continue

        grp = temp.groupby(col)[target].agg(
            n="size",
            rate=lambda x: float((x > 0).mean()),
        )

        if set(grp.index.tolist()) >= {0, 1}:
            rate0 = float(grp.loc[0, "rate"])
            rate1 = float(grp.loc[1, "rate"])
            diff = (rate1 - rate0) * 100
            notes.append(
                f"{col.replace('_', ' ')} is associated with a {diff:+.2f} percentage-point change in observed completion when the indicator is 1 instead of 0."
            )
    return notes


def probability_band(p: float, low: float, high: float) -> str:
    if not np.isfinite(p):
        return "Unknown"
    if p < low:
        return "Low"
    if p <= high:
        return "Medium"
    return "High"


def build_training_frame(df: pd.DataFrame, target_col: str, max_categories: int = 25) -> Tuple[pd.DataFrame, pd.Series]:
    work = df.copy()

    drop_candidates = [c for c in work.columns if c.lower() in {"customerid", "customer_id"}]
    work = work.drop(columns=drop_candidates, errors="ignore")

    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna(subset=[target_col]).copy()
    y = (work[target_col] > 0).astype(int)

    X = work.drop(columns=[target_col], errors="ignore").copy()

    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype).startswith("category"):
            X[col] = X[col].astype(str).fillna("Unknown").replace({"nan": "Unknown"})

            nunique = X[col].nunique(dropna=True)
            if nunique > max_categories:
                top = X[col].value_counts().head(max_categories).index
                X[col] = np.where(X[col].isin(top), X[col], "Other")

    return X, y


def build_pipeline(n_estimators: int, max_depth: Optional[int], random_state: int, X: pd.DataFrame) -> Pipeline:
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

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

    prep = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        random_state=int(random_state),
        n_jobs=-1,
        max_depth=max_depth,
        class_weight="balanced_subsample",
    )

    return Pipeline(steps=[("prep", prep), ("model", model)])


def collapse_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    original_cols: List[str],
) -> pd.DataFrame:
    cols_sorted = sorted(original_cols, key=len, reverse=True)
    mapped: List[str] = []

    for feat in feature_names:
        base = feat
        for col in cols_sorted:
            if feat == col or feat.startswith(f"{col}_"):
                base = col
                break
        mapped.append(base)

    imp = pd.DataFrame({"feature": mapped, "importance": importances})
    imp = imp.groupby("feature", as_index=False)["importance"].sum()
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    return imp


def threshold_metrics(y_true: pd.Series, proba: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred),
        "pred": pred,
    }


def threshold_tradeoff_table(y_true: pd.Series, proba: np.ndarray) -> pd.DataFrame:
    rows = []
    for thr in [0.30, 0.40, 0.50, 0.60, 0.70]:
        m = threshold_metrics(y_true, proba, thr)
        rows.append(
            {
                "Threshold": thr,
                "Accuracy": m["accuracy"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1": m["f1"],
            }
        )
    return pd.DataFrame(rows)


def train_and_evaluate(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    random_state: int,
    n_estimators: int,
    max_depth: Optional[int],
) -> Optional[Dict]:
    X, y = build_training_frame(df, target_col=target_col)

    if len(X) < 200:
        return None
    if y.nunique() < 2:
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=y,
    )

    pipe = build_pipeline(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        X=X_train,
    )
    pipe.fit(X_train, y_train)

    proba_test = pipe.predict_proba(X_test)[:, 1]
    proba_all = pipe.predict_proba(X)[:, 1]

    auc = float(roc_auc_score(y_test, proba_test))
    ap = float(average_precision_score(y_test, proba_test))

    feature_names = list(pipe.named_steps["prep"].get_feature_names_out())
    raw_importances = pipe.named_steps["model"].feature_importances_
    agg_importance = collapse_feature_importance(feature_names, raw_importances, list(X.columns))

    fpr, tpr, roc_thr = roc_curve(y_test, proba_test)
    prec_curve, rec_curve, pr_thr = precision_recall_curve(y_test, proba_test)

    return {
        "pipeline": pipe,
        "X": X,
        "y": y,
        "X_test": X_test,
        "y_test": y_test,
        "proba_test": proba_test,
        "proba_all": proba_all,
        "roc_auc": auc,
        "average_precision": ap,
        "feature_importance": agg_importance,
        "roc": (fpr, tpr, roc_thr),
        "pr": (prec_curve, rec_curve, pr_thr),
    }


def build_data_driven_insights(df: pd.DataFrame, target_col: str, min_group_n: int) -> List[str]:
    notes: List[str] = []

    overall = completion_rate(df[target_col])
    notes.append(
        f"Overall booking completion is {fmt_pct(overall)} in the current analysis scope."
    )

    key_cats = [c for c in COMMON_CATEGORY_PRIORITY if c in df.columns]
    best_segment = None
    worst_segment = None
    best_lift = -np.inf
    worst_lift = np.inf

    for col in key_cats:
        tab, base = rate_table_by_category(df, target_col, col, min_n=min_group_n, top_n=50)
        if tab.empty:
            continue
        top = tab.iloc[0]
        bot = tab.sort_values("lift_vs_overall", ascending=True).iloc[0]

        if float(top["lift_vs_overall"]) > best_lift:
            best_lift = float(top["lift_vs_overall"])
            best_segment = f"{col} = {top[col]}"

        if float(bot["lift_vs_overall"]) < worst_lift:
            worst_lift = float(bot["lift_vs_overall"])
            worst_segment = f"{col} = {bot[col]}"

    if best_segment is not None:
        notes.append(
            f"The strongest observed conversion segment is {best_segment}, performing {best_lift * 100:+.2f} percentage points versus the overall rate."
        )
    if worst_segment is not None:
        notes.append(
            f"The weakest observed conversion segment is {worst_segment}, performing {worst_lift * 100:+.2f} percentage points versus the overall rate."
        )

    for num_col in COMMON_NUMERIC_PRIORITY:
        if num_col in df.columns:
            bins = numeric_bins_rate(df, target_col, num_col, bins=8)
            if not bins.empty:
                top_bin = bins.sort_values("completion_rate", ascending=False).iloc[0]
                low_bin = bins.sort_values("completion_rate", ascending=True).iloc[0]
                notes.append(
                    f"For {num_col}, the strongest observed range is {top_bin['value_min']:.0f} to {top_bin['value_max']:.0f}, while the weakest range is {low_bin['value_min']:.0f} to {low_bin['value_max']:.0f}."
                )
                break

    addon_notes = addon_effects(df, target_col)
    if addon_notes:
        notes.append(addon_notes[0])

    return notes[:5]


def build_model_driven_insights(
    model_result: Optional[Dict],
    threshold: float,
    low_band: float,
    high_band: float,
) -> List[str]:
    if model_result is None:
        return [
            "Model-driven insights are unavailable because the filtered dataset is too small or the target class is not sufficiently mixed."
        ]

    notes: List[str] = []
    y_test = model_result["y_test"]
    proba_test = model_result["proba_test"]
    metrics = threshold_metrics(y_test, proba_test, threshold)
    fi = model_result["feature_importance"]

    notes.append(
        f"The Random Forest reaches ROC-AUC {model_result['roc_auc']:.3f} and average precision {model_result['average_precision']:.3f} on the holdout split."
    )
    notes.append(
        f"At the active decision threshold of {threshold:.2f}, precision is {metrics['precision']:.3f}, recall is {metrics['recall']:.3f}, and F1 is {metrics['f1']:.3f}."
    )

    if fi is not None and not fi.empty:
        top_features = ", ".join(fi.head(4)["feature"].astype(str).tolist())
        notes.append(f"The most influential model drivers are {top_features}.")

    proba_all = model_result["proba_all"]
    high_share = float((proba_all > high_band).mean())
    low_share = float((proba_all < low_band).mean())
    notes.append(
        f"{fmt_pct(high_share)} of records fall into the high-probability completion band, while {fmt_pct(low_share)} fall into the low-probability band."
    )

    return notes[:5]


# =========================================================
# Sidebar controls
# =========================================================
st.sidebar.title("Controls")

use_auto_file = st.sidebar.checkbox("Load local dataset automatically", value=True)

if use_auto_file:
    path = resolve_dataset_path()
    if path is None:
        st.error("Dataset file not found. Place customer_booking.csv, data.csv, or customer_booking_prediction.csv next to app.py.")
        st.stop()
    raw = load_csv_with_fallback_from_path(str(path))
    source_label = path.name
else:
    uploaded = st.sidebar.file_uploader("Upload booking CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV or enable automatic local loading.")
        st.stop()
    raw = load_csv_with_fallback_from_bytes(uploaded.getvalue())
    source_label = uploaded.name

df = standardize_df(raw)
target_default = detect_target(df)

st.sidebar.divider()
st.sidebar.subheader("Target and filters")
target_col = st.sidebar.selectbox(
    "Target column",
    options=df.columns.tolist(),
    index=df.columns.tolist().index(target_default),
)

sales_channels = []
trip_types = []
flight_days = []

if "sales_channel" in df.columns:
    sales_channels = st.sidebar.multiselect(
        "Sales channel",
        options=sorted(df["sales_channel"].dropna().astype(str).unique().tolist()),
        default=[],
    )

if "trip_type" in df.columns:
    trip_types = st.sidebar.multiselect(
        "Trip type",
        options=sorted(df["trip_type"].dropna().astype(str).unique().tolist()),
        default=[],
    )

if "flight_day" in df.columns:
    flight_days = st.sidebar.multiselect(
        "Flight day",
        options=DAY_ORDER,
        default=[],
    )

st.sidebar.divider()
st.sidebar.subheader("Model settings")
test_size = st.sidebar.slider("Test split", 0.10, 0.40, 0.20, step=0.05)
random_state = st.sidebar.number_input("Random state", min_value=0, max_value=10000, value=42, step=1)
n_estimators = st.sidebar.slider("Trees", 100, 600, 250, step=50)
depth_choice = st.sidebar.selectbox("Max depth", ["None", "10", "20", "30"], index=0)
max_depth = None if depth_choice == "None" else int(depth_choice)
threshold = st.sidebar.slider("Decision threshold", 0.10, 0.90, 0.50, step=0.05)

st.sidebar.divider()
st.sidebar.subheader("Insight settings")
low_band = st.sidebar.slider("Low probability cutoff", 0.10, 0.60, 0.40, step=0.05)
high_band = st.sidebar.slider("High probability cutoff", 0.50, 0.95, 0.70, step=0.05)
min_group_n = st.sidebar.slider("Minimum group size", 50, 1000, 200, step=50)

# =========================================================
# Apply filters
# =========================================================
dff = apply_filters(
    df=df,
    sales_channels=sales_channels,
    trip_types=trip_types,
    flight_days=flight_days,
)

if dff.empty:
    st.warning("No rows remain after the selected filters.")
    st.stop()

target_numeric = prepare_binary_target(dff[target_col])
if target_numeric.empty:
    st.error("The selected target column does not contain usable binary values.")
    st.stop()

model_result = train_and_evaluate(
    df=dff,
    target_col=target_col,
    test_size=test_size,
    random_state=int(random_state),
    n_estimators=int(n_estimators),
    max_depth=max_depth,
)

# =========================================================
# Hero
# =========================================================
scope_bits = [f"Source: {source_label}"]
if sales_channels:
    scope_bits.append(f"Sales channel: {len(sales_channels)} selected")
if trip_types:
    scope_bits.append(f"Trip type: {len(trip_types)} selected")
if flight_days:
    scope_bits.append(f"Flight day: {len(flight_days)} selected")
scope_bits.append(f"Records in view: {len(dff):,}")

hero_text = (
    "This dashboard focuses on customer booking conversion: which segments complete bookings at higher rates, "
    "which features matter most to the classifier, and how scenario changes affect predicted completion probability."
)

st.markdown(
    f"""
    <div class="hero">
      <div class="hero-title">Customer Booking Conversion Intelligence Dashboard</div>
      <p class="hero-sub">{esc(hero_text)}</p>
      <div class="hero-strip">{esc(' | '.join(scope_bits))}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# KPI row
# =========================================================
overall_rate = completion_rate(dff[target_col])
completed_count = int((pd.to_numeric(dff[target_col], errors="coerce").fillna(0) > 0).sum())
proba_high_share = np.nan
top_driver = "N/A"
best_segment_text = "N/A"

if model_result is not None:
    proba_high_share = float((model_result["proba_all"] > high_band).mean())
    if not model_result["feature_importance"].empty:
        top_driver = str(model_result["feature_importance"].iloc[0]["feature"])

key_segments = []
for col in [c for c in COMMON_CATEGORY_PRIORITY if c in dff.columns]:
    tab, _ = rate_table_by_category(dff, target_col, col, min_n=min_group_n, top_n=50)
    if not tab.empty:
        top_row = tab.iloc[0]
        key_segments.append((f"{col} = {top_row[col]}", float(top_row["completion_rate"])))

if key_segments:
    key_segments = sorted(key_segments, key=lambda x: x[1], reverse=True)
    best_segment_text = key_segments[0][0]

render_kpis(
    [
        ("Completion rate", fmt_pct(overall_rate), "Observed share of completed bookings"),
        ("Completed bookings", f"{completed_count:,}", "Positive target count in the current slice"),
        ("High-probability share", fmt_pct(proba_high_share), "Records above the high-probability completion band"),
        ("Best observed segment", best_segment_text, "Highest segment completion rate after minimum sample filtering"),
        ("Model ROC-AUC", f"{model_result['roc_auc']:.3f}" if model_result is not None else "N/A", "Holdout ranking quality"),
        ("Top model driver", top_driver, "Most influential aggregated feature"),
    ]
)

# =========================================================
# Tabs
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Executive summary",
        "Booking behavior",
        "Model evaluation",
        "Scenario tool",
        "Data appendix",
    ]
)

# =========================================================
# Executive summary
# =========================================================
with tab1:
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        render_list_card(
            "1. Data-driven insights",
            build_data_driven_insights(dff, target_col, min_group_n=min_group_n),
            "Observed completion patterns from the filtered dataset.",
        )

        st.write("")
        render_section_title("Observed completion by segment")
        available_cat = [c for c in COMMON_CATEGORY_PRIORITY if c in dff.columns]
        if available_cat:
            selected_cat = st.selectbox("Segment dimension", available_cat, key="summary_segment")
            seg_tab, base = rate_table_by_category(dff, target_col, selected_cat, min_n=min_group_n, top_n=12)
            if not seg_tab.empty:
                seg_tab["completion_rate_pct"] = seg_tab["completion_rate"] * 100
                fig = px.bar(
                    seg_tab.sort_values("completion_rate_pct"),
                    x="completion_rate_pct",
                    y=selected_cat,
                    orientation="h",
                    labels={"completion_rate_pct": "Completion rate (%)", selected_cat: selected_cat},
                )
                fig.add_vline(x=base * 100, line_dash="dash")
                fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No groups meet the minimum sample requirement for this dimension.")
        else:
            st.info("No expected categorical segment columns were found.")

    with right:
        render_list_card(
            "2. Model-driven insights",
            build_model_driven_insights(model_result, threshold=threshold, low_band=low_band, high_band=high_band),
            "Classifier-based insights that complement the observed segment patterns.",
        )

        st.write("")
        render_section_title("Probability band distribution")
        if model_result is not None:
            band_df = pd.DataFrame(
                {
                    "predicted_probability": model_result["proba_all"],
                }
            )
            band_df["band"] = [
                probability_band(p, low=low_band, high=high_band)
                for p in band_df["predicted_probability"]
            ]
            band_counts = (
                band_df["band"]
                .value_counts()
                .reindex(["Low", "Medium", "High", "Unknown"])
                .dropna()
                .reset_index()
            )
            band_counts.columns = ["band", "count"]
            band_counts["share"] = band_counts["count"] / band_counts["count"].sum()

            fig = px.bar(
                band_counts,
                x="band",
                y="share",
                labels={"band": "Probability band", "share": "Share of records"},
            )
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Probability bands are unavailable because the model could not be trained.")

# =========================================================
# Booking behavior
# =========================================================
with tab2:
    c1, c2 = st.columns([1.0, 1.0], gap="large")

    with c1:
        render_section_title("Numeric conversion profile")
        numeric_candidates = [c for c in COMMON_NUMERIC_PRIORITY if c in dff.columns]
        if numeric_candidates:
            selected_num = st.selectbox("Numeric feature", numeric_candidates, key="num_profile")
            bins_df = numeric_bins_rate(dff, target_col, selected_num, bins=8)
            if not bins_df.empty:
                bins_df["completion_rate_pct"] = bins_df["completion_rate"] * 100
                fig = px.line(
                    bins_df,
                    x="value_median",
                    y="completion_rate_pct",
                    markers=True,
                    labels={
                        "value_median": f"{selected_num} bin median",
                        "completion_rate_pct": "Completion rate (%)",
                    },
                )
                fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig, use_container_width=True)

                display_bins = bins_df.copy()
                display_bins["value_min"] = display_bins["value_min"].round(2)
                display_bins["value_max"] = display_bins["value_max"].round(2)
                display_bins["completion_rate_pct"] = display_bins["completion_rate_pct"].round(2)
                display_bins = display_bins[["value_min", "value_max", "n", "completion_rate_pct"]]
                st.dataframe(display_bins, use_container_width=True, hide_index=True)
            else:
                st.info("Not enough valid numeric values were found for this feature.")
        else:
            st.info("No expected numeric booking features were found.")

    with c2:
        render_section_title("Add-on association view")
        addon_rows = []
        for col in COMMON_BINARY_COLS:
            if col in dff.columns:
                temp = pd.DataFrame(
                    {
                        col: pd.to_numeric(dff[col], errors="coerce"),
                        target_col: pd.to_numeric(dff[target_col], errors="coerce"),
                    }
                ).dropna()
                if temp.empty:
                    continue
                grp = temp.groupby(col)[target_col].agg(
                    n="size",
                    completion_rate=lambda x: float((x > 0).mean()),
                ).reset_index()
                grp["feature"] = col
                addon_rows.append(grp)

        if addon_rows:
            addon_df = pd.concat(addon_rows, ignore_index=True)
            addon_df["completion_rate_pct"] = addon_df["completion_rate"] * 100
            addon_df["label"] = addon_df["feature"] + " = " + addon_df.iloc[:, 0].astype(int).astype(str)

            fig = px.bar(
                addon_df,
                x="feature",
                y="completion_rate_pct",
                color=addon_df.iloc[:, 0].astype(str),
                barmode="group",
                labels={"feature": "Add-on feature", "completion_rate_pct": "Completion rate (%)", "color": "Flag"},
            )
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)

            display_addons = addon_df[[addon_df.columns[0], "feature", "n", "completion_rate_pct"]].copy()
            display_addons.columns = ["flag", "feature", "n", "completion_rate_pct"]
            display_addons["completion_rate_pct"] = display_addons["completion_rate_pct"].round(2)
            st.dataframe(display_addons, use_container_width=True, hide_index=True)
        else:
            st.info("No usable add-on indicator columns were found.")

    st.write("")
    render_section_title("Observed completion heatmap")
    heatmap_dims = [c for c in ["sales_channel", "trip_type", "flight_day"] if c in dff.columns]
    if len(heatmap_dims) >= 2:
        row_dim = st.selectbox("Rows", heatmap_dims, index=0, key="heat_row")
        col_choices = [c for c in heatmap_dims if c != row_dim]
        col_dim = st.selectbox("Columns", col_choices, index=0, key="heat_col")

        temp = dff[[row_dim, col_dim, target_col]].copy()
        temp[target_col] = pd.to_numeric(temp[target_col], errors="coerce")
        temp = temp.dropna()
        if not temp.empty:
            pivot = temp.pivot_table(
                index=row_dim,
                columns=col_dim,
                values=target_col,
                aggfunc=lambda x: float((x > 0).mean()),
            )
            fig = px.imshow(
                pivot * 100,
                text_auto=".1f",
                aspect="auto",
                labels=dict(
                x=col_dim,
                y=row_dim,
                color="Completion rate (%)",
            ),
         )
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("The selected dimensions do not have enough usable rows.")

# =========================================================
# Model evaluation
# =========================================================
with tab3:
    if model_result is None:
        st.info("Model evaluation is unavailable because the filtered dataset is too small or the target class is not sufficiently mixed.")
    else:
        y_test = model_result["y_test"]
        proba_test = model_result["proba_test"]
        metrics = threshold_metrics(y_test, proba_test, threshold)
        report_df = pd.DataFrame(
            classification_report(
                y_test,
                metrics["pred"],
                output_dict=True,
                zero_division=0,
            )
        ).T

        render_kpis(
            [
                ("Accuracy", f"{metrics['accuracy']:.3f}", "Share of correct holdout predictions"),
                ("Precision", f"{metrics['precision']:.3f}", "Share of predicted completions that are correct"),
                ("Recall", f"{metrics['recall']:.3f}", "Share of actual completions captured"),
                ("F1", f"{metrics['f1']:.3f}", "Balance between precision and recall"),
                ("ROC-AUC", f"{model_result['roc_auc']:.3f}", "Probability ranking quality"),
                ("Average precision", f"{model_result['average_precision']:.3f}", "Precision-recall summary"),
            ]
        )

        a, b = st.columns([1.0, 1.0], gap="large")

        with a:
            render_section_title("Confusion matrix")
            cm_df = pd.DataFrame(
                metrics["confusion_matrix"],
                index=["Actual 0", "Actual 1"],
                columns=["Pred 0", "Pred 1"],
            )
            fig = px.imshow(cm_df, text_auto=True, aspect="auto")
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with b:
            render_section_title("Threshold trade-off table")
            thr_df = threshold_tradeoff_table(y_test, proba_test).copy()
            for col in ["Accuracy", "Precision", "Recall", "F1"]:
                thr_df[col] = thr_df[col].round(3)
            st.dataframe(thr_df, use_container_width=True, hide_index=True)

        st.write("")
        c, d = st.columns([1.0, 1.0], gap="large")

        with c:
            render_section_title("ROC curve")
            fpr, tpr, _ = model_result["roc"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Baseline"))
            fig.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis_title="False positive rate",
                yaxis_title="True positive rate",
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with d:
            render_section_title("Precision-recall curve")
            prec_curve, rec_curve, _ = model_result["pr"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=rec_curve, y=prec_curve, mode="lines", name="PR"))
            fig.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis_title="Recall",
                yaxis_title="Precision",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.write("")
        render_section_title("Aggregated feature importance")
        fi = model_result["feature_importance"].head(15).copy()
        fig = px.bar(
            fi.iloc[::-1],
            x="importance",
            y="feature",
            orientation="h",
            labels={"importance": "Importance", "feature": "Feature"},
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.write("")
        render_section_title("Classification report")
        show_cols = [c for c in ["precision", "recall", "f1-score", "support"] if c in report_df.columns]
        st.dataframe(report_df[show_cols], use_container_width=True)

# =========================================================
# Scenario tool
# =========================================================
with tab4:
    if model_result is None:
        st.info("The scenario tool is unavailable because the filtered dataset is too small or the target class is not sufficiently mixed.")
    else:
        render_section_title("Booking completion scenario")
        st.caption("Change inputs below to estimate the probability that a booking will be completed.")

        X_model = model_result["X"].copy()
        input_row: Dict[str, object] = {}

        cols_ui = st.columns(3, gap="large")
        for i, col in enumerate(X_model.columns):
            box = cols_ui[i % 3]
            s = X_model[col]

            if pd.api.types.is_numeric_dtype(s):
                vals = pd.to_numeric(s, errors="coerce").dropna()
                if vals.empty:
                    input_row[col] = 0.0
                    continue

                q01 = float(vals.quantile(0.01))
                q99 = float(vals.quantile(0.99))
                median = float(vals.median())

                is_integer_like = bool(np.allclose(vals.dropna() % 1, 0))
                if is_integer_like:
                    input_row[col] = box.number_input(
                        col,
                        min_value=int(np.floor(q01)),
                        max_value=int(np.ceil(q99)),
                        value=int(round(median)),
                        step=1,
                    )
                else:
                    input_row[col] = box.slider(
                        col,
                        min_value=float(q01),
                        max_value=float(q99),
                        value=float(median),
                    )
            else:
                values = sorted(pd.Series(s).dropna().astype(str).unique().tolist())
                if not values:
                    input_row[col] = ""
                else:
                    default_value = values[0]
                    if col in COMMON_CATEGORY_PRIORITY:
                        mode_vals = pd.Series(s).mode(dropna=True)
                        if not mode_vals.empty and str(mode_vals.iloc[0]) in values:
                            default_value = str(mode_vals.iloc[0])

                    input_row[col] = box.selectbox(
                        col,
                        options=values,
                        index=values.index(default_value),
                    )

        X_new = pd.DataFrame([input_row], columns=X_model.columns)
        p = float(model_result["pipeline"].predict_proba(X_new)[:, 1][0])
        pred_class = int(p >= threshold)
        band = probability_band(p, low=low_band, high=high_band)

        render_kpis(
            [
                ("Predicted completion probability", fmt_pct(p), "Model-estimated conversion likelihood"),
                ("Probability band", band, "Based on the current low and high cutoffs"),
                ("Decision threshold", f"{threshold:.2f}", "Current classification threshold"),
                ("Predicted class", "Completed" if pred_class == 1 else "Not completed", "Threshold-based decision"),
                ("Low cutoff", f"{low_band:.2f}", "Start of the medium probability band"),
                ("High cutoff", f"{high_band:.2f}", "Start of the high probability band"),
            ]
        )

        st.write("")
        render_section_title("Scenario interpretation")
        scenario_notes = [
            f"This scenario sits in the {band.lower()} probability band.",
            f"The model would classify the case as {'completed' if pred_class == 1 else 'not completed'} at the active threshold.",
            "Use this tool to compare scenarios rather than treating the output as a guaranteed customer action.",
        ]
        render_list_card(
            "Scenario notes",
            scenario_notes,
            "Probability-based interpretation for the values entered above.",
        )

# =========================================================
# Data appendix
# =========================================================
with tab5:
    render_section_title("Source and structure")
    render_badges(
        [
            f"Source: {source_label}",
            f"Rows in current slice: {len(dff):,}",
            f"Columns: {dff.shape[1]:,}",
            f"Target: {target_col}",
        ]
    )

    st.write("")
    a, b = st.columns([1.0, 1.0], gap="large")

    with a:
        render_section_title("Schema")
        schema = pd.DataFrame({"column": dff.columns, "dtype": [str(t) for t in dff.dtypes]})
        st.dataframe(schema, use_container_width=True, hide_index=True)

    with b:
        render_section_title("Missingness summary")
        miss = dff.isna().mean().sort_values(ascending=False).reset_index()
        miss.columns = ["column", "missing_share"]
        miss["missing_share"] = miss["missing_share"].map(lambda x: fmt_pct(x))
        st.dataframe(miss, use_container_width=True, hide_index=True)

    st.write("")
    render_section_title("Filtered preview")
    preview_rows = st.slider("Rows to preview", 20, 200, 50, step=10)
    st.dataframe(dff.head(preview_rows), use_container_width=True, hide_index=True)

    st.write("")
    csv_bytes = dff.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered data as CSV",
        data=csv_bytes,
        file_name="customer_booking_filtered.csv",
        mime="text/csv",
    )