import time
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st
from streamlit_lottie import st_lottie, st_lottie_spinner
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "events_10000_sessions.csv"

FEATURE_COLUMNS = ["num_clicks", "num_carts", "num_events", "num_unique_items"]
SCALED_MODELS = {"Logistic Regression", "SVC"}

# --- Validated palette (dataviz skill reference palette; light mode) ---
BLUE = "#2a78d6"        # categorical slot 1 / sequential hue
ORANGE = "#eb6834"      # categorical slot 2
AQUA = "#1baf7a"        # categorical slot 3
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
ORDINAL_BLUE = ["#86b6ef", "#2a78d6", "#104281"]  # funnel stages (>=step 250)
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
FONT_FAMILY = "system-ui, -apple-system, 'Segoe UI', sans-serif"
MODEL_COLORS = {"Random Forest": BLUE, "Logistic Regression": ORANGE, "SVC": AQUA}

#Lottie animation
LOTTIE_WELCOME = "https://assets5.lottiefiles.com/packages/lf20_V9t630.json"
LOTTIE_SUCCESS = "https://raw.githubusercontent.com/ariyanshiputech/custom_quick_alert/main/assets/animations/success.json"
LOTTIE_LOADING = "https://raw.githubusercontent.com/ariyanshiputech/custom_quick_alert/main/assets/animations/loading.json"

@st.cache_data(show_spinner=False, ttl=3600)
def load_lottie_url(url: str):
    """Fetch a Lottie animation's JSON from a URL. Returns None on any
    failure (empty URL, network error, bad response, invalid JSON) so a
    missing or stale animation never crashes the page -- callers just skip
    rendering it."""
    if not url:
        return None
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError):
        return None


st.set_page_config(page_title="E-commerce Purchase Predictor", page_icon="🛒", layout="wide")

def style_fig(fig: go.Figure, height: int = 320, showlegend: bool = False) -> go.Figure:
    """Apply one consistent, minimal look to every chart in the app."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PRIMARY_INK, family=FONT_FAMILY, size=13),
        showlegend=showlegend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(bgcolor="white", font=dict(color=PRIMARY_INK, family=FONT_FAMILY)),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=GRIDLINE, color=MUTED_INK)
    fig.update_yaxes(showgrid=True, gridcolor=GRIDLINE, zeroline=False, color=MUTED_INK)
    return fig

@st.cache_data(show_spinner=False)
def load_events() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def build_session_features(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw events into one row per session, plus the order label.

    num_events is deliberately "clicks + carts" rather than the raw
    per-session event count used in the exploratory notebook: the raw count
    also includes any "orders" rows, which would leak the label we're
    trying to predict straight into a feature. For a session that hasn't
    ordered yet (the real prediction use case) only clicks/carts have
    happened, so training and live inference stay consistent.
    """
    target = (
        events.groupby("session")["type"]
        .apply(lambda values: "orders" in values.values)
        .astype(int)
        .rename("target")
    )

features = events.groupby("session").agg(
        num_clicks=("type", lambda values: (values == "clicks").sum()),
        num_carts=("type", lambda values: (values == "carts").sum()),
        num_unique_items=("aid", "nunique"),
    )
    features["num_events"] = features["num_clicks"] + features["num_carts"]
    features = features[FEATURE_COLUMNS]

    return features.join(target)
@st.cache_resource(show_spinner="Training Random Forest, Logistic Regression, and SVC...")
def train_models(session_data: pd.DataFrame) -> dict:
    x = session_data[FEATURE_COLUMNS]
    y = session_data["target"]

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    x_train_val_scaled = scaler.fit_transform(x_train_val)
    x_test_scaled = scaler.transform(x_test)