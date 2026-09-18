import time
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st
try:
    from streamlit_lottie import st_lottie, st_lottie_spinner
except ModuleNotFoundError:
    st_lottie = None
    from contextlib import contextmanager

    @contextmanager
    def st_lottie_spinner(*_args, **_kwargs):
        yield
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

try:
    from src.e_commerce_ml.explanations import explain_prediction
except ImportError:
    from e_commerce_ml.explanations import explain_prediction

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "events_10000_sessions.csv"

FEATURE_COLUMNS = ["num_clicks", "num_carts", "num_events", "num_unique_items"]
MODEL_FEATURE_COLUMNS = FEATURE_COLUMNS + [
    "session_duration_seconds",
    "hour",
    "weekday",
]
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
API_URL = os.getenv("API_URL", "http://localhost:8000")

# --- Lottie animations (https://lottiefiles.com) -----------------------------
# LOTTIE_WELCOME is a verified, working public animation (a friendly wave) so
# the app has at least one Lottie moment out of the box. LOTTIE_SUCCESS and
# LOTTIE_LOADING are left as placeholders: pick any free animation you like on
# lottiefiles.com, open it, use "Download" -> "Lottie JSON" (or the copy-URL
# button) to get a direct .json URL, and paste it in below. Every call site
# checks for None first, so leaving a placeholder empty (or a URL that stops
# working later) never breaks the app -- it just skips that animation.
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

st.markdown(
    """
    <style>
    .explanation-kicker {
        color: #2a78d6;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: -0.45rem;
    }
    .explanation-title {
        color: #0b0b0b;
        font-size: 1.45rem;
        font-weight: 750;
        margin-bottom: 0.2rem;
    }
    .explanation-summary {
        color: #52514e;
        font-size: 1rem;
        margin-bottom: 0.7rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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


def request_prediction(payload: dict) -> dict | None:
    """Request a prediction from the FastAPI domain service."""
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=0.5)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError):
        return None


@st.cache_data(show_spinner=False, ttl=30)
def get_api_model_info() -> dict | None:
    """Read the model metadata advertised by the prediction service."""
    try:
        response = requests.get(f"{API_URL}/model-info", timeout=0.5)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError):
        return None


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

    rows = []
    for session_id, session_events in events.groupby("session"):
        session_events = session_events.sort_values("ts")
        order_mask = session_events["type"] == "orders"
        feature_events = session_events.iloc[: order_mask.values.argmax()] if order_mask.any() else session_events
        feature_events = feature_events[feature_events["type"].isin(["clicks", "carts"])]
        if feature_events.empty:
            continue
        rows.append(
            {
                "session": session_id,
                "num_clicks": int((feature_events["type"] == "clicks").sum()),
                "num_carts": int((feature_events["type"] == "carts").sum()),
                "num_events": len(feature_events),
                "num_unique_items": feature_events["aid"].nunique(),
                "session_duration_seconds": (feature_events["ts"].max() - feature_events["ts"].min()) / 1000.0,
                "hour": int(feature_events["hour"].iloc[-1]),
                "weekday": feature_events["weekday"].iloc[-1],
            }
        )
    features = pd.DataFrame(rows).set_index("session")
    features["weekday"] = features["weekday"].map(
        {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
         "Friday": 4, "Saturday": 5, "Sunday": 6}
    ).fillna(0)
    features = features[MODEL_FEATURE_COLUMNS]

    return features.join(target)


@st.cache_resource(show_spinner="Training Random Forest, Logistic Regression, and SVC...")
def train_models(session_data: pd.DataFrame) -> dict:
    x = session_data.reindex(columns=MODEL_FEATURE_COLUMNS, fill_value=0)
    y = session_data["target"]

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    x_train_val_scaled = scaler.fit_transform(x_train_val)
    x_test_scaled = scaler.transform(x_test)

    # Fixed, already-reasonable hyperparameters instead of the notebook's
    # full GridSearchCV sweep, so the app trains in seconds rather than
    # minutes on startup.
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "SVC": SVC(class_weight="balanced", probability=True, random_state=42),
    }

    comparison_rows = []
    fitted = {}
    for name, model in models.items():
        if name in SCALED_MODELS:
            model.fit(x_train_val_scaled, y_train_val)
            pred = model.predict(x_test_scaled)
            prob = model.predict_proba(x_test_scaled)[:, 1]
        else:
            model.fit(x_train_val, y_train_val)
            pred = model.predict(x_test)
            prob = model.predict_proba(x_test)[:, 1]

        comparison_rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "f1_score": f1_score(y_test, pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, prob),
            }
        )
        fitted[name] = model

    comparison = pd.DataFrame(comparison_rows).sort_values("f1_score", ascending=False)
    best_name = comparison.iloc[0]["model"]
    best_model = fitted[best_name]
    best_needs_scaling = best_name in SCALED_MODELS

    best_x_test = x_test_scaled if best_needs_scaling else x_test
    best_pred = best_model.predict(best_x_test)
    matrix = confusion_matrix(y_test, best_pred)
    report = classification_report(
        y_test, best_pred, target_names=["No order", "Order"], zero_division=0
    )

    importances = None
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(
            best_model.feature_importances_, index=MODEL_FEATURE_COLUMNS
        ).sort_values()

    return {
        "best_name": best_name,
        "best_model": best_model,
        "best_needs_scaling": best_needs_scaling,
        "scaler": scaler,
        "comparison": comparison,
        "matrix": matrix,
        "report": report,
        "importances": importances,
    }


def make_gauge(probability_pct: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability_pct,
            number={"suffix": "%", "font": {"size": 42, "color": PRIMARY_INK}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": MUTED_INK,
                    "tickfont": {"color": MUTED_INK},
                },
                "bar": {"color": BLUE, "thickness": 0.3},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 33], "color": SEQ_BLUE[0]},
                    {"range": [33, 66], "color": SEQ_BLUE[2]},
                    {"range": [66, 100], "color": SEQ_BLUE[4]},
                ],
                "threshold": {
                    "line": {"color": PRIMARY_INK, "width": 2},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=30, r=30, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": PRIMARY_INK, "family": FONT_FAMILY},
    )
    return fig


def render_predict_tab(model_info: dict | None) -> None:
    st.subheader("Try a session")
    st.write("Describe a session so far and watch the model estimate a purchase probability.")

    col1, col2, col3 = st.columns(3)
    with col1:
        num_clicks = st.number_input("Clicks", min_value=0, value=10, step=1)
        num_carts = st.number_input("Add-to-carts", min_value=0, value=0, step=1)
    with col2:
        num_unique_items = st.number_input("Unique items viewed", min_value=0, value=8, step=1)
        session_duration_seconds = st.number_input(
            "Session duration (seconds)", min_value=0.0, value=120.0, step=10.0
        )
    with col3:
        hour = st.slider("Last activity hour", min_value=0, max_value=23, value=18)
        weekday = st.selectbox(
            "Last activity weekday",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            index=4,
        )

    go_button = st.button("Predict purchase likelihood", type="primary", use_container_width=True)

    num_events = num_clicks + num_carts
    api_prediction = None
    if go_button:
        api_prediction = request_prediction(
            {
                "num_clicks": int(num_clicks),
                "num_carts": int(num_carts),
                "num_events": int(num_events),
                "num_unique_items": int(num_unique_items),
                "session_duration_seconds": float(session_duration_seconds),
                "hour": int(hour),
                "weekday": weekday,
            }
        )
        if api_prediction is None:
            st.error("The prediction service is unavailable. Start the API and try again.")
            return

    probability = float(api_prediction["probability"]) if api_prediction else 0.0
    probability_pct = probability * 100

    if api_prediction is None:
        st.plotly_chart(make_gauge(0), use_container_width=True, key="gauge_waiting")
        st.info("Enter the session details and click Predict to request a backend prediction.")
        return

    gauge_slot = st.empty()

    if go_button:
        # Animate the gauge sweeping up to the predicted value.
        frames = 20
        for step in range(1, frames + 1):
            gauge_slot.plotly_chart(
                make_gauge(probability_pct * step / frames),
                use_container_width=True,
                key=f"gauge_frame_{step}",
            )
            time.sleep(0.02)
        if probability >= 0.66:
            st.balloons()
            success_animation = load_lottie_url(LOTTIE_SUCCESS)
            if st_lottie is not None and success_animation is not None:
                st_lottie(success_animation, height=160, key="success_lottie")
    else:
        gauge_slot.plotly_chart(make_gauge(probability_pct), use_container_width=True, key="gauge_static")

    if probability >= 0.66:
        st.success(f" Likely to order — {probability:.1%} purchase probability")
    elif probability >= 0.33:
        st.warning(f" Uncertain — {probability:.1%} purchase probability, could go either way")
    else:
        st.info(f" Unlikely to order — {probability:.1%} purchase probability")

    explanation = api_prediction
    st.markdown('<div class="explanation-kicker">Model interpretation</div>', unsafe_allow_html=True)
    st.markdown('<div class="explanation-title">Why this result?</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="explanation-summary">{explanation["summary"]}</div>', unsafe_allow_html=True)

    with st.container(border=True):
        metric_cols = st.columns(4)
        metric_cols[0].metric("Purchase probability", f"{probability:.1%}")
        metric_cols[1].metric("Session duration", f"{session_duration_seconds:.0f} sec")
        metric_cols[2].metric("Last activity", f"{weekday[:3]} {hour:02d}:00")
        metric_cols[3].metric("Intent level", explanation["risk_level"].title())

        st.markdown("**Signals from this session**")
        for reason in explanation["reasons"]:
            st.markdown(f"- {reason}")

        company_col, customer_col = st.columns(2)
        with company_col:
            st.markdown("**Recommended company action**")
            st.info(explanation["company_action"])
        with customer_col:
            st.markdown("**Draft customer message**")
            st.info(explanation["customer_message"])

    st.caption(
        "Prediction source: FastAPI service"
        if api_prediction is not None
        else "Prediction source: local model (API unavailable)"
    )

    backend_model_name = model_info.get("model_name") if model_info else None
    if backend_model_name is not None:
        st.caption(f"Model used by the backend: **{backend_model_name}**")
    else:
        st.caption("Backend model information is unavailable.")


def render_performance_tab(results: dict | None, model_info: dict | None = None) -> None:
    if model_info is not None:
        st.subheader(f"Backend model: {model_info.get('model_name', 'Unknown')}")
        st.caption("These metrics belong to the model currently used by the prediction API.")
        metrics = model_info.get("test_metrics", {})
        if metrics:
            metric_columns = st.columns(len(metrics))
            for column, (name, value) in zip(metric_columns, metrics.items()):
                column.metric(name.replace("_", " ").title(), f"{value:.3f}")
        st.info("Model selection, training, and inference are owned by the backend service.")
        return

    if results is None:
        st.subheader("Backend model performance")
        st.info("Connect to the backend API to view the model used for predictions.")
        return

    st.subheader("Model comparison")
    st.caption(
        "Local comparison of three candidate models. The API prediction model is selected "
        "during the reproducible training pipeline and may differ from this comparison."
    )

    melted = results["comparison"].melt(
        id_vars="model",
        value_vars=["accuracy", "precision", "recall", "f1_score", "roc_auc"],
        var_name="metric",
        value_name="value",
    )
    melted["metric"] = melted["metric"].str.replace("_", " ").str.title()

    comparison_fig = px.bar(
        melted,
        x="metric",
        y="value",
        color="model",
        barmode="group",
        color_discrete_map=MODEL_COLORS,
        category_orders={"model": ["Random Forest", "Logistic Regression", "SVC"]},
    )
    comparison_fig.update_traces(hovertemplate="%{x}<br>%{y:.3f}<extra>%{fullData.name}</extra>")
    comparison_fig.update_yaxes(title=None, range=[0, 1])
    comparison_fig.update_xaxes(title=None)
    st.plotly_chart(style_fig(comparison_fig, height=360, showlegend=True), use_container_width=True)

    st.divider()

    col_matrix, col_importance = st.columns(2)

    with col_matrix:
        st.subheader(f"Confusion matrix — {results['best_name']}")
        labels_x = ["Predicted: No order", "Predicted: Order"]
        labels_y = ["Actual: No order", "Actual: Order"]
        matrix = results["matrix"]

        heat_fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=labels_x,
                y=labels_y,
                colorscale=[[0, SEQ_BLUE[0]], [1, SEQ_BLUE[5]]],
                showscale=False,
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            )
        )
        max_val = matrix.max()
        for i, y_label in enumerate(labels_y):
            for j, x_label in enumerate(labels_x):
                val = matrix[i][j]
                heat_fig.add_annotation(
                    x=x_label,
                    y=y_label,
                    text=str(val),
                    showarrow=False,
                    font=dict(
                        color="#ffffff" if val > max_val * 0.5 else PRIMARY_INK,
                        size=18,
                    ),
                )
        st.plotly_chart(style_fig(heat_fig, height=320), use_container_width=True)

    with col_importance:
        st.subheader("Feature importance")
        importances = results["importances"]
        if importances is not None:
            names = importances.index.str.replace("_", " ").str.title()
            importance_fig = go.Figure(
                go.Bar(x=importances.values, y=names, orientation="h", marker_color=BLUE)
            )
            importance_fig.update_xaxes(title=None)
            importance_fig.update_yaxes(title=None)
            st.plotly_chart(style_fig(importance_fig, height=320), use_container_width=True)
        else:
            st.info(f"{results['best_name']} doesn't expose feature importances directly.")

    st.subheader("Classification report")
    st.code(results["report"])


def render_analysis_tab(events: pd.DataFrame, session_data: pd.DataFrame) -> None:
    st.write(
        f"{len(events):,} events across {session_data.shape[0]:,} sessions "
        f"({session_data['target'].mean():.1%} of sessions placed an order)."
    )

    col_funnel, col_hour = st.columns(2)

    with col_funnel:
        st.subheader("Conversion funnel")
        event_counts = events["type"].value_counts()
        funnel_fig = go.Figure(
            go.Funnel(
                y=["Clicks", "Add to cart", "Orders"],
                x=[
                    int(event_counts.get("clicks", 0)),
                    int(event_counts.get("carts", 0)),
                    int(event_counts.get("orders", 0)),
                ],
                marker={"color": ORDINAL_BLUE},
                textinfo="value+percent initial",
                connector={"line": {"color": GRIDLINE, "width": 1}},
            )
        )
        st.plotly_chart(style_fig(funnel_fig, height=320), use_container_width=True)

    with col_hour:
        st.subheader("Events by hour of day")
        hour_counts = events.groupby("hour").size().reindex(range(24), fill_value=0)
        hour_fig = go.Figure(
            go.Scatter(
                x=hour_counts.index,
                y=hour_counts.values,
                mode="lines",
                line=dict(color=BLUE, width=2),
                fill="tozeroy",
                fillcolor="rgba(42,120,214,0.15)",
                hovertemplate="Hour %{x}:00<br>%{y:,} events<extra></extra>",
            )
        )
        hour_fig.update_xaxes(title="Hour", dtick=4)
        hour_fig.update_yaxes(title=None)
        st.plotly_chart(style_fig(hour_fig, height=320), use_container_width=True)

    col_weekday, col_items = st.columns(2)

    with col_weekday:
        st.subheader("Events by weekday")
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekday_counts = events["weekday"].value_counts().reindex(weekday_order, fill_value=0)
        weekday_fig = go.Figure(
            go.Bar(
                x=[day[:3] for day in weekday_order],
                y=weekday_counts.values,
                marker_color=BLUE,
                hovertemplate="%{x}<br>%{y:,} events<extra></extra>",
            )
        )
        weekday_fig.update_yaxes(title=None)
        st.plotly_chart(style_fig(weekday_fig, height=300), use_container_width=True)

    with col_items:
        st.subheader("Top 10 most-viewed items")
        top_items = events["aid"].value_counts().head(10).sort_values()
        items_fig = go.Figure(
            go.Bar(
                x=top_items.values,
                y=top_items.index.astype(str),
                orientation="h",
                marker_color=BLUE,
                hovertemplate="Item %{y}<br>%{x:,} events<extra></extra>",
            )
        )
        items_fig.update_xaxes(title=None)
        # Force a category axis: the item ids are numeric-looking strings, and
        # without this Plotly infers a linear axis and mangles the labels into
        # tick values like "0.5M" instead of showing each item id.
        items_fig.update_yaxes(title=None, type="category")
        st.plotly_chart(style_fig(items_fig, height=300), use_container_width=True)

    st.subheader("Session length distribution")
    session_lengths = events.groupby("session").size()
    hist_fig = go.Figure(go.Histogram(x=session_lengths, marker_color=BLUE, nbinsx=40))
    hist_fig.update_xaxes(title="Events per session")
    hist_fig.update_yaxes(title="Number of sessions")
    st.plotly_chart(style_fig(hist_fig, height=280), use_container_width=True)

    st.subheader("Session features (first 20 sessions)")
    st.dataframe(session_data.head(20), use_container_width=True)


def main() -> None:
    col_title, col_lottie = st.columns([5, 1])
    with col_title:
        st.title("🛒 E-commerce Purchase Predictor")
        st.caption(
            "Will this browsing session end in an order? Random Forest, Logistic "
            "Regression, and SVC trained on session-level clickstream behavior."
        )
        welcome_animation = load_lottie_url(LOTTIE_WELCOME)
        if st_lottie is not None and welcome_animation is not None:
            st_lottie(welcome_animation, height=110, key="welcome_lottie")

    if not DATA_PATH.exists():
        st.error(f"Couldn't find the training data at `{DATA_PATH}`.")
        st.stop()

    events = load_events()
    session_data = build_session_features(events)

    model_info = get_api_model_info()
    if model_info is None:
        st.warning("The backend API is not reachable. The dashboard will show analytics, but predictions require the API.")

    predict_tab, performance_tab, analysis_tab = st.tabs(
        [" Predict", " Model performance", " Analysis"]
    )

    with predict_tab:
        render_predict_tab(model_info)
    with performance_tab:
        render_performance_tab(None, model_info)
    with analysis_tab:
        render_analysis_tab(events, session_data)


if __name__ == "__main__":
    main()
"""Entry point for the Streamlit UI."""

import sys
from pathlib import Path

# Ensure project root and Frontend directory are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "Frontend"

for p in (str(PROJECT_ROOT), str(FRONTEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from Frontend.ui import main
except ImportError:
    try:
        from ui import main
    except ImportError:
        def main():
            import streamlit as st
            st.error("Frontend UI module not found.")

if __name__ == "__main__":
    main()