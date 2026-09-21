"""Streamlit views for prediction, backend metrics, and data analysis."""

import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from frontend.ui_charts import make_gauge, style_fig
from frontend.ui_config import BLUE, FONT_FAMILY, GRIDLINE, MODEL_COLORS, MUTED_INK, ORDINAL_BLUE, PRIMARY_INK, SEQ_BLUE, WEEKDAYS
from frontend.ui_services import request_prediction


def render_predict_tab(model_info: dict | None, load_lottie_url, success_animation_url: str) -> None:
    st.subheader("Try a session")
    st.write("Describe a session so far and watch the backend estimate a purchase probability.")
    col1, col2, col3 = st.columns(3)
    with col1:
        num_clicks = st.number_input("Clicks", min_value=0, value=10, step=1)
        num_carts = st.number_input("Add-to-carts", min_value=0, value=0, step=1)
    with col2:
        num_unique_items = st.number_input("Unique items viewed", min_value=0, value=8, step=1)
        session_duration_seconds = st.number_input("Session duration (seconds)", min_value=0.0, value=120.0, step=10.0)
    with col3:
        hour = st.slider("Last activity hour", min_value=0, max_value=23, value=18)
        weekday = st.selectbox("Last activity weekday", WEEKDAYS, index=4)

    go_button = st.button("Predict purchase likelihood", type="primary", use_container_width=True)
    num_events = num_clicks + num_carts
    prediction = None
    if go_button:
        prediction = request_prediction({
            "num_clicks": int(num_clicks), "num_carts": int(num_carts), "num_events": int(num_events),
            "num_unique_items": int(num_unique_items), "session_duration_seconds": float(session_duration_seconds),
            "hour": int(hour), "weekday": weekday,
        })
        if prediction is None:
            st.error("The prediction service is unavailable. Start the API and try again.")
            return
    if prediction is None:
        st.plotly_chart(make_gauge(0), use_container_width=True, key="gauge_waiting")
        st.info("Enter the session details and click Predict to request a backend prediction.")
        return

    probability = float(prediction["probability"])
    gauge_slot = st.empty()
    if go_button:
        for step in range(1, 21):
            gauge_slot.plotly_chart(make_gauge(probability * 100 * step / 20), use_container_width=True, key=f"gauge_frame_{step}")
            time.sleep(0.02)
    else:
        gauge_slot.plotly_chart(make_gauge(probability * 100), use_container_width=True, key="gauge_static")

    if probability >= 0.66:
        st.success(f"Likely to order - {probability:.1%} purchase probability")
    elif probability >= 0.33:
        st.warning(f"Uncertain - {probability:.1%} purchase probability")
    else:
        st.info(f"Unlikely to order - {probability:.1%} purchase probability")

    st.markdown('<div class="explanation-kicker">Model interpretation</div>', unsafe_allow_html=True)
    st.markdown('<div class="explanation-title">Why this result?</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="explanation-summary">{prediction["summary"]}</div>', unsafe_allow_html=True)
    with st.container(border=True):
        metrics = st.columns(4)
        metrics[0].metric("Purchase probability", f"{probability:.1%}")
        metrics[1].metric("Session duration", f"{session_duration_seconds:.0f} sec")
        metrics[2].metric("Last activity", f"{weekday[:3]} {hour:02d}:00")
        metrics[3].metric("Intent level", prediction["risk_level"].title())
        st.markdown("**Signals from this session**")
        for reason in prediction["reasons"]:
            st.markdown(f"- {reason}")
        company, customer = st.columns(2)
        with company:
            st.markdown("**Recommended company action**")
            st.info(prediction["company_action"])
        with customer:
            st.markdown("**Draft customer message**")
            st.info(prediction["customer_message"])
    st.caption(f"Prediction source: FastAPI service | Backend model: **{model_info.get('model_name', 'unknown') if model_info else 'unknown'}**")


def render_performance_tab(model_info: dict | None) -> None:
    st.subheader(f"Backend model: {model_info.get('model_name', 'Unknown') if model_info else 'Unavailable'}")
    if model_info is None:
        st.info("Connect to the backend API to view model metrics.")
        return
    st.caption("These metrics belong to the model currently used by the prediction API.")
    metrics = model_info.get("test_metrics", {})
    columns = st.columns(len(metrics)) if metrics else []
    for column, (name, value) in zip(columns, metrics.items()):
        column.metric(name.replace("_", " ").title(), f"{value:.3f}")
    st.info("Model selection, training, and inference are owned by the backend service.")


def render_analysis_tab(events: pd.DataFrame, session_data: pd.DataFrame, style_figure=style_fig) -> None:
    st.write(f"{len(events):,} events across {session_data.shape[0]:,} sessions ({session_data['target'].mean():.1%} of sessions placed an order).")
    funnel, hour = st.columns(2)
    with funnel:
        st.subheader("Conversion funnel")
        counts = events["type"].value_counts()
        figure = go.Figure(go.Funnel(y=["Clicks", "Add to cart", "Orders"], x=[int(counts.get("clicks", 0)), int(counts.get("carts", 0)), int(counts.get("orders", 0))], marker={"color": ORDINAL_BLUE}, textinfo="value+percent initial", connector={"line": {"color": GRIDLINE, "width": 1}}))
        st.plotly_chart(style_figure(figure, height=320), use_container_width=True)
    with hour:
        st.subheader("Events by hour of day")
        counts = events.groupby("hour").size().reindex(range(24), fill_value=0)
        figure = go.Figure(go.Scatter(x=counts.index, y=counts.values, mode="lines", line=dict(color=BLUE, width=2), fill="tozeroy", fillcolor="rgba(42,120,214,0.15)"))
        figure.update_xaxes(title="Hour", dtick=4)
        st.plotly_chart(style_figure(figure, height=320), use_container_width=True)
    weekday, items = st.columns(2)
    with weekday:
        st.subheader("Events by weekday")
        counts = events["weekday"].value_counts().reindex(WEEKDAYS, fill_value=0)
        figure = go.Figure(go.Bar(x=[day[:3] for day in WEEKDAYS], y=counts.values, marker_color=BLUE))
        st.plotly_chart(style_figure(figure, height=300), use_container_width=True)
    with items:
        st.subheader("Top 10 most-viewed items")
        counts = events["aid"].value_counts().head(10).sort_values()
        figure = go.Figure(go.Bar(x=counts.values, y=counts.index.astype(str), orientation="h", marker_color=BLUE))
        figure.update_yaxes(type="category")
        st.plotly_chart(style_figure(figure, height=300), use_container_width=True)
    st.subheader("Session length distribution")
    st.plotly_chart(style_figure(go.Figure(go.Histogram(x=events.groupby("session").size(), marker_color=BLUE, nbinsx=40)), height=280), use_container_width=True)
    st.subheader("Session features (first 20 sessions)")
    st.dataframe(session_data.head(20), use_container_width=True)
