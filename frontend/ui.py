"""Streamlit application entrypoint and backwards-compatible UI exports."""

import streamlit as st

try:
    from streamlit_lottie import st_lottie
except ModuleNotFoundError:
    st_lottie = None

from backend.e_commerce_ml.explanations import explain_prediction
from frontend.ui_charts import make_gauge, style_fig
from frontend.ui_config import DATA_PATH, LOTTIE_SUCCESS, LOTTIE_WELCOME
from frontend.ui_features import build_session_features
from frontend.ui_model_legacy import train_models
from frontend.ui_services import get_api_model_info, load_events, load_lottie_url
from frontend.ui_views import render_analysis_tab, render_performance_tab, render_predict_tab


st.set_page_config(page_title="E-commerce Purchase Predictor", page_icon="🛒", layout="wide")
st.markdown(
    """
    <style>
    .explanation-kicker { color: #2a78d6; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: -0.45rem; }
    .explanation-title { color: #0b0b0b; font-size: 1.45rem; font-weight: 750; margin-bottom: 0.2rem; }
    .explanation-summary { color: #52514e; font-size: 1rem; margin-bottom: 0.7rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def main() -> None:
    st.title("E-commerce Purchase Predictor")
    st.caption("Backend-powered purchase prediction from session-level clickstream behavior.")
    welcome_animation = load_lottie_url(LOTTIE_WELCOME)
    if st_lottie is not None and welcome_animation is not None:
        st_lottie(welcome_animation, height=110, key="welcome_lottie")

    if not DATA_PATH.exists():
        st.error(f"Couldn't find the training data at `{DATA_PATH}`.")
        st.stop()

    events = load_events()
    model_info = get_api_model_info()
    if model_info is None:
        st.warning("The backend API is not reachable. Predictions require the API.")

    predict_tab, performance_tab, analysis_tab = st.tabs(["Predict", "Model performance", "Analysis"])
    with predict_tab:
        render_predict_tab(model_info, load_lottie_url, LOTTIE_SUCCESS)
    with performance_tab:
        render_performance_tab(model_info)
    with analysis_tab:
        if st.button("Load analysis", type="secondary"):
            session_data = build_session_features(events)
            render_analysis_tab(events, session_data)
        else:
            st.info("Load the analysis when you need session-level charts and metrics.")


if __name__ == "__main__":
    main()
