"""External data and backend service access for the Streamlit UI."""

import pandas as pd
import requests
import streamlit as st

from frontend.ui_config import API_URL, DATA_PATH


@st.cache_data(show_spinner=False, ttl=3600)
def load_lottie_url(url: str):
    if not url:
        return None
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError):
        return None


@st.cache_data(show_spinner=False)
def load_events() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def request_prediction(payload: dict) -> dict | None:
    """Request a prediction from the backend domain service."""
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=0.5)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError):
        return None


@st.cache_data(show_spinner=False, ttl=30)
def get_api_model_info() -> dict | None:
    try:
        response = requests.get(f"{API_URL}/model-info", timeout=0.5)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError):
        return None
