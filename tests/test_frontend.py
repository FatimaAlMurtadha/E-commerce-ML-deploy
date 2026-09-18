"""Frontend and UI test suite for the E-commerce Streamlit dashboard."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_FILE = PROJECT_ROOT / "Frontend" / "ui.py"
SRC_UI_FILE = PROJECT_ROOT / "src" / "ui.py"

# Dual-PR compatibility guard:
# If this branch does not have the frontend files, skip cleanly.
if not FRONTEND_FILE.exists() and not SRC_UI_FILE.exists():
    pytestmark = pytest.mark.skip(reason="Frontend code is not present on this branch.")

# Ensure Frontend directory is importable
if str(FRONTEND_FILE.parent) not in sys.path:
    sys.path.insert(0, str(FRONTEND_FILE.parent))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from Frontend import ui
except ImportError:
    import ui


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_events_df():
    """Synthetic event log matching the schema of events_10000_sessions.csv."""
    return pd.DataFrame([
        # Session 1: 3 clicks, 1 cart, ended with an order
        {"session": 1, "aid": 101, "ts": 1000, "type": "clicks", "hour": 10, "weekday": "Monday"},
        {"session": 1, "aid": 102, "ts": 1050, "type": "clicks", "hour": 10, "weekday": "Monday"},
        {"session": 1, "aid": 101, "ts": 1100, "type": "clicks", "hour": 10, "weekday": "Monday"},
        {"session": 1, "aid": 102, "ts": 1150, "type": "carts", "hour": 10, "weekday": "Monday"},
        {"session": 1, "aid": 102, "ts": 1200, "type": "orders", "hour": 10, "weekday": "Monday"},

        # Session 2: 2 clicks, no carts, no order
        {"session": 2, "aid": 201, "ts": 2000, "type": "clicks", "hour": 14, "weekday": "Tuesday"},
        {"session": 2, "aid": 202, "ts": 2050, "type": "clicks", "hour": 14, "weekday": "Tuesday"},

        # Session 3: 1 click, 2 carts, no order
        {"session": 3, "aid": 301, "ts": 3000, "type": "clicks", "hour": 16, "weekday": "Friday"},
        {"session": 3, "aid": 301, "ts": 3050, "type": "carts", "hour": 16, "weekday": "Friday"},
        {"session": 3, "aid": 302, "ts": 3100, "type": "carts", "hour": 16, "weekday": "Friday"},
    ])


@pytest.fixture
def synthetic_session_features():
    """Balanced synthetic session dataset for testing model training logic."""
    np.random.seed(42)
    n = 60
    clicks = np.random.randint(1, 30, size=n)
    carts = np.random.randint(0, 8, size=n)
    events = clicks + carts
    unique_items = np.random.randint(1, 15, size=n)
    # Higher carts -> higher chance of order
    probabilities = 1 / (1 + np.exp(-(0.6 * carts + 0.05 * clicks - 2)))
    target = (np.random.rand(n) < probabilities).astype(int)
    # Guarantee both classes are present
    target[0] = 1
    target[1] = 0

    return pd.DataFrame({
        "num_clicks": clicks,
        "num_carts": carts,
        "num_events": events,
        "num_unique_items": unique_items,
        "target": target,
    })


# ---------------------------------------------------------------------------
# Feature Engineering & Target Leakage Prevention Tests
# ---------------------------------------------------------------------------

def test_build_session_features_aggregates_clicks_carts_unique_items(sample_events_df):
    features = ui.build_session_features(sample_events_df)

    assert set(features.index) == {1, 2, 3}
    assert list(features.columns) == ["num_clicks", "num_carts", "num_events", "num_unique_items", "target"]

    # Session 1: 3 clicks, 1 cart, 2 unique items (101, 102)
    s1 = features.loc[1]
    assert s1["num_clicks"] == 3
    assert s1["num_carts"] == 1
    assert s1["num_unique_items"] == 2

    # Session 2: 2 clicks, 0 carts, 2 unique items
    s2 = features.loc[2]
    assert s2["num_clicks"] == 2
    assert s2["num_carts"] == 0
    assert s2["num_unique_items"] == 2


def test_build_session_features_prevents_label_leakage(sample_events_df):
    """
    CRITICAL ML INTEGRITY TEST:
    num_events must strictly equal num_clicks + num_carts.
    Raw order rows must NOT be included in num_events to prevent target leakage.
    """
    features = ui.build_session_features(sample_events_df)

    for session_id, row in features.iterrows():
        assert row["num_events"] == row["num_clicks"] + row["num_carts"], (
            f"Session {session_id} has num_events != clicks + carts, possible leakage"
        )

    # Session 1 had 3 clicks, 1 cart, 1 order (total 5 raw rows in events table)
    # num_events must be 4, NOT 5!
    assert features.loc[1]["num_events"] == 4


def test_build_session_features_assigns_correct_target(sample_events_df):
    features = ui.build_session_features(sample_events_df)

    # Session 1 had an order -> target = 1
    assert features.loc[1]["target"] == 1
    # Sessions 2 and 3 had no order -> target = 0
    assert features.loc[2]["target"] == 0
    assert features.loc[3]["target"] == 0


# ---------------------------------------------------------------------------
# Model Training & Evaluation Tests
# ---------------------------------------------------------------------------

def test_train_models_returns_required_output_structure(synthetic_session_features):
    results = ui.train_models(synthetic_session_features)

    expected_keys = {
        "best_name",
        "best_model",
        "best_needs_scaling",
        "scaler",
        "comparison",
        "matrix",
        "report",
        "importances",
    }
    assert expected_keys.issubset(results.keys())
    assert results["best_name"] in {"Random Forest", "Logistic Regression", "SVC"}
    assert results["matrix"].shape == (2, 2)
    assert isinstance(results["scaler"], StandardScaler)


def test_train_models_comparison_metrics_are_valid(synthetic_session_features):
    results = ui.train_models(synthetic_session_features)
    comparison = results["comparison"]

    assert set(comparison["model"]) == {"Random Forest", "Logistic Regression", "SVC"}
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    for col in metrics:
        assert col in comparison.columns
        # Every metric must be a valid probability or score in [0.0, 1.0]
        assert comparison[col].between(0.0, 1.0).all()


# ---------------------------------------------------------------------------
# Visualization & Helper Function Tests
# ---------------------------------------------------------------------------

def test_make_gauge_creates_valid_indicator():
    fig = ui.make_gauge(68.5)

    assert fig is not None
    data = fig.data[0]
    assert data.type == "indicator"
    assert data.mode == "gauge+number"
    assert data.value == 68.5
    assert data.number.suffix == "%"
    assert data.gauge.axis.range == (0, 100)
    assert data.gauge.threshold.value == 50


def test_style_fig_applies_layout_properties():
    import plotly.graph_objects as go
    raw_fig = go.Figure(go.Bar(x=[1, 2], y=[3, 4]))
    styled = ui.style_fig(raw_fig, height=250, showlegend=True)

    assert styled.layout.height == 250
    assert styled.layout.showlegend is True
    assert styled.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert styled.layout.plot_bgcolor == "rgba(0,0,0,0)"


def test_load_lottie_url_handles_empty_or_bad_urls():
    assert ui.load_lottie_url("") is None
    assert ui.load_lottie_url(None) is None


def test_load_lottie_url_handles_network_failure(monkeypatch):
    import requests
    ui.load_lottie_url.clear()

    def mock_get(*args, **kwargs):
        raise requests.RequestException("Connection timed out")

    monkeypatch.setattr(requests, "get", mock_get)
    result = ui.load_lottie_url("https://example.com/network_fail.json")
    assert result is None


def test_load_lottie_url_returns_parsed_json(monkeypatch):
    import requests
    ui.load_lottie_url.clear()

    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"v": "5.5.7", "fr": 30, "w": 500, "h": 500}

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: mock_resp)
    result = ui.load_lottie_url("https://example.com/valid.json")
    assert result == {"v": "5.5.7", "fr": 30, "w": 500, "h": 500}


# ---------------------------------------------------------------------------
# Streamlit App Execution Flow (Headless Simulation)
# ---------------------------------------------------------------------------

def test_streamlit_app_loads_headless():
    """Run the Streamlit app headlessly using AppTest."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(FRONTEND_FILE))
    at.run(timeout=30)

    # The app should complete execution without unhandled exceptions
    assert not at.exception

    # Verify key widgets exist
    number_inputs = at.number_input
    labels = [ni.label for ni in number_inputs]
    assert "Clicks" in labels
    assert "Add-to-carts" in labels
    assert "Unique items viewed" in labels

    # Verify the Predict button exists
    buttons = [b.label for b in at.button]
    assert any("Predict" in b for b in buttons)

    # Verify tabs exist
    tabs = at.tabs
    assert len(tabs) >= 3


def test_streamlit_app_predict_button_triggers_prediction():
    """Simulate user interaction: change inputs and trigger predict button."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(FRONTEND_FILE))
    at.run(timeout=30)

    # Set clicks to 15, carts to 3
    clicks_input = next(ni for ni in at.number_input if ni.label == "Clicks")
    carts_input = next(ni for ni in at.number_input if ni.label == "Add-to-carts")
    clicks_input.set_value(15)
    carts_input.set_value(3)

    # Click the predict button
    predict_button = next(b for b in at.button if "Predict" in b.label)
    predict_button.click().run()

    assert not at.exception
