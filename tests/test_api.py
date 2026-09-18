"""Unit and contract tests for the E-commerce FastAPI application."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from src import api
from src.api import app, load_model, SessionInput, PredictionOutput


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Input Validation & Schema Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("missing_field", [
    "num_clicks",
    "num_carts",
    "num_events",
    "num_unique_items",
])
def test_predict_missing_required_field_returns_422(client, missing_field):
    payload = {
        "num_clicks": 5,
        "num_carts": 1,
        "num_events": 6,
        "num_unique_items": 3,
    }
    payload.pop(missing_field)

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    errors = response.json().get("detail", [])
    assert any(missing_field in err["loc"] for err in errors)


@pytest.mark.parametrize("invalid_field, invalid_value", [
    ("num_clicks", "five"),
    ("num_carts", "not-a-number"),
    ("num_events", None),
    ("session_duration_seconds", "long"),
])
def test_predict_invalid_data_types_return_422(client, invalid_field, invalid_value):
    payload = {
        "num_clicks": 1,
        "num_carts": 0,
        "num_events": 1,
        "num_unique_items": 1,
        invalid_field: invalid_value,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_accepts_extra_fields(client, monkeypatch):
    monkeypatch.setattr(api, "model", None)
    payload = {
        "num_clicks": 2,
        "num_carts": 0,
        "num_events": 2,
        "num_unique_items": 1,
        "extra_device_info": "mobile_safari",
        "custom_tag": 999,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_health_reports_model_name(client, monkeypatch):
    monkeypatch.setattr(api, "model", object())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_name"] == "SVC"


def test_model_info_reports_backend_artifact(client):
    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["model_name"] == "SVC"
    assert "test_metrics" in response.json()


def test_predict_zero_and_edge_values(client, monkeypatch):
    monkeypatch.setattr(api, "model", None)
    payload = {
        "num_clicks": 0,
        "num_carts": 0,
        "num_events": 0,
        "num_unique_items": 0,
        "session_duration_seconds": 0.0,
        "hour": 0,
        "weekday": 0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 0
    assert data["order"] is False
    assert data["probability"] == 0.05
    assert data["risk_level"] == "low"
    assert data["company_action"]
    assert data["customer_message"]


def test_predict_returns_reasons_and_action_for_high_intent_session(client, monkeypatch):
    monkeypatch.setattr(api, "model", None)
    response = client.post(
        "/predict",
        json={"num_clicks": 20, "num_carts": 2, "num_events": 22, "num_unique_items": 8},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["risk_level"] == "high"
    assert len(data["reasons"]) >= 3
    assert "cart" in data["reasons"][0].lower()
    assert "discount" in data["company_action"].lower()


# ---------------------------------------------------------------------------
# Fallback Predictor Logic
# ---------------------------------------------------------------------------

def test_fallback_probability_minimum_clamp(client, monkeypatch):
    monkeypatch.setattr(api, "model", None)
    # 0 * 0.3 + 0 * 0.02 = 0.0 -> clamped to 0.05
    response = client.post(
        "/predict",
        json={"num_clicks": 0, "num_carts": 0, "num_events": 0, "num_unique_items": 0},
    )
    assert response.status_code == 200
    assert response.json()["probability"] == 0.05
    assert response.json()["prediction"] == 0
    assert response.json()["order"] is False


def test_fallback_probability_maximum_clamp(client, monkeypatch):
    monkeypatch.setattr(api, "model", None)
    # 50 * 0.3 + 50 * 0.02 = 16.0 -> clamped to 0.95
    response = client.post(
        "/predict",
        json={"num_clicks": 50, "num_carts": 50, "num_events": 100, "num_unique_items": 20},
    )
    assert response.status_code == 200
    assert response.json()["probability"] == 0.95
    assert response.json()["prediction"] == 1
    assert response.json()["order"] is True


def test_fallback_probability_decision_boundary(client, monkeypatch):
    monkeypatch.setattr(api, "model", None)
    # 1 cart (0.30) + 10 clicks (0.20) = 0.50 -> pred = 1, order = True
    response_at_50 = client.post(
        "/predict",
        json={"num_clicks": 10, "num_carts": 1, "num_events": 11, "num_unique_items": 5},
    )
    assert response_at_50.status_code == 200
    assert response_at_50.json()["probability"] == 0.50
    assert response_at_50.json()["prediction"] == 1
    assert response_at_50.json()["order"] is True

    # 1 cart (0.30) + 9 clicks (0.18) = 0.48 -> pred = 0, order = False
    response_below_50 = client.post(
        "/predict",
        json={"num_clicks": 9, "num_carts": 1, "num_events": 10, "num_unique_items": 5},
    )
    assert response_below_50.status_code == 200
    assert response_below_50.json()["probability"] == 0.48
    assert response_below_50.json()["prediction"] == 0
    assert response_below_50.json()["order"] is False


# ---------------------------------------------------------------------------
# HTTP Method and Contract Tests
# ---------------------------------------------------------------------------

def test_predict_disallows_get_request(client):
    response = client.get("/predict")
    assert response.status_code == 405


def test_predict_disallows_put_request(client):
    response = client.put("/predict", json={})
    assert response.status_code == 405


def test_nonexistent_route_returns_404(client):
    response = client.get("/nonexistent_endpoint")
    assert response.status_code == 404


def test_malformed_json_returns_422(client):
    response = client.post(
        "/predict",
        content="this is not json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Model Loader Tests
# ---------------------------------------------------------------------------

def test_load_model_returns_none_for_missing_file(tmp_path):
    missing_file = tmp_path / "does_not_exist.joblib"
    result = load_model(missing_file)
    assert result is None


def test_load_model_returns_none_for_corrupt_file(tmp_path):
    corrupt_file = tmp_path / "corrupt.joblib"
    corrupt_file.write_text("not a valid pickle file")
    result = load_model(corrupt_file)
    assert result is None


def test_load_model_handles_dict_with_model_and_scaler(tmp_path):
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    target_file = tmp_path / "bundle.joblib"
    joblib.dump({"model": LogisticRegression(), "scaler": StandardScaler()}, target_file)

    loaded = load_model(target_file)
    assert isinstance(loaded, Pipeline)
    assert "scaler" in loaded.named_steps
    assert "classifier" in loaded.named_steps