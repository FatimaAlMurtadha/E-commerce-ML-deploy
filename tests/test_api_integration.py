"""Integration tests for the FastAPI endpoints."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src import api


@pytest.fixture
def client():
	return TestClient(api.app)


def test_root_endpoint_returns_api_status(client):
	response = client.get("/")

	assert response.status_code == 200
	assert response.json() == {"message": "E-commerce API", "status": "ok"}


def test_health_endpoint_reports_when_model_is_not_loaded(client, monkeypatch):
	monkeypatch.setattr(api, "model", None)

	response = client.get("/health")

	assert response.status_code == 200
	assert response.json() == {"status": "ok", "model_loaded": False}


def test_predict_endpoint_uses_fallback_for_low_order_probability(client, monkeypatch):
	monkeypatch.setattr(api, "model", None)

	response = client.post(
		"/predict",
		json={
			"num_clicks": 1,
			"num_carts": 0,
			"num_events": 2,
			"num_unique_items": 2,
		},
	)

	assert response.status_code == 200
	data = response.json()
	assert data["prediction"] == 0
	assert data["order"] is False
	assert data["probability"] == 0.05
	assert data["risk_level"] == "low"
	assert data["reasons"]
	assert data["company_action"]
	assert data["customer_message"]


def test_predict_endpoint_uses_fallback_for_high_order_probability(client, monkeypatch):
	monkeypatch.setattr(api, "model", None)

	response = client.post(
		"/predict",
		json={
			"num_clicks": 10,
			"num_carts": 10,
			"num_events": 20,
			"num_unique_items": 8,
		},
	)

	assert response.status_code == 200
	data = response.json()
	assert data["prediction"] == 1
	assert data["order"] is True
	assert data["probability"] == 0.95
	assert data["risk_level"] == "high"
	assert data["reasons"]
	assert data["company_action"]
	assert data["customer_message"]


def test_predict_endpoint_uses_loaded_model(client, monkeypatch):
	class FakeModel:
		def predict(self, features):
			assert list(features.columns) == [
				"num_clicks",
				"num_carts",
				"num_events",
				"num_unique_items",
			]
			return np.array([1])

		def predict_proba(self, features):
			return np.array([[0.1234, 0.8766]])

	monkeypatch.setattr(api, "model", FakeModel())

	response = client.post(
		"/predict",
		json={
			"num_clicks": 5,
			"num_carts": 1,
			"num_events": 10,
			"num_unique_items": 4,
		},
	)

	assert response.status_code == 200
	data = response.json()
	assert data["prediction"] == 1
	assert data["order"] is True
	assert data["probability"] == 0.8766
	assert data["risk_level"] == "high"
	assert data["reasons"]
	assert data["company_action"]
	assert data["customer_message"]


def test_predict_endpoint_rejects_incomplete_payload(client):
	response = client.post(
		"/predict",
		json={"num_clicks": 5, "num_carts": 1},
	)

	assert response.status_code == 422


def test_health_endpoint_reports_when_model_is_loaded(client, monkeypatch):
	class DummyModel:
		pass

	monkeypatch.setattr(api, "model", DummyModel())

	response = client.get("/health")

	assert response.status_code == 200
	assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_endpoint_with_persisted_model(client):
	# If the model artifact exists on disk, test real inference end-to-end
	if api.model is None:
		pytest.skip("Persisted model is not loaded in api.model")

	response = client.post(
		"/predict",
		json={
			"num_clicks": 4,
			"num_carts": 2,
			"num_events": 6,
			"num_unique_items": 3,
			"session_duration_seconds": 95.0,
			"hour": 14,
			"weekday": "Wednesday",
		},
	)

	assert response.status_code == 200
	data = response.json()
	assert data["prediction"] in (0, 1)
	assert data["order"] == (data["prediction"] == 1)
	assert 0.0 <= data["probability"] <= 1.0


def test_predict_endpoint_handles_weekday_string(client, monkeypatch):
	received_data = {}

	class SpyModel:
		feature_names_in_ = np.array([
			"num_clicks", "num_carts", "num_events", "num_unique_items",
			"session_duration_seconds", "hour", "weekday"
		])

		def predict(self, df):
			received_data.update(df.iloc[0].to_dict())
			return np.array([1])

		def predict_proba(self, df):
			return np.array([[0.25, 0.75]])

	monkeypatch.setattr(api, "model", SpyModel())

	response = client.post(
		"/predict",
		json={
			"num_clicks": 3,
			"num_carts": 1,
			"num_events": 4,
			"num_unique_items": 2,
			"weekday": "Monday",
		},
	)

	assert response.status_code == 200
	# "Monday" maps to 0 in WEEKDAY_MAP
	assert received_data["weekday"] == 0
	# missing optional fields should take defaults
	assert received_data["session_duration_seconds"] == 0.0
	assert received_data["hour"] == 12


def test_cors_preflight_headers(client):
	response = client.options(
		"/predict",
		headers={
			"Origin": "http://localhost:8501",
			"Access-Control-Request-Method": "POST",
		},
	)
	assert response.status_code == 200
	assert response.headers.get("access-control-allow-origin") in ("*", "http://localhost:8501")