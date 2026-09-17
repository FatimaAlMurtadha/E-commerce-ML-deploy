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
	assert response.json() == {
		"prediction": 0,
		"order": False,
		"probability": 0.05,
	}


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
	assert response.json() == {
		"prediction": 1,
		"order": True,
		"probability": 0.95,
	}


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
	assert response.json() == {
		"prediction": 1,
		"order": True,
		"probability": 0.8766,
	}


def test_predict_endpoint_rejects_incomplete_payload(client):
	response = client.post(
		"/predict",
		json={"num_clicks": 5, "num_carts": 1},
	)

	assert response.status_code == 422
