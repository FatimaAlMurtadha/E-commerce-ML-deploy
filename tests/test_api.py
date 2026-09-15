"""Unit tests for the model API without requiring a trained artifact."""

import json
from pathlib import Path

import numpy as np
import pytest

from src import api


class FakeScaler:
	def __init__(self):
		self.received = None

	def transform(self, features):
		self.received = features
		return features


class FakeModel:
	def __init__(self, prediction=1, probability=0.8):
		self.prediction = prediction
		self.probability = probability
		self.received = None

	def predict(self, features):
		self.received = features
		return np.array([self.prediction])

	def predict_proba(self, features):
		return np.array([[1 - self.probability, self.probability]])


def test_load_model_package_raises_when_model_is_missing(monkeypatch):
	missing_path = Path("missing-model.joblib")
	monkeypatch.setattr(api, "MODEL_PATH", missing_path)

	with pytest.raises(FileNotFoundError, match="Model not found"):
		api.load_model_package()


def test_load_model_package_returns_serialized_package(monkeypatch, tmp_path):
	model_path = tmp_path / "model.joblib"
	model_path.write_bytes(b"model")
	package = {"model": FakeModel(), "scaler": None, "model_name": "test-model"}

	monkeypatch.setattr(api, "MODEL_PATH", model_path)
	monkeypatch.setattr(api.joblib, "load", lambda path: package)

	assert api.load_model_package() == package


def test_predict_session_scales_features_and_returns_prediction():
	scaler = FakeScaler()
	model = FakeModel(prediction=1, probability=0.8)
	package = {"model": model, "scaler": scaler, "model_name": "test-model"}
	session = {
		"num_clicks": 15,
		"num_carts": 3,
		"num_events": 20,
		"num_unique_items": 8,
	}

	result = api.predict_session(session, package)

	assert result == {
		"prediction": 1,
		"probability": 0.8,
		"model_name": "test-model",
		"interpretation": "Order likely",
	}
	assert list(scaler.received.columns) == list(session)
	assert model.received is scaler.received






"""  Later, we can add separate integration tests using:

A small real trained model fixture, or
A real Joblib model loaded from a temporary test file.
That would verify that the actual serialized model, scaler, metadata, and API work together.

ex. test_api_integration.py  // real Joblib artifact tests later
"""
