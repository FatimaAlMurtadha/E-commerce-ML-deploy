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