import json

import pandas as pd
from fastapi.testclient import TestClient

from backend import api
from backend.config import META_PATH, MODEL_PATH
from backend.e_commerce_ml.data_processing import DatasetProcessor, FEATURE_COLUMNS


def test_checked_in_model_matches_metadata_contract():
    assert MODEL_PATH.exists(), f"Missing production model artifact: {MODEL_PATH}"
    assert META_PATH.exists(), f"Missing production model metadata: {META_PATH}"

    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    model = api.load_model(MODEL_PATH)

    assert model is not None
    assert metadata["model_name"] == "SVC"
    assert metadata["feature_names"] == FEATURE_COLUMNS
    assert list(model.feature_names_in_) == FEATURE_COLUMNS
    assert metadata["requires_scaling"] is True
    assert hasattr(model, "predict_proba")


def test_checked_in_model_predicts_on_real_training_data():
    events = DatasetProcessor().load_events(
        api.MODEL_PATH.parents[1] / "data" / "events_10000_sessions.csv"
    )
    session_features = DatasetProcessor().create_session_features(events)
    features, targets, _ = DatasetProcessor().prepare_features(session_features)
    model = api.load_model(MODEL_PATH)

    predictions = model.predict(features)
    probabilities = model.predict_proba(features)

    assert len(features) == len(targets) > 0
    assert predictions.shape == (len(features),)
    assert probabilities.shape == (len(features), 2)
    assert set(predictions).issubset({0, 1})
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
    assert not pd.isna(probabilities).any()


def test_api_predict_uses_checked_in_model():
    assert api.model is not None, "The API must load the checked-in production model"
    client = TestClient(api.app)

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
    result = response.json()
    assert result["prediction"] in (0, 1)
    assert result["order"] == (result["prediction"] == 1)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["risk_level"] in {"low", "medium", "high"}