import json

import joblib

from .data_processing import FEATURE_COLUMNS


# the artifacts module is responsible for saving the trained model and its metadata to disk
def save_model(model, scaler, model_name, test_metrics, model_path, metadata_path):
    """Persist the trained model package and its evaluation metadata."""
    print("Saving model...")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "model_name": model_name}, model_path)
    print("Model saved to:", model_path)

    metadata = {
        "model_name": model_name,
        "requires_scaling": scaler is not None,
        "test_metrics": test_metrics,
        "feature_names": FEATURE_COLUMNS,
        "target": {"0": "No order", "1": "Order"},
        "data_leakage_prevention": [
            "Orders are never used as input features",
            "For sessions that order, only clicks/carts before the FIRST order are used",
            "Split is chronological (train = earliest sessions, test = most recent)",
            "Scaler is fit only on training data, then applied to val/test",
            "Test data is untouched during model selection and hyperparameter tuning",
        ],
    }
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
    print("Metadata saved to:", metadata_path)
