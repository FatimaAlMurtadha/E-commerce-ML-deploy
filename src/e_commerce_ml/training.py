import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from src.config import META_PATH, MODEL_DIR, MODEL_PATH
except ImportError:
    from config import META_PATH, MODEL_DIR, MODEL_PATH

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

FEATURE_COLUMNS = [
    "num_clicks",
    "num_carts",
    "num_events",
    "num_unique_items",
    "session_duration_seconds",
    "hour",
    "weekday",
]

WEEKDAY_MAP = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}

# create a dataclass to hold the split data

@dataclass
class DatasetSplit:
    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    x_train_val: pd.DataFrame
    y_train_val: pd.Series


class SessionFeatureBuilder:
    """Convert raw event rows into one labelled row per usable session."""

    def build(self, events: pd.DataFrame) -> pd.DataFrame:
        print("Creating features and target...")
        events = events.copy().sort_values(["session", "ts"])
        rows = []

        for session_id, session_data in events.groupby("session"):
            session_data = session_data.sort_values("ts")
            order_mask = session_data["type"] == "orders"

            if order_mask.any():
                target = 1
                first_order_pos = order_mask.values.argmax()
                feature_data = session_data.iloc[:first_order_pos]
            else:
                target = 0
                feature_data = session_data

            feature_data = feature_data[feature_data["type"].isin(["clicks", "carts"])]
            if len(feature_data) == 0:
                continue

            first_ts = feature_data["ts"].min()
            last_ts = feature_data["ts"].max()
            rows.append(
                {
                    "session": session_id,
                    "num_clicks": int((feature_data["type"] == "clicks").sum()),
                    "num_carts": int((feature_data["type"] == "carts").sum()),
                    "num_events": len(feature_data),
                    "num_unique_items": feature_data["aid"].nunique(),
                    "session_duration_seconds": (last_ts - first_ts) / 1000.0,
                    "hour": int(feature_data["hour"].iloc[-1]),
                    "weekday": feature_data["weekday"].iloc[-1],
                    "target": target,
                    "prediction_timestamp": last_ts,
                }
            )

        session_features = pd.DataFrame(rows).sort_values("prediction_timestamp")
        print("Number of sessions:", len(session_features))
        print("Number of features:", len(FEATURE_COLUMNS))
        print("Target distribution:")
        print(session_features["target"].value_counts())
        return session_features
