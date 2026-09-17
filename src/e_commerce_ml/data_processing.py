from dataclasses import dataclass
from pathlib import Path

import pandas as pd

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


class DatasetProcessor:
    """Prepare session features and chronological train/validation/test data."""

    def load_events(self, data_path: str | Path) -> pd.DataFrame:
        print("Loading data...")
        events = pd.read_csv(data_path)
        print("Rows:", len(events))
        print("Columns:", list(events.columns))
        return events

    def create_session_features(self, events: pd.DataFrame) -> pd.DataFrame:
        print("Creating features and target...")
        events = events.copy().sort_values(["session", "ts"])
        rows = []

        for session_id, session_events in events.groupby("session"):
            session_events = session_events.sort_values("ts")
            order_mask = session_events["type"] == "orders"

            if order_mask.any():
                target = 1
                first_order_position = order_mask.values.argmax()
                feature_events = session_events.iloc[:first_order_position]
            else:
                target = 0
                feature_events = session_events

            feature_events = feature_events[
                feature_events["type"].isin(["clicks", "carts"])
            ]
            if feature_events.empty:
                continue

            first_timestamp = feature_events["ts"].min()
            last_timestamp = feature_events["ts"].max()
            rows.append(
                {
                    "session": session_id,
                    "num_clicks": int((feature_events["type"] == "clicks").sum()),
                    "num_carts": int((feature_events["type"] == "carts").sum()),
                    "num_events": len(feature_events),
                    "num_unique_items": feature_events["aid"].nunique(),
                    "session_duration_seconds": (
                        last_timestamp - first_timestamp
                    )
                    / 1000.0,
                    "hour": int(feature_events["hour"].iloc[-1]),
                    "weekday": feature_events["weekday"].iloc[-1],
                    "target": target,
                    "prediction_timestamp": last_timestamp,
                }
            )

        session_features = pd.DataFrame(rows).sort_values("prediction_timestamp")
        print("Number of sessions:", len(session_features))
        print("Number of features:", len(FEATURE_COLUMNS))
        print("Target distribution:")
        print(session_features["target"].value_counts())
        return session_features

    def prepare_features(self, session_features: pd.DataFrame):
        print("Preparing features...")
        data = session_features.copy()
        data["weekday"] = data["weekday"].map(WEEKDAY_MAP).fillna(0)
        return data[FEATURE_COLUMNS], data["target"], data["prediction_timestamp"]

    def split_data(self, x, y, prediction_timestamps) -> DatasetSplit:
        print("Splitting data train and test...")
        data = x.copy()
        data["target"] = y.values
        data["prediction_timestamp"] = prediction_timestamps.values
        data = data.sort_values("prediction_timestamp")

        train_end = int(len(data) * 0.60)
        validation_end = int(len(data) * 0.80)
        train_data = data.iloc[:train_end]
        validation_data = data.iloc[train_end:validation_end]
        test_data = data.iloc[validation_end:]

        split = DatasetSplit(
            x_train=train_data[FEATURE_COLUMNS],
            x_val=validation_data[FEATURE_COLUMNS],
            x_test=test_data[FEATURE_COLUMNS],
            y_train=train_data["target"],
            y_val=validation_data["target"],
            y_test=test_data["target"],
            x_train_val=pd.concat(
                [train_data[FEATURE_COLUMNS], validation_data[FEATURE_COLUMNS]]
            ),
            y_train_val=pd.concat([train_data["target"], validation_data["target"]]),
        )
        print("Train rows:", len(split.x_train))
        print("Validation rows:", len(split.x_val))
        print("Test rows:", len(split.x_test))
        return split
