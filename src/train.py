import json
import warnings
from pathlib import Path

import joblib
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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

from config import MODEL_PATH, META_PATH, MODEL_DIR

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



# 1. Load data


def load_data(data_path):
    print("Loading data...")
    df = pd.read_csv(data_path)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    return df



# 2. Build leakage-safe features and target
#
# The key leakage fix: for a session that eventually orders,
# we only use the clicks/carts that happened BEFORE the first
# order in that session. Events after the first order (and the
# order event itself) are never used as features. Sessions that
# never order use all their clicks/carts.


def create_features_and_target(df):
    print("Creating leakage-safe features and target...")

    df = df.copy()
    df = df.sort_values(["session", "ts"])

    rows = []

    for session_id, session_data in df.groupby("session"):
        session_data = session_data.sort_values("ts")

        order_mask = session_data["type"] == "orders"
        has_order = order_mask.any()

        if has_order:
            target = 1
            first_order_pos = order_mask.values.argmax()
            # Only events strictly before the first order
            feature_data = session_data.iloc[:first_order_pos]
        else:
            target = 0
            feature_data = session_data

        # Never let an order event leak into the features
        feature_data = feature_data[feature_data["type"].isin(["clicks", "carts"])]

        if len(feature_data) == 0:
            # No pre-order browsing signal at all — nothing to predict from
            continue

        num_clicks = int((feature_data["type"] == "clicks").sum())
        num_carts = int((feature_data["type"] == "carts").sum())
        num_events = len(feature_data)
        num_unique_items = feature_data["aid"].nunique()

        first_ts = feature_data["ts"].min()
        last_ts = feature_data["ts"].max()
        session_duration_seconds = (last_ts - first_ts) / 1000.0  # ts is epoch ms

        hour = int(feature_data["hour"].iloc[-1])
        weekday = feature_data["weekday"].iloc[-1]

        rows.append(
            {
                "session": session_id,
                "num_clicks": num_clicks,
                "num_carts": num_carts,
                "num_events": num_events,
                "num_unique_items": num_unique_items,
                "session_duration_seconds": session_duration_seconds,
                "hour": hour,
                "weekday": weekday,
                "target": target,
                "prediction_timestamp": last_ts,
            }
        )

    session_features = pd.DataFrame(rows)
    session_features = session_features.sort_values("prediction_timestamp")

    print("Number of sessions:", len(session_features))
    print("Number of features:", len(FEATURE_COLUMNS))
    print("Target distribution:")
    print(session_features["target"].value_counts())

    return session_features



# 3. Prepare X / y


def prepare_features(session_features):
    print("Preparing features...")

    data = session_features.copy()
    data["weekday"] = data["weekday"].map(WEEKDAY_MAP).fillna(0)

    x = data[FEATURE_COLUMNS]
    y = data["target"]
    prediction_timestamps = data["prediction_timestamp"]

    return x, y, prediction_timestamps



# 4. Split data chronologically
#
# A random split can put a session that happened AFTER another
# session into the training set while the earlier one ends up
# in test. That is a subtle form of leakage for time-ordered
# clickstream data, so we split by time instead: earliest
# sessions train, most recent sessions test.


def split_data(x, y, prediction_timestamps):
    print("Splitting data chronologically...")

    data = x.copy()
    data["target"] = y.values
    data["prediction_timestamp"] = prediction_timestamps.values
    data = data.sort_values("prediction_timestamp")

    n = len(data)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)

    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]

    x_train, y_train = train_data[FEATURE_COLUMNS], train_data["target"]
    x_val, y_val = val_data[FEATURE_COLUMNS], val_data["target"]
    x_test, y_test = test_data[FEATURE_COLUMNS], test_data["target"]

    x_train_val = pd.concat([x_train, x_val])
    y_train_val = pd.concat([y_train, y_val])

    print("Train rows:", len(x_train))
    print("Validation rows:", len(x_val))
    print("Test rows:", len(x_test))

    return x_train, x_val, x_test, y_train, y_val, y_test, x_train_val, y_train_val


# 5. Scale data — fit only on training data


def scale_data(x_train, x_val, x_test):
    print("Scaling data...")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_val_scaled, x_test_scaled, scaler



# 6. Evaluate model


def evaluate_model(model_name, y_true, predictions, probabilities):
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1_score": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
    }



# 7. Train models


def train_models(x_train_scaled, x_train, x_val_scaled, x_val, y_train, y_val):
    print("Training models...")
    models_data = {}

    # Logistic Regression
    print("Training Logistic Regression...")
    log_reg = LogisticRegression(class_weight="balanced", random_state=42)
    log_reg_params = {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l2"],
        "solver": ["lbfgs", "liblinear"],
        "max_iter": [500, 1000, 2000],
    }
    log_grid = GridSearchCV(log_reg, log_reg_params, scoring="f1", cv=5, n_jobs=-1)
    log_grid.fit(x_train_scaled, y_train)
    best_log_reg = log_grid.best_estimator_
    log_val_pred = best_log_reg.predict(x_val_scaled)
    log_val_prob = best_log_reg.predict_proba(x_val_scaled)[:, 1]
    print("Best Logistic Regression parameters:", log_grid.best_params_)
    models_data["logreg"] = {
        "model": best_log_reg, "scaler": True,
        "predictions": log_val_pred, "probabilities": log_val_prob,
    }

    # SVC
    print("Training SVC...")
    svc = SVC(class_weight="balanced", probability=True, random_state=42)
    svc_params = {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
    }
    svc_grid = GridSearchCV(svc, svc_params, scoring="f1", cv=5, n_jobs=-1)
    svc_grid.fit(x_train_scaled, y_train)
    best_svc = svc_grid.best_estimator_
    svc_val_pred = best_svc.predict(x_val_scaled)
    svc_val_prob = best_svc.predict_proba(x_val_scaled)[:, 1]
    print("Best SVC parameters:", svc_grid.best_params_)
    models_data["svc"] = {
        "model": best_svc, "scaler": True,
        "predictions": svc_val_pred, "probabilities": svc_val_prob,
    }

    # Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
    rf_params = {
        "n_estimators": [200, 300, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    rf_grid = GridSearchCV(rf, rf_params, scoring="f1", cv=5, n_jobs=-1)
    rf_grid.fit(x_train, y_train)
    best_rf = rf_grid.best_estimator_
    rf_val_pred = best_rf.predict(x_val)
    rf_val_prob = best_rf.predict_proba(x_val)[:, 1]
    print("Best Random Forest parameters:", rf_grid.best_params_)
    models_data["rf"] = {
        "model": best_rf, "scaler": False,
        "predictions": rf_val_pred, "probabilities": rf_val_prob,
    }

    results = pd.DataFrame([
        evaluate_model("Logistic Regression", y_val, log_val_pred, log_val_prob),
        evaluate_model("SVC", y_val, svc_val_pred, svc_val_prob),
        evaluate_model("Random Forest", y_val, rf_val_pred, rf_val_prob),
    ]).sort_values("f1_score", ascending=False)

    print("\nValidation results:")
    print(results.to_string(index=False))

    return models_data, results


# 8. Select best model and retrain on train+val


def select_and_retrain_best_model(
    models_data, results, x_test, y_test, x_train_val, y_train_val,
):
    print("Selecting best model...")
    best_model_name = results.iloc[0]["model"]
    print("Best model:", best_model_name)

    if best_model_name == "Logistic Regression":
        print("Retraining Logistic Regression...")
        final_scaler = StandardScaler()
        x_train_val_final = final_scaler.fit_transform(x_train_val)
        x_test_final = final_scaler.transform(x_test)
        best_model = models_data["logreg"]["model"]
    elif best_model_name == "SVC":
        print("Retraining SVC...")
        final_scaler = StandardScaler()
        x_train_val_final = final_scaler.fit_transform(x_train_val)
        x_test_final = final_scaler.transform(x_test)
        best_model = models_data["svc"]["model"]
    else:
        print("Retraining Random Forest...")
        final_scaler = None
        x_train_val_final = x_train_val
        x_test_final = x_test
        best_model = models_data["rf"]["model"]

    best_model.fit(x_train_val_final, y_train_val)

    best_test_pred = best_model.predict(x_test_final)
    best_test_prob = best_model.predict_proba(x_test_final)[:, 1]

    test_metrics = {
        "accuracy": accuracy_score(y_test, best_test_pred),
        "precision": precision_score(y_test, best_test_pred, zero_division=0),
        "recall": recall_score(y_test, best_test_pred, zero_division=0),
        "f1_score": f1_score(y_test, best_test_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, best_test_prob),
        "average_precision": average_precision_score(y_test, best_test_prob),
    }

    print("\nClassification report:")
    print(classification_report(
        y_test, best_test_pred, target_names=["No order", "Order"], zero_division=0,
    ))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, best_test_pred))

    print("\nTest metrics:")
    for metric, value in test_metrics.items():
        print(metric, ":", round(value, 4))

    return best_model, final_scaler, best_model_name, test_metrics


# 9. Save model


def save_model(model, scaler, model_name, test_metrics):
    print("Saving model...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_package = {"model": model, "scaler": scaler, "model_name": model_name}
    joblib.dump(model_package, MODEL_PATH)
    print("Model saved to:", MODEL_PATH)

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
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print("Metadata saved to:", META_PATH)



# 10. Full pipeline


def train_pipeline(data_path=None):
    if data_path is None:
        data_path = (
            Path(__file__).resolve().parents[1] / "data" / "events_10000_sessions.csv"
        )

    print("Starting training pipeline")
    print("-------------------------")

    df = load_data(data_path)
    session_features = create_features_and_target(df)
    x, y, prediction_timestamps = prepare_features(session_features)

    (x_train, x_val, x_test, y_train, y_val, y_test,
     x_train_val, y_train_val) = split_data(x, y, prediction_timestamps)

    x_train_scaled, x_val_scaled, x_test_scaled, _ = scale_data(x_train, x_val, x_test)

    models_data, results = train_models(
        x_train_scaled, x_train, x_val_scaled, x_val, y_train, y_val,
    )

    best_model, final_scaler, model_name, test_metrics = select_and_retrain_best_model(
        models_data, results, x_test, y_test, x_train_val, y_train_val,
    )

    save_model(best_model, final_scaler, model_name, test_metrics)

    print("-------------------------")
    print("Training completed!")
    print("Best model:", model_name)
    return best_model, final_scaler


if __name__ == "__main__":
    train_pipeline()