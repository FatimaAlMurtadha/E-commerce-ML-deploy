"""Legacy local model comparison retained for backwards-compatible tests.

The running Streamlit app uses the backend model and does not call this module.
"""

import pandas as pd
import streamlit as st
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from frontend.ui_config import MODEL_FEATURE_COLUMNS, SCALED_MODELS


@st.cache_resource(show_spinner="Training local comparison models...")
def train_models(session_data: pd.DataFrame) -> dict:
    x = session_data.reindex(columns=MODEL_FEATURE_COLUMNS, fill_value=0)
    y = session_data["target"]
    x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    x_train_val_scaled = scaler.fit_transform(x_train_val)
    x_test_scaled = scaler.transform(x_test)
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1),
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "SVC": CalibratedClassifierCV(
            SVC(class_weight="balanced", random_state=42),
            ensemble=False,
        ),
    }
    comparison_rows = []
    fitted = {}
    for name, model in models.items():
        train_data, test_data = (x_train_val_scaled, x_test_scaled) if name in SCALED_MODELS else (x_train_val, x_test)
        model.fit(train_data, y_train_val)
        pred = model.predict(test_data)
        prob = model.predict_proba(test_data)[:, 1]
        comparison_rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1_score": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, prob),
        })
        fitted[name] = model
    comparison = pd.DataFrame(comparison_rows).sort_values("f1_score", ascending=False)
    best_name = comparison.iloc[0]["model"]
    best_model = fitted[best_name]
    best_needs_scaling = best_name in SCALED_MODELS
    best_x_test = x_test_scaled if best_needs_scaling else x_test
    best_pred = best_model.predict(best_x_test)
    importances = None
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=MODEL_FEATURE_COLUMNS).sort_values()
    return {
        "best_name": best_name,
        "best_model": best_model,
        "best_needs_scaling": best_needs_scaling,
        "scaler": scaler,
        "comparison": comparison,
        "matrix": confusion_matrix(y_test, best_pred),
        "report": classification_report(y_test, best_pred, target_names=["No order", "Order"], zero_division=0),
        "importances": importances,
    }
