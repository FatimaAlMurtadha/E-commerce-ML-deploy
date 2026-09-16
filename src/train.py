# need to convert following code into an mlflow training script

"""import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)

# =========================
# 1) Create features and target variable
# =========================

# Is there an order in the session? (1 if yes, 0 if no)
session_target = (
    df_original.groupby("session")["type"]
    .apply(lambda values: ("orders" in values.values))
    .astype(int)
)

# Aggregate browsing events (excluding orders to prevent target leakage)
df_browsing = df_original[df_original["type"] != "orders"]
session_features = df_browsing.groupby("session").agg(
    num_clicks=("type", lambda x: (x == "clicks").sum()),
    num_carts=("type", lambda x: (x == "carts").sum()),
    num_events=("type", "count"),
    num_unique_items=("aid", "nunique"),
)

session_features["target"] = session_target

x = session_features.drop(columns=["target"])
y = session_features["target"]

# =========================
# 2) Train-test split
# =========================

x_train_val, x_test, y_train_val, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

x_train, x_val, y_train, y_val = train_test_split(
    x_train_val, y_train_val, test_size=0.15, random_state=40, stratify=y_train_val
)

# =========================
# 3) Scaling only for train and validation sets (not for test set)
# =========================

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)
x_test_scaled = scaler.transform(x_test)

# =========================
# 4) Models + GridSearchCV
# =========================

# ---- Logistic Regression ----
log_reg_base = LogisticRegression(class_weight="balanced", random_state=42)

param_grid_logreg = {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l2"],
    "solver": ["lbfgs", "liblinear"],
    "max_iter": [500, 1000, 2000],
}

grid_log = GridSearchCV(
    estimator=log_reg_base,
    param_grid=param_grid_logreg,
    scoring="f1",
    cv=5,
    n_jobs=-1,
)

grid_log.fit(x_train_scaled, y_train)
best_log_reg = grid_log.best_estimator_

log_val_pred = best_log_reg.predict(x_val_scaled)
log_val_prob = best_log_reg.predict_proba(x_val_scaled)[:, 1]

# ---- SVC + CalibratedClassifierCV ----
svc_base = SVC(class_weight="balanced", probability=True, random_state=42)

param_grid_svc = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"],
}

grid_svc = GridSearchCV(
    estimator=svc_base,
    param_grid=param_grid_svc,
    scoring="f1",
    cv=5,
    n_jobs=-1,
)

grid_svc.fit(x_train_scaled, y_train)
best_svc_base = grid_svc.best_estimator_

best_svc = CalibratedClassifierCV(
    estimator=best_svc_base,
    method="sigmoid",
    cv=5,
)
best_svc.fit(x_train_scaled, y_train)

svc_val_pred = best_svc.predict(x_val_scaled)
svc_val_prob = best_svc.predict_proba(x_val_scaled)[:, 1]

# ---- Random Forest ----
rf_base = RandomForestClassifier(
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

param_grid_rf = {
    "n_estimators": [200, 300, 500],
    "max_depth": [None, 10, 20, 30],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}

grid_rf = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid_rf,
    scoring="f1",
    cv=5,
    n_jobs=-1,
)

grid_rf.fit(x_train, y_train)  # For Random Forest, we don't scale the features because it's a tree-based model
best_rf = grid_rf.best_estimator_

rf_val_pred = best_rf.predict(x_val)
rf_val_prob = best_rf.predict_proba(x_val)[:, 1]

# =========================
# 5) Evaluation function + comparison of models
# =========================

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


results = pd.DataFrame(
    [
        evaluate_model("Logistic Regression", y_val, log_val_pred, log_val_prob),
        evaluate_model("SVC", y_val, svc_val_pred, svc_val_prob),
        evaluate_model("Random Forest", y_val, rf_val_pred, rf_val_prob),
    ]
).sort_values("f1_score", ascending=False)

print(results)

best_model_name = results.iloc[0]["model"]
print(f"Best model based on validation F1-score: {best_model_name}")

# =========================
# 6) Retrain the best model on the combined train+val set and evaluate on the test set
# =========================

if best_model_name == "Logistic Regression":
    final_scaler = StandardScaler()
    x_train_val_final = final_scaler.fit_transform(x_train_val)
    x_test_final = final_scaler.transform(x_test)
    best_model = best_log_reg

elif best_model_name == "SVC":
    final_scaler = StandardScaler()
    x_train_val_final = final_scaler.fit_transform(x_train_val)
    x_test_final = final_scaler.transform(x_test)
    best_model = best_svc

else:  # Random Forest
    x_train_val_final = x_train_val
    x_test_final = x_test
    best_model = best_rf

best_model.fit(x_train_val_final, y_train_val)
best_test_pred = best_model.predict(x_test_final)
best_test_prob = best_model.predict_proba(x_test_final)[:, 1]

print("\nTest-set classification report:")
print(
    classification_report(
        y_test,
        best_test_pred,
        target_names=["No order", "Order"],
        zero_division=0,
    )
)

print("Confusion matrix:")
print(confusion_matrix(y_test, best_test_pred))

print(f"Accuracy:          {accuracy_score(y_test, best_test_pred):.4f}")
print(f"Precision:         {precision_score(y_test, best_test_pred, zero_division=0):.4f}")
print(f"Recall:            {recall_score(y_test, best_test_pred, zero_division=0):.4f}")
print(f"F1-score:          {f1_score(y_test, best_test_pred, zero_division=0):.4f}")
print(f"ROC-AUC:           {roc_auc_score(y_test, best_test_prob):.4f}")
print(f"Average precision: {average_precision_score(y_test, best_test_prob):.4f}")

"""