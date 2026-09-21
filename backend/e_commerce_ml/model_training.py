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

from .data_processing import DatasetSplit


class ModelTrainer:
    """Train candidate estimators, select the best one, and evaluate it."""

    @staticmethod
    def evaluate(y_true, predictions, probabilities):
        return {
            "accuracy": accuracy_score(y_true, predictions),
            "precision": precision_score(y_true, predictions, zero_division=0),
            "recall": recall_score(y_true, predictions, zero_division=0),
            "f1_score": f1_score(y_true, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_true, probabilities),
            "average_precision": average_precision_score(y_true, probabilities),
        }

    @staticmethod
    def scale_features(split: DatasetSplit):
        print("Scaling data...")
        scaler = StandardScaler()
        return (
            scaler.fit_transform(split.x_train),
            scaler.transform(split.x_val),
            scaler,
        )

    def train_candidates(self, split: DatasetSplit):
        print("Training models...")
        train_scaled, validation_scaled, _ = self.scale_features(split)
        candidates = {
            "logreg": (
                "Logistic Regression",
                LogisticRegression(class_weight="balanced", random_state=42),
                {
                    "C": [0.01, 0.1, 1, 10, 100],
                    "penalty": ["l2"],
                    "solver": ["lbfgs", "liblinear"],
                    "max_iter": [500, 1000, 2000],
                },
                train_scaled,
                validation_scaled,
            ),
            "svc": (
                "SVC",
                SVC(class_weight="balanced", probability=True, random_state=42),
                {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "rbf"],
                    "gamma": ["scale", "auto"],
                },
                train_scaled,
                validation_scaled,
            ),
            "rf": (
                "Random Forest",
                RandomForestClassifier(
                    class_weight="balanced", random_state=42, n_jobs=-1
                ),
                {
                    "n_estimators": [200, 300, 500],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2"],
                },
                split.x_train,
                split.x_val,
            ),
        }

        models = {}
        validation_results = []
        search_results = []
        for key, (name, estimator, parameters, train_data, validation_data) in candidates.items():
            print(f"Training {name}...")
            search = GridSearchCV(estimator, parameters, scoring="f1", cv=5, n_jobs=-1)
            search.fit(train_data, split.y_train)
            model = search.best_estimator_
            predictions = model.predict(validation_data)
            probabilities = model.predict_proba(validation_data)[:, 1]
            print(f"Best {name} parameters:", search.best_params_)
            models[key] = model
            search_results.append(
                {
                    "model_key": key,
                    "model": name,
                    "best_params": search.best_params_,
                    "best_cv_score": search.best_score_,
                    "trials": [
                        {
                            "params": trial_params,
                            "mean_test_score": mean_score,
                            "std_test_score": std_score,
                            "rank_test_score": rank,
                        }
                        for trial_params, mean_score, std_score, rank in zip(
                            search.cv_results_["params"],
                            search.cv_results_["mean_test_score"],
                            search.cv_results_["std_test_score"],
                            search.cv_results_["rank_test_score"],
                        )
                    ],
                }
            )
            validation_results.append(
                {
                    "model_key": key,
                    "model": name,
                    "best_params": search.best_params_,
                    "best_cv_score": search.best_score_,
                    **self.evaluate(split.y_val, predictions, probabilities),
                }
            )

        results = pd.DataFrame(validation_results).sort_values(
            "f1_score", ascending=False
        )
        print("\nValidation results:")
        print(results.to_string(index=False))
        return models, results, search_results

    def select_and_evaluate(self, models, results, split: DatasetSplit):
        print("Selecting best model...")
        model_name = results.iloc[0]["model"]
        model_key = {
            "Logistic Regression": "logreg",
            "SVC": "svc",
            "Random Forest": "rf",
        }[model_name]
        print("Best model:", model_name)

        if model_key == "rf":
            scaler = None
            train_data = split.x_train_val
            test_data = split.x_test
        else:
            scaler = StandardScaler()
            train_data = scaler.fit_transform(split.x_train_val)
            test_data = scaler.transform(split.x_test)

        model = models[model_key]
        model.fit(train_data, split.y_train_val)
        predictions = model.predict(test_data)
        probabilities = model.predict_proba(test_data)[:, 1]
        test_metrics = self.evaluate(split.y_test, predictions, probabilities)

        print("\nClassification report:")
        print(
            classification_report(
                split.y_test,
                predictions,
                target_names=["No order", "Order"],
                zero_division=0,
            )
        )
        print("Confusion matrix:")
        print(confusion_matrix(split.y_test, predictions))
        print("\nTest metrics:")
        for metric, value in test_metrics.items():
            print(metric, ":", round(value, 4))
        return model, scaler, model_name, test_metrics
