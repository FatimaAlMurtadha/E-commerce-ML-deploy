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

