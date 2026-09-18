"""Shared Streamlit configuration and visual constants."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "events_10000_sessions.csv"
API_URL = os.getenv("API_URL", "http://localhost:8000")

FEATURE_COLUMNS = ["num_clicks", "num_carts", "num_events", "num_unique_items"]
MODEL_FEATURE_COLUMNS = FEATURE_COLUMNS + [
    "session_duration_seconds",
    "hour",
    "weekday",
]
SCALED_MODELS = {"Logistic Regression", "SVC"}

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
ORDINAL_BLUE = ["#86b6ef", "#2a78d6", "#104281"]
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
FONT_FAMILY = "system-ui, -apple-system, 'Segoe UI', sans-serif"
MODEL_COLORS = {"Random Forest": BLUE, "Logistic Regression": ORANGE, "SVC": AQUA}

LOTTIE_WELCOME = "https://assets5.lottiefiles.com/packages/lf20_V9t630.json"
LOTTIE_SUCCESS = "https://raw.githubusercontent.com/ariyanshiputech/custom_quick_alert/main/assets/animations/success.json"

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEEKDAY_MAP = {day: index for index, day in enumerate(WEEKDAYS)}
