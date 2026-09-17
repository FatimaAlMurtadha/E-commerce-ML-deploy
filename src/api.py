# Importeringar
from pathlib import Path
from typing import Optional, Union
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

try:
    from src.config import MODEL_PATH
except ImportError:
    try:
        from config import MODEL_PATH
    except ImportError:
        MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "ecommerce_pipeline.joblib"

if not MODEL_PATH.exists():
    local_fallback = Path(__file__).resolve().parent / "model.joblib"
    if local_fallback.exists():
        MODEL_PATH = local_fallback

# Datamodeller
class SessionInput(BaseModel):
    num_clicks: int
    num_carts: int
    num_events: int
    num_unique_items: int
    session_duration_seconds: Optional[float] = None
    hour: Optional[int] = None
    weekday: Optional[Union[int, str]] = None

    model_config = {"extra": "allow"}

class PredictionOutput(BaseModel):
    prediction: int
    order: bool
    probability: float

# Initiering och CORS
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WEEKDAY_MAP = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}

FEATURE_DEFAULTS = {
    "session_duration_seconds": 0.0,
    "hour": 12,
    "weekday": 0,
}

# Modellhantering
def load_model(path: Path = MODEL_PATH):
    if not path.exists():
        return None
    try:
        loaded = joblib.load(path)
        if isinstance(loaded, dict):
            m = loaded.get("model")
            s = loaded.get("scaler")
            if s is not None and m is not None:
                return Pipeline([("scaler", s), ("classifier", m)])
            return m
        return loaded
    except Exception:
        return None

model = load_model(MODEL_PATH)

# Endpoints
@app.get("/")
def read_root():
    return {"message": "E-commerce API", "status": "ok"}

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionOutput)
def predict_order(session: SessionInput):
    if model is not None:
        raw_data = session.model_dump(exclude_unset=True)
        expected_features = getattr(model, "feature_names_in_", None)

        if expected_features is not None:
            data = {}
            for col in expected_features:
                if col in raw_data and raw_data[col] is not None:
                    val = raw_data[col]
                    if col == "weekday" and isinstance(val, str):
                        val = WEEKDAY_MAP.get(val, 0)
                    data[col] = val
                else:
                    data[col] = FEATURE_DEFAULTS.get(col, 0)
            df = pd.DataFrame([data], columns=list(expected_features))
        else:
            df = pd.DataFrame([raw_data])

        pred = int(model.predict(df)[0])
        prob = float(model.predict_proba(df)[0][1])
    else:
        prob = min(0.95, max(0.05, session.num_carts * 0.3 + session.num_clicks * 0.02))
        pred = 1 if prob >= 0.5 else 0

    return PredictionOutput(
        prediction=pred,
        order=(pred == 1),
        probability=round(prob, 4)
    )

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)