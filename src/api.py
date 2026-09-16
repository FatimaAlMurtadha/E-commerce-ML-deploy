# Importeringar
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

# Datamodeller
class SessionInput(BaseModel):
    num_clicks: int
    num_carts: int
    num_events: int
    num_unique_items: int

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

# Modellhantering
MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
model = None

if MODEL_PATH.exists():
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        model = None

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
        df = pd.DataFrame([session.model_dump()])
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
    uvicorn.run("api:app", host="0.0.0.0", port=port)