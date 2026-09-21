from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "ecommerce_pipeline.joblib"
META_PATH = MODEL_DIR / "metadata.json"
