# need edits


"""
Gemensamma sökvägar och namn för hela paketet.

Sökvägarna utgår från projektroten (mappen med pyproject.toml), inte från
den mapp man råkar stå i när man kör kommandot. Det gör att `uv run train`,
`uv run uvicorn ...` och `uv run pytest` hittar samma register oavsett
varifrån de startas.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# MLflow sparar två saker lokalt: metadata (körningar, mätvärden, registret)
# i en SQLite-databas och själva artefakterna (modellfiler, MLmodel, ...) i en mapp.
TRACKING_DB = PROJECT_ROOT / "mlflow.db"
TRACKING_URI = f"sqlite:///{TRACKING_DB}"
ARTIFACT_ROOT = PROJECT_ROOT / "mlruns"

EXPERIMENT_NAME = "iris"
MODEL_NAME = "iris-classifier"
ALIAS = "production"
MODEL_URI = f"models:/{MODEL_NAME}@{ALIAS}"