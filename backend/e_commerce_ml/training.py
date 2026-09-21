from pathlib import Path

from backend.config import META_PATH, MODEL_PATH

from .artifacts import save_model
from .data_processing import (
    FEATURE_COLUMNS,
    WEEKDAY_MAP,
    DatasetProcessor,
    DatasetSplit,
)
from .model_training import ModelTrainer


class TrainingPipeline:
    """Coordinate data processing, model training, and artifact persistence."""

    def __init__(self, dataset=None, trainer=None):
        self.dataset = dataset or DatasetProcessor()
        self.trainer = trainer or ModelTrainer()

    def run(self, data_path: str | Path | None = None):
        if data_path is None:
            data_path = Path(__file__).resolve().parents[2] / "data" / "events_10000_sessions.csv"

        print("Starting training pipeline")
        print("-------------------------")
        events = self.dataset.load_events(data_path)
        session_features = self.dataset.create_session_features(events)
        x, y, timestamps = self.dataset.prepare_features(session_features)
        split = self.dataset.split_data(x, y, timestamps)
        models, results, search_results = self.trainer.train_candidates(split)
        model, scaler, model_name, test_metrics = self.trainer.select_and_evaluate(
            models, results, split
        )
        save_model(
            model,
            scaler,
            model_name,
            test_metrics,
            MODEL_PATH,
            META_PATH,
            validation_results=results.to_dict(orient="records"),
            search_results=search_results,
        )
        print("-------------------------")
        print("Training completed!")
        print("Best model:", model_name)
        return model, scaler
