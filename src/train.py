# this file has become the entry point for training the model, and is used in the CI/CD pipeline to compile the training modules
from pathlib import Path

try:
    from src.e_commerce_ml.training import TrainingPipeline
except ImportError:
    from e_commerce_ml.training import TrainingPipeline


def train_pipeline(data_path: str | Path | None = None):
    """Run the training pipeline and return the trained model and scaler."""
    return TrainingPipeline().run(data_path)


if __name__ == "__main__":
    train_pipeline()