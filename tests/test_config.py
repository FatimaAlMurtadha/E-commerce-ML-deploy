"""Tests for shared project paths."""

from pathlib import Path

from backend import config


def test_project_root_is_repository_root():
    assert config.PROJECT_ROOT == Path(__file__).resolve().parents[1]
    assert (config.PROJECT_ROOT / "pyproject.toml").exists()


def test_model_and_metadata_paths_are_under_model_directory():
    assert config.MODEL_DIR == config.PROJECT_ROOT / "model"
    assert config.MODEL_PATH == config.MODEL_DIR / "ecommerce_pipeline.joblib"
    assert config.META_PATH == config.MODEL_DIR / "metadata.json"
    


# To run the tests, use one of the following command:
# python -m pytest tests 
# python -m pytest tests -q
# python -m pytest tests/test_config.py -q
# python -m pytest tests/test_api.py -q