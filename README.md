# E-commerce ML

Machine-learning project that predicts whether an e-commerce session will result in an
order. It contains a training pipeline, a FastAPI prediction service, a planned Streamlit
frontend, a Docker Compose setup, and GitHub Actions CI.

## Project status

The API and trained model are available. `src/ui.py` is currently a placeholder for the
Streamlit frontend, but it is already included in the Compose setup so the UI can be added
without changing the deployment structure.

## Start the project

You do not need to install Python, `uv`, or the project dependencies. You only need:

- Git
- Docker Desktop with Docker Compose

1. Clone the repository:

	```bash
	git clone https://github.com/arbazshah52/E-commerce-ML.git
	cd E-commerce-ML
	```

2. Start the application:

	```bash
	docker compose up --build
	```

	The first start downloads the Python base image and builds the application image. This
	can take a few minutes. Later starts are faster because Docker reuses the image layers.

3. Open the services in a browser:

	- API documentation: <http://localhost:8000/docs>
	- API health: <http://localhost:8000/health>
	- Streamlit frontend: <http://localhost:8501>

4. Stop the application by pressing `Ctrl+C` in the terminal, or from another terminal run:

	```bash
	docker compose down
	```

The API uses the trained model included in the repository. No training command is required
to start the application. The current Streamlit page is only a placeholder while the user
interface is being developed.

## Architecture

```text
data/events_10000_sessions.csv
				|
				v
src/train.py  -->  model/ecommerce_pipeline.joblib
													 |
													 v
										 src/api.py
													 |
													 v
							POST /predict and GET /health
													 |
													 v
											src/ui.py
```

The API loads the checked-in model when it starts. If the model cannot be loaded, the API
uses a simple fallback probability calculation so that the endpoint remains available.

## Repository structure

```text
data/                         Training dataset
model/                        Trained model and metadata
notebooks/                    Exploratory and preparation notebooks
src/api.py                    FastAPI application
src/config.py                 Shared project and model paths
src/train.py                  Training command-line entry point
src/e_commerce_ml/            Training package
	data_processing.py          Feature engineering and data splitting
	model_training.py           Model search, selection, and evaluation
	artifacts.py                Model and metadata persistence
	training.py                 Training workflow orchestration
src/ui.py                     Streamlit entry point
tests/                        API and configuration tests
Dockerfile                    Application image definition
docker-compose.yml            API and UI services
.github/workflows/ci.yml      Automated tests and Docker checks
pyproject.toml                Project dependencies and metadata
uv.lock                       Locked dependency versions
```

## Data and feature engineering

The source file `data/events_10000_sessions.csv` contains 628,747 events with these
columns:

```text
aid, ts, type, session, datetime, date, hour, weekday
```

The available event types are `clicks`, `carts`, and `orders`. `src/train.py` groups events
by session and creates one training row per usable session. The target is:

```text
0 = No order
1 = Order
```

The model uses these seven features:

```text
num_clicks
num_carts
num_events
num_unique_items
session_duration_seconds
hour
weekday
```

For sessions containing an order, only clicks and carts before the first order are used.
The data is split chronologically into 60% training, 20% validation, and 20% test data.
Scaling is fitted only on training data. Model selection and hyperparameter tuning do not
use the test set.

## Model artifacts

The current model is stored in `model/ecommerce_pipeline.joblib`. It contains the selected
SVC model and its scaler. `model/metadata.json` records the model name, features, test
metrics, and the leakage-prevention decisions.

Current recorded test metrics:

| Metric            |  Value |
| ----------------- | -----: |
| Accuracy          | 0.8233 |
| Precision         | 0.0950 |
| Recall            | 0.4156 |
| F1 score          | 0.1546 |
| ROC AUC           | 0.7556 |
| Average precision | 0.0882 |

These metrics are stored metadata from the checked-in model and are not recomputed when
the API starts.

## API

### Start locally

```bash
uv run uvicorn src.api:app --reload --port 8000
```

API documentation is available at <http://localhost:8000/docs>. The root endpoint is
<http://localhost:8000/>.

### Endpoints

`GET /` returns the service status:

```json
{ "message": "E-commerce API", "status": "ok" }
```

`GET /health` reports whether the model loaded:

```json
{ "status": "ok", "model_loaded": true }
```

`POST /predict` accepts the required session fields and optional time fields:

```json
{
  "num_clicks": 10,
  "num_carts": 2,
  "num_events": 14,
  "num_unique_items": 8,
  "session_duration_seconds": 320.5,
  "hour": 18,
  "weekday": "Friday"
}
```

The response has this shape:

```json
{
  "prediction": 1,
  "order": true,
  "probability": 0.8766
}
```

The required fields are `num_clicks`, `num_carts`, `num_events`, and
`num_unique_items`. Missing optional fields use defaults. `weekday` accepts either an
integer or a weekday name such as `Friday`.

## Dependencies and local setup

This project uses `pyproject.toml` for dependency declarations and `uv.lock` for
reproducible versions. The standard library modules `pathlib`, `typing`, `json`, and `os`
do not need to be listed as dependencies.

Install `uv`, then run:

```bash
uv sync
```

The development dependency group includes `pytest` and `httpx` for testing. The runtime
dependencies include FastAPI, Uvicorn, pandas, NumPy, scikit-learn, joblib, Streamlit,
MLflow, and requests.

## Train a new model

Training reads the default dataset from `data/events_10000_sessions.csv` and overwrites
the model artifacts in `model/`:

```bash
uv run python src/train.py
```

The training process performs feature creation, chronological splitting, scaling, grid
search for Logistic Regression, SVC, and Random Forest models, validation-based model
selection, final test evaluation, and artifact saving.

### Object-oriented design

The implementation in `src/e_commerce_ml/` uses classes so each object has one main
responsibility. The modules are separated by technical concern:

| Class              | Responsibility                                | OOP idea      |
| ------------------ | --------------------------------------------- | ------------- |
| `DatasetProcessor` | Loads, prepares, and splits session data      | Encapsulation |
| `ModelTrainer`     | Scales, trains, evaluates, and selects models | Encapsulation |
| `TrainingPipeline` | Coordinates processing, training, and saving  | Composition   |

`TrainingPipeline` owns a `DatasetProcessor` and `ModelTrainer` instead of implementing
every detail itself. This is composition: complex behavior is built from smaller
collaborating objects. Model persistence is kept as a focused `save_model()` function,
because a separate repository class would add indirection without enough benefit for this
project. Data processing is in `data_processing.py`, model work is in `model_training.py`,
and persistence is in `artifacts.py`. `src/train.py` is only the command-line entry point.

## Tests

Run all tests:

```bash
uv run python -m pytest -q
```

The tests cover project paths and the API root, health, prediction, fallback, loaded-model,
and validation behavior.

## Docker Compose

Docker Compose runs the API and Streamlit services together:

```bash
docker compose up --build
```

Open:

- API documentation: <http://localhost:8000/docs>
- Streamlit: <http://localhost:8501>

The `ui` service receives `API_URL=http://api:8000`, which is the internal Compose address.
The UI waits for the API health check before starting. Stop the services with:

```bash
docker compose down
```

The Docker image uses Python 3.12, copies the source code, model, and data into `/app`, and
installs the project from `pyproject.toml`. `.dockerignore` excludes local environments,
notebooks, caches, and Git metadata from the build context.

## GitHub Actions

The workflow at `.github/workflows/ci.yml` runs on every branch push and pull request. It:

1. Checks out the repository.
2. Installs Python 3.12 and the pinned `uv` version.
3. Installs dependencies with `uv sync --frozen`.
4. Runs the test suite.
5. Validates `docker-compose.yml`.
6. Builds the Docker image tagged `e-commerce-ml:ci`.

## Notebooks

The notebooks in `notebooks/` contain exploratory and preparation work. They are useful
for understanding the dataset, but the reproducible application training command is
`uv run python src/train.py`.

## Troubleshooting

If Docker reports that it cannot connect to the Docker API, start Docker Desktop and retry:

```bash
docker compose up --build
```

If the API reports `model_loaded: false`, verify that
`model/ecommerce_pipeline.joblib` exists and that the model was created with compatible
versions of Python, scikit-learn, NumPy, and joblib.
