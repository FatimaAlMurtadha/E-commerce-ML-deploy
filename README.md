# E-commerce ML

Machine-learning project that predicts whether an e-commerce session will result in an order. It contains a backend prediction service, a Streamlit frontend, a reproducible training pipeline, Docker Compose configuration, and CI checks.

## Start the project

The easiest way to run the complete system is Docker Desktop with Docker Compose:

```bash
docker compose up --build
```

Open:

- API documentation: http://localhost:8000/docs
- API health: http://localhost:8000/health
- Streamlit frontend: http://localhost:8501

Stop the services with:

```bash
docker compose down
```

## Project structure

```text
backend/
    api.py                    FastAPI application and prediction endpoint
    config.py                 Model and metadata paths
    train.py                  Training command-line entrypoint
    e_commerce_ml/            Data processing, training, and artifacts
frontend/
    streamlit_app.py          Streamlit entrypoint
    ui.py                     Page orchestration
    ui_services.py            Backend API and dataset access
    ui_features.py            Session analytics feature preparation
    ui_views.py               Streamlit tab views
    ui_charts.py              Chart styling and gauge component
    ui_config.py              Frontend configuration and constants
    ui_model_legacy.py        Compatibility-only local model tests
data/                          Training data
model/                         Checked-in model and metadata
tests/                         Backend and frontend tests
Dockerfile                     Container image definition
docker-compose.yml             API and frontend services
```

The frontend does not train or select a prediction model. It collects the seven session inputs, calls the backend, and displays the returned prediction and explanation. The backend owns model loading, feature handling, inference, and model metadata.

## Data and model

The event data contains `clicks`, `carts`, and `orders`. The model uses these seven features:

```text
num_clicks
num_carts
num_events
num_unique_items
session_duration_seconds
hour
weekday
```

For sessions containing an order, only clicks and carts before the first order are used. The checked-in model is currently an SVC with scaling. Its metadata and recorded test metrics are stored in `model/metadata.json`.

The saved model artifact must be loaded with the same scikit-learn version used during training. Recreate or retrain the artifact when changing scikit-learn versions; otherwise model loading can produce compatibility warnings or invalid inference results.

## Backend API

Run the backend locally from the repository root:

```bash
uv run uvicorn backend.api:app --reload --port 8000
```

### `GET /health`

Returns service and model status:

```json
{ "status": "ok", "model_loaded": true, "model_name": "SVC" }
```

### `GET /model-info`

Returns the checked-in model name, feature list, and recorded test metrics. The frontend uses this endpoint for its model-performance view.

### `POST /predict`

The request body contains session information:

```json
{
  "num_clicks": 10,
  "num_carts": 2,
  "num_events": 12,
  "num_unique_items": 8,
  "session_duration_seconds": 320.5,
  "hour": 18,
  "weekday": "Friday"
}
```

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "num_clicks": 10,
    "num_carts": 2,
    "num_events": 12,
    "num_unique_items": 8,
    "session_duration_seconds": 320.5,
    "hour": 18,
    "weekday": "Friday"
  }'
```

The response includes the numeric prediction plus a risk level, reasons, a suggested company action, and a draft customer message. These are decision-support suggestions, not causal explanations or guarantees.

## Training

Run the reproducible training pipeline from the repository root:

```bash
uv run python -m backend.train
```

Training reads `data/events_10000_sessions.csv` and updates the artifacts under `model/`. The pipeline performs feature creation, chronological splitting, model comparison, final evaluation, and artifact saving.

## Tests

Run the complete test suite:

```bash
uv run python -m pytest -q
```

The test suite covers:

- FastAPI endpoints, request validation, response schemas, fallback predictions, CORS, and model loading.
- The model input contract: feature names, feature order, default values, weekday conversion, and prediction probabilities.
- The real checked-in model: artifact/metadata compatibility, inference on the real event dataset, and an end-to-end API prediction.
- Streamlit application startup, prediction interaction, feature preparation, and chart helpers.

Run the real production-model checks explicitly:

```bash
uv run python -m pytest tests/test_real_model.py -q
```

Compile the application packages:

```bash
uv run python -m compileall -q backend frontend
```

## CI and Docker

GitHub Actions installs dependencies, runs the tests, compiles both application packages, validates Docker Compose, and builds the image. Docker Compose runs:

- `backend.api:app` on port 8000
- `frontend/streamlit_app.py` on port 8501

The frontend receives `API_URL=http://api:8000` inside the Compose network.

## Deploying the services

The frontend and backend are separate processes. Deploy the backend as a web
service first, then deploy the Streamlit app and set its environment variable:

For Streamlit Community Cloud, add this under **App settings > Secrets**:

```toml
API_URL = "https://<your-backend-host>"
API_TIMEOUT_SECONDS = "5"
```

For other hosting providers, configure the same values as environment
variables:

```text
API_URL=https://<your-backend-host>
API_TIMEOUT_SECONDS=5
```

`BACKEND_URL` is also accepted as an environment variable. Do not set this to
`localhost` in a hosted Streamlit app: that points to the frontend container,
not to the API service.
The backend must listen on `0.0.0.0` and use the hosting provider's `PORT`
value, for example:

```bash
uvicorn backend.api:app --host 0.0.0.0 --port $PORT
```

Verify the backend before opening the frontend by visiting
`https://<your-backend-host>/health`. It should return `status: ok` and
`model_loaded: true`. The frontend uses a 0.5-second request timeout by
default; set `API_TIMEOUT_SECONDS=5` or higher if the provider has a slower
cold start.
