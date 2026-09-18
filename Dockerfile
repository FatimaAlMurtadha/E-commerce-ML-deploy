FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY backend ./backend
COPY frontend ./frontend
COPY model ./model
COPY data ./data

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

EXPOSE 8000 8501

CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
