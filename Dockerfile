FROM python:3.11-slim

WORKDIR /hotel_app

# Install system dependencies needed by lightgbm and shap
RUN apt-get update && apt-get install -y \
    libgomp1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY app/         ./app/
COPY models/      ./models/
COPY src/         ./src/
COPY data/processed/feature_names.json ./data/processed/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]