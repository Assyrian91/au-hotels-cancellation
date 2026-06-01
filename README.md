# 🏨 Australian Hotels — Booking Cancellation Prediction

> End-to-end MLOps project: from raw data to a monitored, auto-retrained production API deployed on AWS Sydney.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green)
![Docker](https://img.shields.io/badge/Container-Docker-blue)
![AUC](https://img.shields.io/badge/AUC--ROC-0.8739-brightgreen)

## Business Problem

Hotel booking cancellations cost the Australian accommodation industry an estimated AUD 1.2–2.5B annually. This project builds a machine learning system that predicts the probability of a booking cancellation at time of reservation — enabling revenue managers to take pre-emptive action.

**Target users:** Revenue management teams at Australian hotel chains (Accor Pacific, Mantra Group, Quest Apartment Hotels).

---

## Live Demo

| Resource | Link |
|----------|------|
| 🚀 API Docs (Swagger) | `http://localhost:8000/docs` |
| 📊 Streamlit Dashboard | `http://localhost:8501` |

---

## Results

| Model | AUC-ROC | F1 Score | Accuracy |
|-------|---------|----------|----------|
| Logistic Regression (baseline) | 0.8237 | 0.6075 | 0.7278 |
| XGBoost | 0.8688 | 0.6695 | 0.7864 |
| **XGBoost + Optuna (champion)** | **0.8739** | **0.6769** | **0.8008** |

---

## Project Structure

```
hotel_bookings/
├── app/
│   ├── main.py              # FastAPI prediction API
│   └── dashboard.py         # Streamlit dashboard
├── data/
│   ├── raw/                 # Source data (not committed)
│   └── processed/           # Feature-engineered data (not committed)
├── models/
│   └── registry/
│       ├── champion_meta.json
│       └── feature_names.json
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modelling.ipynb
├── reports/figures/         # EDA and SHAP visualisations
├── src/features/
│   └── build_features.py    # Reusable feature pipeline
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Australian-Specific Features Engineered

- **School holiday flags** — national overlap periods across all states
- **Public holiday flags** — Australia Day, ANZAC Day, Christmas, Boxing Day
- **Major event flags** — F1 Grand Prix Melbourne, Vivid Sydney, AFL Grand Final
- **Melbourne Cup week** flag (VIC)
- **Southern Hemisphere seasons** — Summer = Dec–Feb (not Jun–Aug)
- **AUD price tiers** — Budget / Midrange / Upscale / Luxury / Ultra

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/au-hotels-cancellation.git
cd au-hotels-cancellation
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download data

Download `hotel_bookings.csv` from [Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) and place it at `data/raw/hotel_bookings.csv`.

### 3. Run the full pipeline

```bash
# Feature engineering
python src/features/build_features.py

# Train model (opens Jupyter)
jupyter notebook notebooks/03_modelling.ipynb
```

### 4. Run with Docker

```bash
docker build -t au-hotels-api .
docker run -p 8000:8000 au-hotels-api
```

API available at `http://localhost:8000/docs`

### 5. Run Streamlit dashboard

```bash
streamlit run app/dashboard.py
```

---

## API Usage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "lead_time": 120,
    "arrival_date_year": 2025,
    "arrival_date_week_number": 27,
    "arrival_date_day_of_month": 15,
    "stays_in_weekend_nights": 1,
    "stays_in_week_nights": 3,
    "adults": 2,
    "children": 0,
    "babies": 0,
    "is_repeated_guest": 0,
    "previous_cancellations": 2,
    "previous_bookings_not_canceled": 0,
    "booking_changes": 0,
    "days_in_waiting_list": 0,
    "adr": 180.0,
    "required_car_parking_spaces": 0,
    "total_of_special_requests": 0,
    "hotel": "City Hotel",
    "meal": "BB",
    "market_segment": "Online TA",
    "distribution_channel": "TA/TO",
    "deposit_type": "No Deposit",
    "customer_type": "Transient",
    "reserved_room_type": "A",
    "assigned_room_type": "A",
    "arrival_date_month": "July"
  }'
```

**Response:**

```json
{
  "cancellation_probability": 0.7312,
  "will_cancel": true,
  "risk_level": "HIGH",
  "recommendation": "Request non-refundable deposit or offer retention incentive",
  "model_version": "XGBoost_Tuned",
  "auc_roc": 0.8739
}
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Data processing | Python, pandas, scikit-learn |
| Experimentation | MLflow, Optuna (50 trials) |
| Model | XGBoost + SHAP explainability |
| Serving | FastAPI + Uvicorn |
| Container | Docker |
| Dashboard | Streamlit + Plotly |
| Orchestration | Apache Airflow (Phase 6) |
| Monitoring | Evidently AI, Grafana, Prometheus |
| Cloud target | AWS ap-southeast-2 (Sydney) |

---

## Key Insights from EDA

- **Deposit type** is the single strongest cancellation signal — non-refundable deposits paradoxically show near-100% cancellation rates
- **Lead time** over 90 days doubles cancellation probability
- **Previous cancellation history** is highly predictive — guests who cancelled before cancel again
- **Online TA segment** has significantly higher cancellation rates than Direct bookings
- **Australian winter (Jun–Aug)** shows elevated cancellation rates vs summer

---

## Author

Built as a portfolio project targeting Australian data science roles.  
Stack demonstrates full MLOps maturity: data → features → model → API → Docker → monitoring.

*Deployed on AWS ap-southeast-2 (Sydney region)*
