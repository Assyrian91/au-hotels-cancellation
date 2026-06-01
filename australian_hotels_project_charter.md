# Project Charter — Australian Hotel Booking Intelligence

## Problem Statement

Hotel booking cancellations cost the Australian accommodation industry an estimated AUD 1.2–2.5B annually in lost revenue, rebooking costs, and operational inefficiency. This project builds an end-to-end machine learning system that predicts the probability of a booking cancellation at the time of reservation, enabling revenue managers to take pre-emptive action (targeted retention offers, overbooking adjustments, dynamic deposit policies).

## Business Objective

Enable hotel revenue management teams to reduce cancellation-driven revenue loss by identifying high-risk bookings early enough to intervene.

**Target users:** Revenue managers and operations teams at Australian hotel chains (e.g. Accor Pacific, Mantra Group, Quest Apartment Hotels).

## Dataset

| Source | Description | Rows | License |
|--------|-------------|------|---------|
| Kaggle — Hotel Booking Demand (Jesse Mostipak) | 119,390 bookings across Resort and City hotels, 32 features | 119,390 | CC0 Public Domain |
| ABS Tourist Accommodation Australia | State-level occupancy rates and ADR benchmarks | Aggregated | Open Government |
| Inside Airbnb — Sydney / Melbourne / Brisbane | Listing-level pricing and availability (supplementary) | ~150,000 | CC0 |

**Primary modelling dataset:** Kaggle source, augmented with Australian-specific engineered features.

## Target Variable

`is_canceled` — binary (0 = not cancelled, 1 = cancelled)

**Class balance:** ~37% cancellations in raw data. Will apply class-weight balancing or SMOTE if needed.

## Success Metrics

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| AUC-ROC | ≥ 0.88 | Primary metric; reflects ranking quality across thresholds |
| F1-Score (cancelled class) | ≥ 0.75 | Balances precision and recall on the minority class |
| Precision @ 80% recall | ≥ 0.70 | Business constraint: catch most cancellations without over-alerting staff |
| Inference latency | ≤ 100ms p95 | API SLA for real-time booking systems |

## Scope

**In scope:**
- Binary cancellation prediction at time of booking
- REST API serving predictions in real time
- Automated weekly retraining pipeline
- Model monitoring for data drift and performance degradation
- Streamlit dashboard for stakeholder exploration

**Out of scope:**
- Room pricing optimisation (Phase 2 of project)
- Integration with live PMS (Property Management Systems)
- Multi-label prediction (cancellation reason)

## Australian Context Features (to be engineered)

- State-mapped synthetic hotel locations (NSW, VIC, QLD, WA, SA)
- School holiday flags per state (dates differ by state)
- Australian public holidays (ANZAC Day, Australia Day, Melbourne Cup for VIC)
- Southern Hemisphere seasonal labels (summer = Dec–Feb)
- Major event proximity flags (F1 Grand Prix Melbourne, Vivid Sydney, AFL Grand Final)
- AUD pricing normalisation

## Technical Stack

| Layer | Technology |
|-------|------------|
| Data processing | Python, pandas, scikit-learn Pipeline |
| Experimentation | MLflow, XGBoost / LightGBM |
| Explainability | SHAP |
| Serving | FastAPI + Docker |
| Cloud | AWS ap-southeast-2 (Sydney) |
| Orchestration | Apache Airflow |
| Monitoring | Evidently AI, Prometheus, Grafana |
| Portfolio | Streamlit, GitHub |

## Timeline (Estimated)

| Phase | Duration |
|-------|----------|
| Phase 1: Dataset & Charter | Week 1 |
| Phase 2: EDA | Week 1–2 |
| Phase 3: Feature Engineering | Week 2–3 |
| Phase 4: Modelling & MLflow | Week 3–4 |
| Phase 5: Deployment | Week 4–5 |
| Phase 6: Automation | Week 5–6 |
| Phase 7: Monitoring | Week 6–7 |

## Portfolio Value Statement

This project demonstrates full MLOps maturity: from raw data to a monitored, auto-retrained production API deployed on AWS Sydney infrastructure. The Australian-specific feature engineering and ABS data integration make it directly relevant to employers in the AU hospitality, travel, and data consultancy sectors.

---
*Author: [Your Name] | Started: 2026 | Stack: Python · XGBoost · FastAPI · Airflow · AWS ap-southeast-2*
