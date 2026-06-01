from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import json
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
import uvicorn

# ── Load model & metadata ─────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
MODEL_PATH    = BASE / "models/registry/champion_model.ubj"
FEATURES_PATH = BASE / "models/registry/feature_names.json"
META_PATH     = BASE / "models/registry/champion_meta.json"

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

with open(FEATURES_PATH) as f:
    feature_names = json.load(f)

with open(META_PATH) as f:
    meta = json.load(f)

# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Australian Hotels — Cancellation Prediction API",
    description="Predicts booking cancellation probability for Australian hotels.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Request schema ────────────────────────────────────────────────────────
class BookingRequest(BaseModel):
    lead_time:                      int   = Field(..., example=45)
    arrival_date_year:              int   = Field(..., example=2025)
    arrival_date_week_number:       int   = Field(..., example=27)
    arrival_date_day_of_month:      int   = Field(..., example=15)
    stays_in_weekend_nights:        int   = Field(..., example=1)
    stays_in_week_nights:           int   = Field(..., example=3)
    adults:                         int   = Field(..., example=2)
    children:                       int   = Field(..., example=0)
    babies:                         int   = Field(..., example=0)
    is_repeated_guest:              int   = Field(..., example=0)
    previous_cancellations:         int   = Field(..., example=0)
    previous_bookings_not_canceled: int   = Field(..., example=0)
    booking_changes:                int   = Field(..., example=0)
    days_in_waiting_list:           int   = Field(..., example=0)
    adr:                            float = Field(..., example=120.0)
    required_car_parking_spaces:    int   = Field(..., example=0)
    total_of_special_requests:      int   = Field(..., example=1)
    hotel:                          str   = Field(..., example="City Hotel")
    meal:                           str   = Field(..., example="BB")
    market_segment:                 str   = Field(..., example="Online TA")
    distribution_channel:           str   = Field(..., example="TA/TO")
    deposit_type:                   str   = Field(..., example="No Deposit")
    customer_type:                  str   = Field(..., example="Transient")
    reserved_room_type:             str   = Field(..., example="A")
    assigned_room_type:             str   = Field(..., example="A")
    arrival_date_month:             str   = Field(..., example="July")


# ── Constants ─────────────────────────────────────────────────────────────
MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

AU_PUBLIC_HOLIDAYS = {(1, 1), (1, 26), (4, 25), (12, 25), (12, 26)}

AU_SCHOOL_HOLIDAYS = [
    (1, 1,  1, 31),
    (4, 1,  4, 21),
    (7, 1,  7, 21),
    (9, 20, 10, 7),
    (12, 10, 12, 31)
]

AU_MAJOR_EVENTS = [
    (3, 10, 3, 17),
    (5, 24, 6, 15),
    (9, 20, 9, 30)
]


# ── Helpers ───────────────────────────────────────────────────────────────
def _in_period(m, d, periods):
    for ms, ds, me, de in periods:
        if pd.Timestamp(2000, ms, ds) <= pd.Timestamp(2000, m, d) <= pd.Timestamp(2000, me, de):
            return 1
    return 0


def _bucket(value, bins, labels):
    for i in range(len(bins) - 1):
        if bins[i] < value <= bins[i + 1]:
            return labels[i]
    return labels[-1]


def au_season(month):
    if month in [12, 1, 2]:
        return "Summer"
    elif month in [3, 4, 5]:
        return "Autumn"
    elif month in [6, 7, 8]:
        return "Winter"
    else:
        return "Spring"


# ── Feature engineering ───────────────────────────────────────────────────
def build_input(req: BookingRequest) -> pd.DataFrame:
    m = MONTH_MAP[req.arrival_date_month]
    d = req.arrival_date_day_of_month
    total_nights = req.stays_in_weekend_nights + req.stays_in_week_nights
    arrival = pd.Timestamp(req.arrival_date_year, m, d)

    row = {
        "lead_time":                       req.lead_time,
        "arrival_date_year":               req.arrival_date_year,
        "arrival_date_week_number":        req.arrival_date_week_number,
        "arrival_date_day_of_month":       d,
        "stays_in_weekend_nights":         req.stays_in_weekend_nights,
        "stays_in_week_nights":            req.stays_in_week_nights,
        "adults":                          req.adults,
        "children":                        req.children,
        "babies":                          req.babies,
        "is_repeated_guest":               req.is_repeated_guest,
        "previous_cancellations":          req.previous_cancellations,
        "previous_bookings_not_canceled":  req.previous_bookings_not_canceled,
        "booking_changes":                 req.booking_changes,
        "days_in_waiting_list":            req.days_in_waiting_list,
        "adr":                             req.adr,
        "required_car_parking_spaces":     req.required_car_parking_spaces,
        "total_of_special_requests":       req.total_of_special_requests,
        "total_nights":                    total_nights,
        "total_guests":                    req.adults + req.children + req.babies,
        "total_stay_value":                req.adr * total_nights,
        "has_special_request":             int(req.total_of_special_requests > 0),
        "needs_parking":                   int(req.required_car_parking_spaces > 0),
        "room_was_changed":                int(req.reserved_room_type != req.assigned_room_type),
        "has_cancel_history":              int(req.previous_cancellations > 0),
        "day_of_week":                     arrival.dayofweek,
        "is_weekend_arrival":              int(arrival.dayofweek in [4, 5, 6]),
        "quarter":                         arrival.quarter,
        "is_public_holiday":               int((m, d) in AU_PUBLIC_HOLIDAYS),
        "is_school_holiday":               _in_period(m, d, AU_SCHOOL_HOLIDAYS),
        "is_major_event":                  _in_period(m, d, AU_MAJOR_EVENTS),
        "is_melbourne_cup":                int(m == 11 and 1 <= d <= 7),
        "hotel":                           req.hotel,
        "meal":                            req.meal,
        "market_segment":                  req.market_segment,
        "distribution_channel":            req.distribution_channel,
        "deposit_type":                    req.deposit_type,
        "customer_type":                   req.customer_type,
        "au_season":                       au_season(m),
        "reserved_room_type":              req.reserved_room_type,
        "assigned_room_type":              req.assigned_room_type,
        "lead_time_bucket":                _bucket(
                                               req.lead_time,
                                               [0, 7, 30, 90, 180, 365, 9999],
                                               ["0-7d", "8-30d", "31-90d", "91-180d", "181-365d", "365d+"]
                                           ),
        "adr_tier":                        _bucket(
                                               req.adr,
                                               [0, 60, 120, 200, 300, 9999],
                                               ["Budget", "Midrange", "Upscale", "Luxury", "Ultra"]
                                           ),
    }

    df = pd.DataFrame([row])

    # One-hot encode
    ohe_cols = ["hotel", "meal", "market_segment", "distribution_channel",
                "deposit_type", "customer_type", "au_season"]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    # Label encode ordinal
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in ["reserved_room_type", "assigned_room_type", "lead_time_bucket", "adr_tier"]:
        df[col] = le.fit_transform(df[col].astype(str))

    # Align to training feature set
    df = df.reindex(columns=feature_names, fill_value=0)
    return df


# ── Routes ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Australian Hotels Cancellation API",
        "model": meta["model_type"],
        "auc_roc": meta["auc_roc"]
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "features": meta["feature_count"]
    }


@app.get("/model-info")
def model_info():
    return meta


@app.post("/predict")
def predict(booking: BookingRequest):
    try:
        X = build_input(booking)
        prob = float(model.predict_proba(X)[0][1])
        pred = int(prob >= meta["threshold"])
        risk = "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.4 else "LOW"

        return {
            "cancellation_probability": round(prob, 4),
            "will_cancel": bool(pred),
            "risk_level": risk,
            "recommendation": {
                "HIGH":   "Request non-refundable deposit or offer retention incentive",
                "MEDIUM": "Monitor — consider flexible rebooking offer",
                "LOW":    "Standard booking — no action required"
            }[risk],
            "model_version": meta["model_type"],
            "auc_roc": meta["auc_roc"]
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
