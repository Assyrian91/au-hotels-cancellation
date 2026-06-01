import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import json

MONTH_MAP = {"January":1,"February":2,"March":3,"April":4,"May":5,"June":6,
             "July":7,"August":8,"September":9,"October":10,"November":11,"December":12}

AU_PUBLIC_HOLIDAYS = {(1,1),(1,26),(4,25),(12,25),(12,26)}

AU_SCHOOL_HOLIDAY_PERIODS = [
    (1,1,1,31),(4,1,4,21),(7,1,7,21),(9,20,10,7),(12,10,12,31)
]

AU_MAJOR_EVENTS = [(3,10,3,17),(5,24,6,15),(9,20,9,30)]

def is_public_holiday(m, d):
    return int((m, d) in AU_PUBLIC_HOLIDAYS)

def is_school_holiday(m, d):
    for ms,ds,me,de in AU_SCHOOL_HOLIDAY_PERIODS:
        if pd.Timestamp(2000,ms,ds) <= pd.Timestamp(2000,m,d) <= pd.Timestamp(2000,me,de):
            return 1
    return 0

def is_major_event(m, d):
    for ms,ds,me,de in AU_MAJOR_EVENTS:
        if pd.Timestamp(2000,ms,ds) <= pd.Timestamp(2000,m,d) <= pd.Timestamp(2000,me,de):
            return 1
    return 0

def au_season(month):
    if month in [12,1,2]:   return "Summer"
    elif month in [3,4,5]:  return "Autumn"
    elif month in [6,7,8]:  return "Winter"
    else:                   return "Spring"

def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # Drop leakage & high-null
    drop = ["reservation_status","reservation_status_date","company"]
    df.drop(columns=[c for c in drop if c in df.columns], inplace=True)

    # Nulls
    df["agent"].fillna(0, inplace=True)
    df["children"].fillna(0, inplace=True)
    df["country"].fillna("Unknown", inplace=True)
    df["agent"] = df["agent"].astype(int)
    df["children"] = df["children"].astype(int)

    # Filter
    df = df[df["adr"] >= 0]
    df = df[(df["adults"] + df["children"] + df["babies"]) > 0]
    df.drop_duplicates(inplace=True)

    # Date
    df["month_num"] = df["arrival_date_month"].map(MONTH_MAP)
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df = df[df["total_nights"] > 0]

    df["arrival_date"] = pd.to_datetime(
        df["arrival_date_year"].astype(str) + "-" +
        df["month_num"].astype(str) + "-" +
        df["arrival_date_day_of_month"].astype(str), errors="coerce")
    df["day_of_week"]        = df["arrival_date"].dt.dayofweek
    df["is_weekend_arrival"] = df["day_of_week"].isin([4,5,6]).astype(int)
    df["quarter"]            = df["arrival_date"].dt.quarter

    # AU flags
    df["is_public_holiday"] = df.apply(lambda r: is_public_holiday(r["month_num"], r["arrival_date_day_of_month"]), axis=1)
    df["is_school_holiday"]  = df.apply(lambda r: is_school_holiday(r["month_num"], r["arrival_date_day_of_month"]), axis=1)
    df["is_major_event"]     = df.apply(lambda r: is_major_event(r["month_num"], r["arrival_date_day_of_month"]), axis=1)
    df["au_season"]          = df["month_num"].apply(au_season)

    # Behaviour
    df["total_guests"]      = df["adults"] + df["children"] + df["babies"]
    df["total_stay_value"]  = df["adr"] * df["total_nights"]
    df["has_special_request"]= (df["total_of_special_requests"] > 0).astype(int)
    df["needs_parking"]      = (df["required_car_parking_spaces"] > 0).astype(int)
    df["room_was_changed"]   = (df["reserved_room_type"] != df["assigned_room_type"]).astype(int)
    df["has_cancel_history"] = (df["previous_cancellations"] > 0).astype(int)

    df["lead_time_bucket"] = pd.cut(df["lead_time"],
        bins=[0,7,30,90,180,365,9999],
        labels=["0-7d","8-30d","31-90d","91-180d","181-365d","365d+"])
    df["adr_tier"] = pd.cut(df["adr"],
        bins=[0,60,120,200,300,9999],
        labels=["Budget","Midrange","Upscale","Luxury","Ultra"])

    # Encode
    ohe_cols = ["hotel","meal","market_segment","distribution_channel",
                "deposit_type","customer_type","au_season"]
    label_cols = ["reserved_room_type","assigned_room_type","lead_time_bucket","adr_tier"]
    drop_cols  = ["arrival_date","arrival_date_month","country","agent","month_num"]

    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    df = pd.get_dummies(df, columns=[c for c in ohe_cols if c in df.columns], drop_first=True)
    le = LabelEncoder()
    for col in label_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    return df


if __name__ == "__main__":
    raw = pd.read_csv("././data/raw/hotel_bookings.csv")
    df_out = build_features(raw)
    X = df_out.drop(columns=["is_canceled"])
    y = df_out["is_canceled"]
    X.to_csv("././data/processed/X.csv", index=False)
    y.to_csv("././data/processed/y.csv", index=False)
    with open("././data/processed/feature_names.json","w") as f:
        json.dump(list(X.columns), f, indent=2)
    print(f"Done. X shape: {X.shape}")
