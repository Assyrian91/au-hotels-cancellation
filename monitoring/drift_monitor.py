"""
Australian Hotels — Drift Monitoring with Evidently AI
Run weekly after retraining to detect data and prediction drift
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, ClassificationPreset
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    ColumnDriftMetric,
)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE         = Path(__file__).parent.parent
PROC_PATH    = BASE / "data/processed"
REPORTS_PATH = BASE / "reports/monitoring"
REPORTS_PATH.mkdir(parents=True, exist_ok=True)

# ── Key features to monitor closely ───────────────────────────────────────
KEY_FEATURES = [
    "lead_time",
    "adr",
    "total_nights",
    "total_stay_value",
    "is_school_holiday",
    "is_public_holiday",
    "is_major_event",
    "has_cancel_history",
    "total_of_special_requests",
]


def load_data():
    X = pd.read_csv(PROC_PATH / "X.csv")
    y = pd.read_csv(PROC_PATH / "y.csv").squeeze()
    df = X.copy()
    df["is_canceled"] = y
    return df


def split_reference_current(df, reference_frac=0.7):
    """
    Simulate reference vs current split.
    In production: reference = last month's data, current = this month's data.
    """
    split_idx = int(len(df) * reference_frac)
    reference = df.iloc[:split_idx].reset_index(drop=True)
    current   = df.iloc[split_idx:].reset_index(drop=True)
    return reference, current


def run_data_drift_report(reference, current):
    print("Running data drift report...")

    # Use only feature columns (no target) for drift report
    feature_cols = [c for c in reference.columns if c != "is_canceled"]
    ref_features = reference[feature_cols]
    cur_features = current[feature_cols]

    report = Report(metrics=[
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
        *[ColumnDriftMetric(column_name=col)
          for col in KEY_FEATURES if col in ref_features.columns],
    ])

    report.run(reference_data=ref_features, current_data=cur_features)

    # Save HTML report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    html_path = REPORTS_PATH / f"drift_report_{timestamp}.html"
    report.save_html(str(html_path))
    print(f"Drift report saved: {html_path}")

    # Extract summary for logging
    result         = report.as_dict()
    drift_detected = result["metrics"][0]["result"]["dataset_drift"]
    drift_share    = result["metrics"][0]["result"]["share_of_drifted_columns"]

    summary = {
        "timestamp":      timestamp,
        "drift_detected": drift_detected,
        "drift_share":    round(drift_share, 3),
        "reference_rows": len(reference),
        "current_rows":   len(current),
        "report_path":    str(html_path),
    }

    # Save JSON summary
    json_path = REPORTS_PATH / f"drift_summary_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def run_model_performance_report(reference, current):
    """
    Tracks prediction quality over time.
    Both reference and current must include 'is_canceled' column.
    """
    import xgboost as xgb

    print("Running model performance report...")

    model_path = BASE / "models/registry/champion_model.ubj"
    if not model_path.exists():
        print("No champion model found, skipping performance report.")
        return None

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    feature_cols = [c for c in reference.columns if c != "is_canceled"]

    # Work on copies to avoid mutating originals
    ref = reference.copy()
    cur = current.copy()

    ref["prediction"] = model.predict_proba(ref[feature_cols])[:, 1]
    cur["prediction"] = model.predict_proba(cur[feature_cols])[:, 1]

    ref["target"] = ref["is_canceled"]
    cur["target"] = cur["is_canceled"]

    # Evidently classification preset needs prediction + target columns
    report_cols = feature_cols + ["prediction", "target"]

    report = Report(metrics=[ClassificationPreset()])
    report.run(
        reference_data=ref[report_cols],
        current_data=cur[report_cols]
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    html_path = REPORTS_PATH / f"performance_report_{timestamp}.html"
    report.save_html(str(html_path))
    print(f"Performance report saved: {html_path}")

    return str(html_path)


def check_critical_features(summary):
    """
    Flag if key AU-specific features are drifting.
    """
    print("\n── Drift Summary ─────────────────────────────────────────")
    print(f"  Drift detected : {summary['drift_detected']}")
    print(f"  Drifted cols   : {summary['drift_share']:.1%}")
    print(f"  Reference rows : {summary['reference_rows']:,}")
    print(f"  Current rows   : {summary['current_rows']:,}")

    if summary["drift_detected"]:
        print("\n  ⚠️  ACTION REQUIRED:")
        print("  Data distribution has shifted.")
        print("  Recommendation: trigger Airflow retraining DAG manually.")
        print("  Check: seasonal patterns, holiday flags, ADR ranges.")
    else:
        print("\n  ✅ No significant drift detected. Model is stable.")
    print("──────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    print("=" * 55)
    print("  AU Hotels — Drift Monitoring")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    df = load_data()
    reference, current = split_reference_current(df)

    summary = run_data_drift_report(reference, current)
    check_critical_features(summary)

    # Pass full dataframes (with is_canceled) — function handles the split internally
    run_model_performance_report(reference, current)

    print("Monitoring complete. Reports saved to reports/monitoring/")
