"""
Australian Hotels — Automated Retraining Pipeline
Airflow DAG: runs weekly, retrains if AUC drops below threshold or new data arrives
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
import logging

logger = logging.getLogger(__name__)

# ── DAG default args ───────────────────────────────────────────────────────
default_args = {
    "owner":            "data-science-team",
    "depends_on_past":  False,
    "start_date":       datetime(2025, 1, 1),
    "email_on_failure": True,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
}

AUC_THRESHOLD   = 0.85   # retrain if current AUC drops below this
DATA_PATH_RAW   = "/opt/airflow/data/raw/hotel_bookings.csv"
DATA_PATH_PROC  = "/opt/airflow/data/processed"
MODEL_REGISTRY  = "/opt/airflow/models/registry"
MLFLOW_URI      = "http://mlflow:5000"
EXPERIMENT_NAME = "australian-hotels-cancellation"


# ── Task 1: Ingest & validate new data ────────────────────────────────────
def ingest_and_validate(**context):
    import pandas as pd
    import os
    from great_expectations.core import ExpectationSuite
    import great_expectations as ge

    logger.info("Starting data ingestion...")

    df = pd.read_csv(DATA_PATH_RAW)
    logger.info(f"Loaded {len(df)} rows")

    # Basic data quality checks
    issues = []

    if df["adr"].min() < 0:
        issues.append(f"Negative ADR values found: {(df['adr'] < 0).sum()}")

    if df["is_canceled"].isnull().sum() > 0:
        issues.append("Null values in target column")

    if len(df) < 10000:
        issues.append(f"Insufficient data: only {len(df)} rows")

    null_pct = df.isnull().sum() / len(df)
    high_null_cols = null_pct[null_pct > 0.5].index.tolist()
    if high_null_cols:
        logger.warning(f"High null columns (>50%): {high_null_cols}")

    if issues:
        raise ValueError(f"Data quality failed: {issues}")

    # Push stats to XCom for downstream tasks
    context["ti"].xcom_push(key="row_count",        value=len(df))
    context["ti"].xcom_push(key="cancellation_rate", value=float(df["is_canceled"].mean()))
    context["ti"].xcom_push(key="data_validated",    value=True)

    logger.info(f"Data validation passed. Rows: {len(df)}, "
                f"Cancellation rate: {df['is_canceled'].mean():.3f}")


# ── Task 2: Feature engineering ───────────────────────────────────────────
def run_feature_engineering(**context):
    import pandas as pd
    import sys
    sys.path.insert(0, "/opt/airflow")
    from src.features.build_features import build_features

    logger.info("Running feature engineering...")

    df_raw = pd.read_csv(DATA_PATH_RAW)
    df_out = build_features(df_raw)

    X = df_out.drop(columns=["is_canceled"])
    y = df_out["is_canceled"]

    X.to_csv(f"{DATA_PATH_PROC}/X.csv", index=False)
    y.to_csv(f"{DATA_PATH_PROC}/y.csv", index=False)

    context["ti"].xcom_push(key="feature_count", value=X.shape[1])
    logger.info(f"Features built: {X.shape[1]} columns, {len(X)} rows")


# ── Task 3: Check if retraining is needed ─────────────────────────────────
def check_retrain_needed(**context):
    import json
    import os

    meta_path = f"{MODEL_REGISTRY}/champion_meta.json"

    # If no model exists yet — always train
    if not os.path.exists(meta_path):
        logger.info("No existing model found. Will train.")
        return "train_model"

    with open(meta_path) as f:
        meta = json.load(f)

    current_auc = meta.get("auc_roc", 0)
    logger.info(f"Current champion AUC-ROC: {current_auc:.4f} | Threshold: {AUC_THRESHOLD}")

    if current_auc < AUC_THRESHOLD:
        logger.info(f"AUC {current_auc:.4f} below threshold {AUC_THRESHOLD}. Retraining.")
        return "train_model"

    # Check if new data is significantly larger than last training
    row_count     = context["ti"].xcom_pull(key="row_count", task_ids="ingest_and_validate")
    last_trained  = meta.get("dataset_rows", 0)
    growth        = (row_count - last_trained) / max(last_trained, 1)

    if growth > 0.10:  # 10% more data
        logger.info(f"Dataset grew by {growth:.1%}. Retraining.")
        return "train_model"

    logger.info("Model is healthy and data hasn't grown enough. Skipping retraining.")
    return "skip_training"


# ── Task 4: Train model ────────────────────────────────────────────────────
def train_model(**context):
    import pandas as pd
    import xgboost as xgb
    import mlflow
    import mlflow.xgboost
    import json
    import pickle
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

    logger.info("Starting model training...")

    X = pd.read_csv(f"{DATA_PATH_PROC}/X.csv")
    y = pd.read_csv(f"{DATA_PATH_PROC}/y.csv").squeeze()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Best params from Optuna (stored from Phase 4)
    params = {
        "n_estimators":      997,
        "max_depth":         10,
        "learning_rate":     0.0201,
        "subsample":         0.9401,
        "colsample_bytree":  0.8288,
        "min_child_weight":  3,
        "gamma":             0.6079,
        "reg_alpha":         0.5214,
        "reg_lambda":        1.4526,
        "scale_pos_weight":  (y_train == 0).sum() / (y_train == 1).sum(),
        "use_label_encoder": False,
        "eval_metric":       "auc",
        "verbosity":         0,
        "random_state":      42,
    }

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"airflow_retrain_{datetime.now().strftime('%Y%m%d')}"):
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "auc_roc":  round(roc_auc_score(y_test, y_proba), 4),
            "f1":       round(f1_score(y_test, y_pred), 4),
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
        }

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, "model")

        logger.info(f"Training complete: {metrics}")

    # Push metrics for evaluation task
    context["ti"].xcom_push(key="new_auc",      value=metrics["auc_roc"])
    context["ti"].xcom_push(key="new_f1",       value=metrics["f1"])
    context["ti"].xcom_push(key="new_accuracy", value=metrics["accuracy"])
    context["ti"].xcom_push(key="model_obj",    value=None)  # passed via file

    # Save model temporarily for evaluation
    model.save_model(f"{MODEL_REGISTRY}/candidate_model.ubj")
    with open(f"{MODEL_REGISTRY}/candidate_metrics.json", "w") as f:
        json.dump(metrics, f)

    logger.info(f"Candidate model saved. AUC: {metrics['auc_roc']:.4f}")


# ── Task 5: Evaluate — promote or reject ──────────────────────────────────
def evaluate_and_promote(**context):
    import json
    import os
    import shutil

    with open(f"{MODEL_REGISTRY}/candidate_metrics.json") as f:
        new_metrics = json.load(f)

    new_auc = new_metrics["auc_roc"]
    champion_path = f"{MODEL_REGISTRY}/champion_meta.json"

    # Load current champion if exists
    if os.path.exists(champion_path):
        with open(champion_path) as f:
            current_meta = json.load(f)
        current_auc = current_meta.get("auc_roc", 0)
    else:
        current_auc = 0
        current_meta = {}

    logger.info(f"Current champion AUC: {current_auc:.4f} | Candidate AUC: {new_auc:.4f}")

    if new_auc >= current_auc:
        # Promote candidate to champion
        shutil.copy(
            f"{MODEL_REGISTRY}/candidate_model.ubj",
            f"{MODEL_REGISTRY}/champion_model.ubj"
        )

        new_meta = {
            **current_meta,
            **new_metrics,
            "model_type":    "XGBoost_Tuned",
            "threshold":     0.5,
            "trained_on":    datetime.now().strftime("%Y-%m-%d"),
            "dataset_rows":  context["ti"].xcom_pull(
                                key="row_count", task_ids="ingest_and_validate"),
            "feature_count": context["ti"].xcom_pull(
                                key="feature_count", task_ids="run_feature_engineering"),
            "promoted_by":   "airflow_auto_retrain",
        }

        with open(champion_path, "w") as f:
            json.dump(new_meta, f, indent=2)

        logger.info(f"✅ New champion promoted! AUC: {new_auc:.4f} (was {current_auc:.4f})")
        context["ti"].xcom_push(key="promoted", value=True)

    else:
        logger.info(f"❌ Candidate ({new_auc:.4f}) did not beat champion ({current_auc:.4f}). Keeping current model.")
        context["ti"].xcom_push(key="promoted", value=False)

    # Clean up candidate files
    os.remove(f"{MODEL_REGISTRY}/candidate_metrics.json")


# ── Task 6: Send notification ──────────────────────────────────────────────
def send_notification(**context):
    promoted    = context["ti"].xcom_pull(key="promoted",  task_ids="evaluate_and_promote")
    new_auc     = context["ti"].xcom_pull(key="new_auc",   task_ids="train_model")
    row_count   = context["ti"].xcom_pull(key="row_count", task_ids="ingest_and_validate")

    status  = "✅ PROMOTED" if promoted else "⏭️ KEPT EXISTING"
    message = (
        f"AU Hotels Retraining Pipeline — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"Status:     {status}\n"
        f"New AUC:    {new_auc:.4f}\n"
        f"Rows used:  {row_count:,}\n"
        f"Threshold:  {AUC_THRESHOLD}\n"
    )

    logger.info(f"Pipeline notification:\n{message}")
    # In production: replace with Slack webhook or email
    # requests.post(SLACK_WEBHOOK_URL, json={"text": message})


# ── DAG definition ─────────────────────────────────────────────────────────
with DAG(
    dag_id="au_hotels_retrain_pipeline",
    default_args=default_args,
    description="Weekly retraining pipeline for AU Hotels cancellation model",
    schedule_interval="0 2 * * 1",   # Every Monday at 2am
    catchup=False,
    max_active_runs=1,
    tags=["hotels", "australia", "mlops", "retraining"],
) as dag:

    start = EmptyOperator(task_id="start")
    end   = EmptyOperator(task_id="end")

    t_ingest = PythonOperator(
        task_id="ingest_and_validate",
        python_callable=ingest_and_validate,
    )

    t_features = PythonOperator(
        task_id="run_feature_engineering",
        python_callable=run_feature_engineering,
    )

    t_check = BranchPythonOperator(
        task_id="check_retrain_needed",
        python_callable=check_retrain_needed,
    )

    t_train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        execution_timeout=timedelta(hours=2),
    )

    t_evaluate = PythonOperator(
        task_id="evaluate_and_promote",
        python_callable=evaluate_and_promote,
    )

    t_notify = PythonOperator(
        task_id="send_notification",
        python_callable=send_notification,
        trigger_rule="none_failed",
    )

    t_skip = EmptyOperator(task_id="skip_training")

    # ── DAG flow ──────────────────────────────────────────────────────────
    start >> t_ingest >> t_features >> t_check
    t_check   >> t_train >> t_evaluate >> t_notify >> end
    t_check   >> t_skip  >> t_notify  >> end
