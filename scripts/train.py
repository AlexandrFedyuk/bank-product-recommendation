#!/usr/bin/env python3
"""Train bank product recommendation model and log experiment to MLflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mlflow
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from bank_recs.config import MODELS_DIR, PRODUCT_COLS, RANDOM_STATE, REPORTS_DIR, TOP_K
from bank_recs.data import clean_dataframe, load_raw_data, month_slice, available_months
from bank_recs.features import build_train_validation_sets, get_actual_new_products
from bank_recs.metrics import ranking_metrics
from bank_recs.modeling import fit_model, popularity_recommendations, predict_ranked_products, save_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train bank product recommendation model")
    parser.add_argument("--data-path", default=str(PROJECT_ROOT / "data" / "raw" / "train_ver2.csv"))
    parser.add_argument("--nrows", type=int, default=None, help="Optional CSV row limit for quick local runs")
    parser.add_argument("--train-history-month", default="2016-03")
    parser.add_argument("--train-target-month", default="2016-04")
    parser.add_argument("--valid-history-month", default="2016-04")
    parser.add_argument("--valid-target-month", default="2016-05")
    parser.add_argument("--negative-ratio", type=int, default=5)
    parser.add_argument("--max-users", type=int, default=50000, help="Limit users for local experiments; set 0 for full data")
    parser.add_argument("--model-path", default=str(MODELS_DIR / "model.joblib"))
    parser.add_argument("--mlflow-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--experiment-name", default="bank_product_recommendation")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_users = None if args.max_users == 0 else args.max_users

    print("Loading data...")
    df = load_raw_data(args.data_path, nrows=args.nrows)
    df = clean_dataframe(df)
    print(f"Data shape: {df.shape}")
    print(f"Available months: {available_months(df)}")

    print("Building train/validation datasets...")
    X_train, y_train, train_meta, X_valid, y_valid, valid_meta = build_train_validation_sets(
        df,
        args.train_history_month,
        args.train_target_month,
        args.valid_history_month,
        args.valid_target_month,
        negative_ratio=args.negative_ratio,
        max_users=max_users,
        random_state=args.random_state,
    )
    print(f"Train rows: {X_train.shape}, positive rate: {y_train.mean():.6f}")
    print(f"Valid rows: {X_valid.shape if X_valid is not None else None}")

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name="sklearn_logreg_client_product"):
        mlflow.log_params(
            {
                "model_type": "sklearn_logistic_regression",
                "train_history_month": args.train_history_month,
                "train_target_month": args.train_target_month,
                "valid_history_month": args.valid_history_month,
                "valid_target_month": args.valid_target_month,
                "negative_ratio": args.negative_ratio,
                "max_users": max_users or "full",
                "top_k": args.top_k,
                "random_state": args.random_state,
            }
        )

        print("Training model...")
        model = fit_model(X_train, y_train, random_state=args.random_state)

        print("Evaluating model...")
        valid_history = month_slice(df, args.valid_history_month)
        valid_target = month_slice(df, args.valid_target_month)
        actual = get_actual_new_products(valid_history, valid_target)

        model_pred = predict_ranked_products(model, X_valid, valid_meta, k=args.top_k)
        model_metrics = ranking_metrics(actual, model_pred, PRODUCT_COLS, k=args.top_k)
        model_metrics = {f"model_{k}": v for k, v in model_metrics.items()}

        baseline_pred = popularity_recommendations(valid_history.drop_duplicates("ncodpers"), k=args.top_k)
        baseline_metrics = ranking_metrics(actual, baseline_pred, PRODUCT_COLS, k=args.top_k)
        baseline_metrics = {f"baseline_{k}": v for k, v in baseline_metrics.items()}

        all_metrics = {**model_metrics, **baseline_metrics}
        print(json.dumps(all_metrics, indent=2, ensure_ascii=False))
        mlflow.log_metrics(all_metrics)

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        metrics_path = REPORTS_DIR / "metrics.json"
        metrics_path.write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        mlflow.log_artifact(str(metrics_path))

        model_path = save_model(
            model,
            args.model_path,
            metadata={
                "product_cols": PRODUCT_COLS,
                "top_k": args.top_k,
                "metrics": all_metrics,
                "train_history_month": args.train_history_month,
                "train_target_month": args.train_target_month,
                "valid_history_month": args.valid_history_month,
                "valid_target_month": args.valid_target_month,
            },
        )
        mlflow.log_artifact(str(model_path))
        if model_path.with_suffix(".metadata.json").exists():
            mlflow.log_artifact(str(model_path.with_suffix(".metadata.json")))

        try:
            mlflow.sklearn.log_model(model, artifact_path="sklearn_model")
        except Exception as exc:  # noqa: BLE001
            print(f"MLflow model logging failed, artifact file was still saved: {exc}")

        print(f"Model saved to {model_path}")
        print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
