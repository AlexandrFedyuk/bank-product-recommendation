#!/usr/bin/env python3
"""Evaluate a saved model on a pair of months."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from bank_recs.config import PRODUCT_COLS, TOP_K
from bank_recs.data import clean_dataframe, load_raw_data, month_slice
from bank_recs.features import build_client_product_frame, get_actual_new_products
from bank_recs.metrics import ranking_metrics
from bank_recs.modeling import load_model, predict_ranked_products


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved recommendation model")
    parser.add_argument("--data-path", default=str(PROJECT_ROOT / "data" / "raw" / "train_ver2.csv"))
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "models" / "model.joblib"))
    parser.add_argument("--history-month", default="2016-04")
    parser.add_argument("--target-month", default="2016-05")
    parser.add_argument("--max-users", type=int, default=50000)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = clean_dataframe(load_raw_data(args.data_path))
    model = load_model(args.model_path)

    history = month_slice(df, args.history_month)
    target = month_slice(df, args.target_month)
    X, y, meta = build_client_product_frame(
        history,
        target,
        include_all_candidates=True,
        max_users=None if args.max_users == 0 else args.max_users,
    )
    actual = get_actual_new_products(history, target)
    pred = predict_ranked_products(model, X, meta, k=args.top_k)
    metrics = ranking_metrics(actual, pred, PRODUCT_COLS, k=args.top_k)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
