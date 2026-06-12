"""Feature generation for client-product ranking dataset."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from bank_recs.config import (
    BASE_CLIENT_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PRODUCT_COLS,
    USER_COL,
)
from bank_recs.data import clean_dataframe, month_slice, normalize_month


def product_popularity(history: pd.DataFrame) -> dict[str, float]:
    """Share of clients having each product in the history month."""
    if history.empty:
        return {product: 0.0 for product in PRODUCT_COLS}
    return {product: float(history[product].mean()) for product in PRODUCT_COLS}


def _client_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in BASE_CLIENT_FEATURES if col in df.columns] + ["months_from_join", "prev_product_count"]


def _safe_int(value, default: int = 0) -> int:
    if pd.isna(value):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build_client_product_frame(
    history: pd.DataFrame,
    target: pd.DataFrame | None = None,
    *,
    include_all_candidates: bool = False,
    negative_ratio: int = 5,
    max_users: int | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series | None, pd.DataFrame]:
    """Create ranking rows in client-product format.

    Each output row describes a pair `(client, product)` based on the state in a
    history month. If `target` is provided, target = 1 means that the product was
    not owned in history but appeared in the target month.

    Parameters
    ----------
    history:
        Rows for month t.
    target:
        Rows for month t+1. If None, function creates inference candidates.
    include_all_candidates:
        If True, include every product that a client does not own. Useful for
        validation/inference. If False, keep all positives and sample negatives.
    negative_ratio:
        Number of negative examples sampled per positive example for training.
    max_users:
        Optional cap for fast local experiments on large data.
    random_state:
        Reproducible sampling seed.
    """
    rng = np.random.default_rng(random_state)
    history = history.copy()
    if target is not None:
        target = target.copy()

    if max_users is not None and len(history) > max_users:
        users = history[USER_COL].drop_duplicates().sample(max_users, random_state=random_state)
        history = history[history[USER_COL].isin(users)]
        if target is not None:
            target = target[target[USER_COL].isin(users)]

    client_cols = _client_feature_columns(history)
    history = history[[USER_COL] + client_cols + PRODUCT_COLS].drop_duplicates(USER_COL)

    if target is not None:
        target_products = target[[USER_COL] + PRODUCT_COLS].drop_duplicates(USER_COL)
        merged = history.merge(target_products, on=USER_COL, how="left", suffixes=("_hist", "_target"))
    else:
        merged = history.copy()

    popularity = product_popularity(history)
    records: list[dict] = []
    meta: list[dict] = []
    labels: list[int] = []

    for _, row in merged.iterrows():
        base = {col: row.get(col, np.nan) for col in client_cols}
        client_id = row[USER_COL]
        positive_products: list[str] = []
        candidate_negatives: list[str] = []

        for product in PRODUCT_COLS:
            owned = _safe_int(row.get(f"{product}_hist", row.get(product, 0)))
            if target is not None:
                future = _safe_int(row.get(f"{product}_target", 0))
                label = int(owned == 0 and future == 1)
            else:
                label = None

            if owned == 0:
                if label == 1:
                    positive_products.append(product)
                else:
                    candidate_negatives.append(product)

        if include_all_candidates or target is None:
            selected = positive_products + candidate_negatives
        else:
            if positive_products:
                n_neg = min(len(candidate_negatives), max(negative_ratio * len(positive_products), 1))
            else:
                # Keep a small number of pure negatives so the model sees clients
                # without purchases, but do not explode the dataset.
                n_neg = min(len(candidate_negatives), 2)
            selected_negatives = list(rng.choice(candidate_negatives, size=n_neg, replace=False)) if n_neg else []
            selected = positive_products + selected_negatives

        for product in selected:
            owned = _safe_int(row.get(f"{product}_hist", row.get(product, 0)))
            feature_row = dict(base)
            feature_row.update(
                {
                    "product": product,
                    "current_product_owned": owned,
                    "product_popularity": popularity.get(product, 0.0),
                }
            )
            records.append(feature_row)
            meta.append({USER_COL: client_id, "product": product})
            if target is not None:
                future = _safe_int(row.get(f"{product}_target", 0))
                labels.append(int(owned == 0 and future == 1))

    X = pd.DataFrame(records)
    metadata = pd.DataFrame(meta)
    if X.empty:
        raise ValueError("Не удалось сформировать обучающие строки. Проверьте месяцы и наличие продуктов.")

    # Ensure stable schema.
    for col in NUMERIC_FEATURES:
        if col not in X.columns:
            X[col] = np.nan
    for col in CATEGORICAL_FEATURES:
        if col not in X.columns:
            X[col] = "unknown"

    y = pd.Series(labels, name="target") if target is not None else None
    return X[NUMERIC_FEATURES + CATEGORICAL_FEATURES], y, metadata


def build_train_validation_sets(
    df: pd.DataFrame,
    train_history_month: str,
    train_target_month: str,
    valid_history_month: str | None = None,
    valid_target_month: str | None = None,
    *,
    negative_ratio: int = 5,
    max_users: int | None = None,
    random_state: int = 42,
):
    """Prepare train and validation matrices."""
    df = clean_dataframe(df)
    train_history = month_slice(df, train_history_month)
    train_target = month_slice(df, train_target_month)
    if train_history.empty or train_target.empty:
        raise ValueError(
            f"Нет данных для train pair: {train_history_month} -> {train_target_month}. "
            f"Доступные месяцы: {sorted(df['month'].unique())[:5]} ... {sorted(df['month'].unique())[-5:]}"
        )

    X_train, y_train, train_meta = build_client_product_frame(
        train_history,
        train_target,
        include_all_candidates=False,
        negative_ratio=negative_ratio,
        max_users=max_users,
        random_state=random_state,
    )

    if valid_history_month and valid_target_month:
        valid_history = month_slice(df, valid_history_month)
        valid_target = month_slice(df, valid_target_month)
        X_valid, y_valid, valid_meta = build_client_product_frame(
            valid_history,
            valid_target,
            include_all_candidates=True,
            negative_ratio=negative_ratio,
            max_users=max_users,
            random_state=random_state,
        )
    else:
        X_valid = y_valid = valid_meta = None

    return X_train, y_train, train_meta, X_valid, y_valid, valid_meta


def get_actual_new_products(history: pd.DataFrame, target: pd.DataFrame) -> dict[int, list[str]]:
    """Return mapping client -> products bought between history and target months."""
    history = history[[USER_COL] + PRODUCT_COLS].drop_duplicates(USER_COL)
    target = target[[USER_COL] + PRODUCT_COLS].drop_duplicates(USER_COL)
    merged = history.merge(target, on=USER_COL, how="inner", suffixes=("_hist", "_target"))
    actual: dict[int, list[str]] = {}
    for _, row in merged.iterrows():
        products = []
        for product in PRODUCT_COLS:
            owned = int(row[f"{product}_hist"] or 0)
            future = int(row[f"{product}_target"] or 0)
            if owned == 0 and future == 1:
                products.append(product)
        actual[int(row[USER_COL])] = products
    return actual
