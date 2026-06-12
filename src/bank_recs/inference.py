"""Inference helpers for API and batch scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bank_recs.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, PRODUCT_COLS, TOP_K, USER_COL
from bank_recs.modeling import load_model, predict_ranked_products

DEFAULT_POPULARITY = [
    "ind_cco_fin_ult1",
    "ind_recibo_ult1",
    "ind_ecue_fin_ult1",
    "ind_nom_pens_ult1",
    "ind_nomina_ult1",
    "ind_tjcr_fin_ult1",
    "ind_cno_fin_ult1",
    "ind_ctop_fin_ult1",
]


def request_to_candidates(payload: dict, *, top_k: int = TOP_K) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert API request payload to client-product candidate dataframe."""
    client_id = int(payload.get(USER_COL) or payload.get("client_id") or -1)
    products = payload.get("products") or {}

    owned_products = {product for product, value in products.items() if int(value or 0) == 1}
    candidate_products = [product for product in PRODUCT_COLS if product not in owned_products]

    prev_product_count = len(owned_products)
    rows = []
    meta = []
    for product in candidate_products:
        row = {}
        for col in NUMERIC_FEATURES:
            row[col] = payload.get(col, np.nan)
        for col in CATEGORICAL_FEATURES:
            row[col] = payload.get(col, "unknown")
        row["product"] = product
        row["current_product_owned"] = 0
        row["prev_product_count"] = prev_product_count
        row["product_popularity"] = payload.get("product_popularity", 0.0)
        rows.append(row)
        meta.append({USER_COL: client_id, "product": product})

    X = pd.DataFrame(rows)
    metadata = pd.DataFrame(meta)
    for col in NUMERIC_FEATURES:
        if col not in X.columns:
            X[col] = np.nan
    for col in CATEGORICAL_FEATURES:
        if col not in X.columns:
            X[col] = "unknown"
    return X[NUMERIC_FEATURES + CATEGORICAL_FEATURES], metadata


def recommend_from_payload(model, payload: dict, *, top_k: int = TOP_K) -> list[str]:
    """Return product codes for one client."""
    X, metadata = request_to_candidates(payload, top_k=top_k)
    if model is None:
        products = payload.get("products") or {}
        owned = {product for product, value in products.items() if int(value or 0) == 1}
        return [product for product in DEFAULT_POPULARITY if product not in owned][:top_k]
    ranked = predict_ranked_products(model, X, metadata, k=top_k)
    client_id = int(payload.get(USER_COL) or payload.get("client_id") or -1)
    return ranked.get(client_id, [])[:top_k]
