"""Model training and ranking utilities."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from bank_recs.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, PRODUCT_COLS, TOP_K, USER_COL

def to_string_array(values):
    """Convert categorical values to strings before one-hot encoding."""
    return values.astype(str)

def build_sklearn_model(random_state: int = 42) -> Pipeline:
    """Build a reproducible baseline ML pipeline.

    The model is intentionally lightweight and works on a local MacBook. For a
    stronger solution you can replace the classifier with CatBoost/LightGBM while
    keeping the rest of the project unchanged.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("to_string", FunctionTransformer(to_string_array, validate=False)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    classifier = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def fit_model(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> Pipeline:
    """Fit model and return sklearn pipeline."""
    model = build_sklearn_model(random_state=random_state)
    model.fit(X_train, y_train)
    return model


def predict_ranked_products(
    model: Pipeline,
    X_candidates: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    k: int = TOP_K,
) -> dict[int, list[str]]:
    """Predict probabilities and return top-k products per user."""
    if len(X_candidates) != len(metadata):
        raise ValueError("X_candidates and metadata must have equal length")

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_candidates)[:, 1]
    else:
        scores = model.decision_function(X_candidates)

    ranking_df = metadata.copy()
    ranking_df["score"] = scores
    ranking_df = ranking_df.sort_values([USER_COL, "score"], ascending=[True, False])
    result = (
        ranking_df.groupby(USER_COL)["product"]
        .apply(lambda s: s.head(k).tolist())
        .to_dict()
    )
    return {int(user): products for user, products in result.items()}


def popularity_recommendations(
    history: pd.DataFrame,
    *,
    k: int = TOP_K,
) -> dict[int, list[str]]:
    """Recommend globally popular products that the client does not already own."""
    popularity = history[PRODUCT_COLS].mean().sort_values(ascending=False).index.tolist()
    result: dict[int, list[str]] = {}
    for _, row in history.iterrows():
        recs = [product for product in popularity if int(row.get(product, 0) or 0) == 0]
        result[int(row[USER_COL])] = recs[:k]
    return result


def save_model(model: Pipeline, path: str | Path, metadata: dict | None = None) -> Path:
    """Save model and sidecar metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    if metadata is not None:
        path.with_suffix(".metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return path


def load_model(path: str | Path) -> Pipeline:
    """Load saved sklearn pipeline."""
    return joblib.load(path)
