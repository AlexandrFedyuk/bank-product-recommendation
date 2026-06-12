"""Data loading and basic cleaning."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from bank_recs.config import DATE_COL, PRODUCT_COLS, USER_COL


def load_raw_data(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Load raw Santander-style CSV file.

    Parameters
    ----------
    path:
        Path to `train_ver2.csv` or sample CSV.
    nrows:
        Optional row limit for quick local experiments.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Файл данных не найден: {path}. Скачайте датасет и положите CSV в data/raw/."
        )
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def normalize_month(value: str) -> str:
    """Convert `YYYY-MM-DD` or `YYYY-MM` value to `YYYY-MM`."""
    return str(value)[:7]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean types and add helper month fields.

    The function is intentionally conservative: it does not drop rows, because for
    recommendation tasks missingness can itself be a useful signal.
    """
    df = df.copy()

    if DATE_COL not in df.columns:
        raise ValueError(f"В датасете нет обязательной колонки {DATE_COL}")
    if USER_COL not in df.columns:
        raise ValueError(f"В датасете нет обязательной колонки {USER_COL}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df["month"] = df[DATE_COL].dt.to_period("M").astype(str)

    if "fecha_alta" in df.columns:
        df["fecha_alta"] = pd.to_datetime(df["fecha_alta"], errors="coerce")
        df["months_from_join"] = (
            (df[DATE_COL].dt.year - df["fecha_alta"].dt.year) * 12
            + (df[DATE_COL].dt.month - df["fecha_alta"].dt.month)
        )
    else:
        df["months_from_join"] = np.nan

    for col in ["age", "antiguedad", "renta", "ind_nuevo", "indrel", "ind_actividad_cliente", "cod_prov"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in PRODUCT_COLS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int8")

    # Normalize categorical columns to strings so sklearn's OneHotEncoder is stable.
    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        if col != "month":
            df[col] = df[col].fillna("unknown").astype(str).str.strip()

    df["prev_product_count"] = df[PRODUCT_COLS].sum(axis=1).astype("int16")
    return df


def available_months(df: pd.DataFrame) -> list[str]:
    """Return sorted list of months available in the data."""
    if "month" not in df.columns:
        df = clean_dataframe(df)
    return sorted(df["month"].dropna().unique().tolist())


def month_slice(df: pd.DataFrame, month: str) -> pd.DataFrame:
    """Return rows for a given YYYY-MM month."""
    month = normalize_month(month)
    if "month" not in df.columns:
        df = clean_dataframe(df)
    return df.loc[df["month"] == month].copy()


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Create missing columns with NA values."""
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    return df
