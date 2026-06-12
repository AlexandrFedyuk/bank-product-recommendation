#!/usr/bin/env python3
"""Generate a small Santander-like dataset for smoke tests.

This script does not replace the real training data. It only lets you verify
that notebooks, training code and API are wired correctly before downloading the
large dataset.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from bank_recs.config import PRODUCT_COLS, RANDOM_STATE


def main() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    output_path = PROJECT_ROOT / "data" / "raw" / "train_ver2_sample.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    months = pd.to_datetime(["2016-03-28", "2016-04-28", "2016-05-28"])
    n_users = 1200
    users = np.arange(100000, 100000 + n_users)
    rows = []

    for user in users:
        segment = rng.choice(["01 - TOP", "02 - PARTICULARES", "03 - UNIVERSITARIO"], p=[0.05, 0.75, 0.20])
        age = int(np.clip(rng.normal(42, 15), 18, 90))
        base_products = {product: int(rng.random() < 0.03) for product in PRODUCT_COLS}
        # Current account and receipts are common products.
        base_products["ind_cco_fin_ult1"] = int(rng.random() < 0.65)
        base_products["ind_recibo_ult1"] = int(rng.random() < 0.35)
        base_products["ind_ecue_fin_ult1"] = int(rng.random() < 0.15)
        current = base_products.copy()

        for month in months:
            # Some users acquire new products over time.
            if month.month in [4, 5]:
                if rng.random() < 0.10:
                    candidates = [p for p, v in current.items() if v == 0]
                    if candidates:
                        product = rng.choice(candidates)
                        current[product] = 1
                if segment == "03 - UNIVERSITARIO" and rng.random() < 0.08:
                    current["ind_tjcr_fin_ult1"] = 1
                if age > 55 and rng.random() < 0.06:
                    current["ind_nom_pens_ult1"] = 1

            row = {
                "fecha_dato": month.strftime("%Y-%m-%d"),
                "ncodpers": int(user),
                "ind_empleado": rng.choice(["A", "B", "F", "N", "P"], p=[0.02, 0.03, 0.02, 0.90, 0.03]),
                "pais_residencia": "ES",
                "sexo": rng.choice(["H", "V"]),
                "age": age,
                "fecha_alta": "2014-01-01",
                "ind_nuevo": 0,
                "antiguedad": rng.integers(6, 180),
                "indrel": 1,
                "ult_fec_cli_1t": "unknown",
                "indrel_1mes": "1",
                "tiprel_1mes": rng.choice(["A", "I"], p=[0.55, 0.45]),
                "indresi": "S",
                "indext": "N",
                "conyuemp": "unknown",
                "canal_entrada": rng.choice(["KHE", "KAT", "KFC", "KFA"]),
                "indfall": "N",
                "tipodom": 1,
                "cod_prov": int(rng.integers(1, 53)),
                "nomprov": rng.choice(["MADRID", "BARCELONA", "VALENCIA", "SEVILLA"]),
                "ind_actividad_cliente": int(rng.random() < 0.55),
                "renta": float(rng.lognormal(mean=11.0, sigma=0.5)),
                "segmento": segment,
            }
            row.update(current)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Sample data saved to {output_path}")
    print(df.shape)


if __name__ == "__main__":
    main()
