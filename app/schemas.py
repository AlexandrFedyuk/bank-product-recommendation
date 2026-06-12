"""Pydantic schemas for recommendation API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from bank_recs.config import PRODUCT_COLS


class RecommendationRequest(BaseModel):
    """Input data for one client.

    Most fields are optional so the endpoint can be used in a demo mode. The
    trained sklearn pipeline will impute missing numeric values and encode
    unknown categories.
    """

    ncodpers: int | None = Field(default=None, description="Client identifier")
    age: int | float | None = None
    antiguedad: int | float | None = None
    renta: int | float | None = None
    ind_nuevo: int | None = None
    indrel: int | None = None
    ind_actividad_cliente: int | None = None
    months_from_join: int | float | None = None

    ind_empleado: str | None = None
    pais_residencia: str | None = None
    sexo: str | None = None
    indrel_1mes: str | None = None
    tiprel_1mes: str | None = None
    indresi: str | None = None
    indext: str | None = None
    conyuemp: str | None = None
    canal_entrada: str | None = None
    indfall: str | None = None
    cod_prov: str | int | None = None
    nomprov: str | None = None
    segmento: str | None = None

    products: dict[str, int] = Field(
        default_factory=dict,
        description="Current product ownership state: product_code -> 0/1",
        examples=[{product: 0 for product in PRODUCT_COLS[:3]}],
    )


class RecommendationResponse(BaseModel):
    client_id: int | None
    recommendations: list[str]
    top_k: int
    model_loaded: bool


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
