"""FastAPI app for bank product recommendations."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

# Make local package importable when app is launched from project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from app.schemas import HealthResponse, RecommendationRequest, RecommendationResponse
from app.service import RecommendationService
from bank_recs.monitoring import ERROR_COUNT, LATENCY, RECOMMENDED_PRODUCT, REQUEST_COUNT

MODEL_PATH = os.getenv("MODEL_PATH", str(PROJECT_ROOT / "models" / "model.joblib"))
TOP_K = int(os.getenv("TOP_K", "7"))

service = RecommendationService(MODEL_PATH, top_k=TOP_K)

app = FastAPI(
    title="Bank Product Recommendation API",
    description="API for recommending new banking products to clients.",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Check service status and model availability."""
    return HealthResponse(
        status="ok" if service.model_loaded else "ok_without_model",
        model_loaded=service.model_loaded,
        model_path=service.model_path,
    )


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(payload: RecommendationRequest) -> RecommendationResponse:
    """Return top-k product recommendations for one client."""
    started = time.perf_counter()
    REQUEST_COUNT.inc()
    try:
        payload_dict = payload.model_dump(exclude_none=True)
        recs = service.recommend(payload_dict)
        for product in recs:
            RECOMMENDED_PRODUCT.labels(product=product).inc()
        return RecommendationResponse(
            client_id=payload.ncodpers,
            recommendations=recs,
            top_k=TOP_K,
            model_loaded=service.model_loaded,
        )
    except Exception as exc:  # noqa: BLE001
        ERROR_COUNT.inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        LATENCY.observe(time.perf_counter() - started)


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus-compatible metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
