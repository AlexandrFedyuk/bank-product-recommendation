"""Prometheus metrics used by the FastAPI service."""

from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "recommendation_requests_total",
    "Total number of recommendation requests",
)

ERROR_COUNT = Counter(
    "recommendation_errors_total",
    "Total number of failed recommendation requests",
)

LATENCY = Histogram(
    "recommendation_latency_seconds",
    "Recommendation request latency in seconds",
)

RECOMMENDED_PRODUCT = Counter(
    "recommended_product_total",
    "Number of times a product was recommended",
    ["product"],
)
