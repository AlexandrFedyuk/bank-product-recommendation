"""Business logic for API service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bank_recs.inference import recommend_from_payload
from bank_recs.modeling import load_model


class RecommendationService:
    """Thin model wrapper used by FastAPI endpoints."""

    def __init__(self, model_path: str | Path, top_k: int = 7) -> None:
        self.model_path = str(model_path)
        self.top_k = top_k
        self.model = None
        self.load_error: str | None = None
        self.reload()

    def reload(self) -> None:
        path = Path(self.model_path)
        if not path.exists():
            self.model = None
            self.load_error = f"Model file not found: {path}"
            return
        try:
            self.model = load_model(path)
            self.load_error = None
        except Exception as exc:  # noqa: BLE001
            self.model = None
            self.load_error = str(exc)

    @property
    def model_loaded(self) -> bool:
        return self.model is not None

    def recommend(self, payload: dict[str, Any]) -> list[str]:
        return recommend_from_payload(self.model, payload, top_k=self.top_k)
