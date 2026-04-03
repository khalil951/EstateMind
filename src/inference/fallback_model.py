"""Non-leaky tabular fallback models for EstateMind valuation serving."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


@dataclass
class FallbackPrediction:
    """Result returned by the trained fallback tabular model."""
    estimated_price: int
    price_per_m2: int
    prediction_mode: str
    warnings: list[str]
    model_info: dict[str, Any]
    uncertainty_reasons: list[str]


class FallbackTabularModelService:
    """Serve CatBoost fallback models trained on request-time tabular features."""

    def __init__(self, artifacts_dir: str | Path | None = None) -> None:
        root = Path(__file__).resolve().parents[2]
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else root / "artifacts" / "models" / "fallback_tabular"
        self.manifest_path = self.artifacts_dir / "manifest.json"
        self._manifest: dict[str, Any] | None = None
        self._models: dict[str, Any] = {}

    def _load_manifest(self) -> dict[str, Any] | None:
        if self._manifest is None:
            if not self.manifest_path.exists():
                return None
            self._manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return self._manifest

    def _get_model_spec(self, property_type: str, transaction_type: str) -> dict[str, Any] | None:
        manifest = self._load_manifest()
        normalized_type = str(property_type).strip().title()
        normalized_tx = str(transaction_type).strip().lower()
        models = manifest.get("models", {}) #type: ignore
        for key in (f"{normalized_type.lower()}__{normalized_tx}", f"{normalized_type.lower()}__sale", "global__sale"):
            spec = models.get(key)
            if spec:
                return spec
        return None

    def _load_model(self, path: Path) -> Any:
        cache_key = str(path)
        if cache_key not in self._models:
            self._models[cache_key] = joblib.load(path)
        return self._models[cache_key]

    @staticmethod
    def _prepare_features(mapped: dict[str, Any], spec: dict[str, Any]) -> pd.DataFrame:
        priors = spec.get("priors", {})
        city = str(mapped["city"]).strip().lower()
        governorate = str(mapped["governorate"]).strip().lower()
        property_type = str(mapped["property_type"]).strip().title()
        city_governorate = f"{city}__{governorate}"

        local_avg = float(priors.get("city_governorate_price_m2", {}).get(city_governorate, priors.get("global_price_m2", 1450.0)))
        gov_avg = float(priors.get("governorate_price_m2", {}).get(governorate, priors.get("global_price_m2", 1450.0)))
        rooms = float(mapped.get("rooms") or (int(mapped.get("bedrooms", 0)) + (0 if property_type == "Terrain" else 1)))
        surface = float(mapped["surface_m2"])
        fill_values = spec.get("fill_values", {})

        features = {
            "transaction_type": str(mapped.get("transaction_type") or "sale").strip().lower(),
            "property_type": property_type.lower(),
            "surface_m2": surface,
            "rooms": rooms,
            "bedrooms": float(mapped.get("bedrooms", 0)),
            "bathrooms": float(mapped.get("bathrooms", 0)),
            "governorate": governorate,
            "city": city,
            "latitude": float(mapped["latitude"]) if mapped.get("latitude") is not None else float(fill_values.get("latitude", 36.8)),
            "longitude": float(mapped["longitude"]) if mapped.get("longitude") is not None else float(fill_values.get("longitude", 10.18)),
            "city_governorate": city_governorate,
            "local_avg_price_m2": local_avg,
            "gov_avg_price_m2": gov_avg,
            "size_x_local_price": surface * local_avg,
        }
        return pd.DataFrame([features])

    def predict(self, mapped: dict[str, Any]) -> FallbackPrediction | None:
        """Predict from the fallback artifact family if a compatible model exists."""

        try:
            spec = self._get_model_spec(mapped["property_type"], mapped.get("transaction_type", "sale"))
        except Exception:
            return None
        if spec is None:
            return None

        model_path = self.artifacts_dir / spec["artifact"]
        if not model_path.exists():
            return None
        model = self._load_model(model_path)
        frame = self._prepare_features(mapped, spec)
        pred_log = float(model.predict(frame[spec["feature_columns"]])[0])
        estimated_price = int(round(float(np.expm1(pred_log))))
        ppm = int(round(estimated_price / max(float(mapped["surface_m2"]), 1.0)))
        return FallbackPrediction(
            estimated_price=max(estimated_price, 1),
            price_per_m2=max(ppm, 1),
            prediction_mode="fallback_model",
            warnings=["fallback_tabular_model_used"],
            model_info={
                "artifact": str(model_path),
                "model_name": spec.get("model_name", "catboost"),
                "scope": spec.get("scope", "fallback"),
                "metrics": spec.get("metrics", {}),
            },
            uncertainty_reasons=["fallback_tabular_model_used"],
        )
