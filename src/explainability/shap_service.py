"""Feature contribution summaries for EstateMind explainability."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from catboost import Pool

from src.inference.model_registry import ModelHandle


class ShapService:
    """Model-aware SHAP adapter with explicit fallback mode reporting."""

    def _catboost_shap(
        self,
        estimated_price: int,
        feature_frame: pd.DataFrame,
        model_handle: ModelHandle,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, list[str]]:
        estimator = model_handle.estimator
        warnings: list[str] = []
        cat_feature_indices = [
            idx
            for idx, dtype in enumerate(feature_frame.dtypes)
            if str(dtype) in {"object", "category", "string"}
        ]
        shap_pool = Pool(feature_frame, cat_features=cat_feature_indices)
        shap_values = estimator.get_feature_importance(shap_pool, type="ShapValues") #type: ignore
        row = shap_values[0]
        contributions = row[:-1]
        baseline_log = float(row[-1])
        baseline_price = float(np.expm1(baseline_log))

        non_zero: list[dict[str, Any]] = []
        for feature_name, value in zip(feature_frame.columns.tolist(), contributions.tolist()):
            amount = int(round(float(value) * estimated_price))
            if amount != 0:
                non_zero.append({"feature": str(feature_name), "value": amount})
        non_zero = sorted(non_zero, key=lambda item: abs(int(item["value"])), reverse=True)[:6]

        shap = [{"feature": "Baseline", "value": int(max(0, round(baseline_price)))}]
        shap.extend(non_zero)
        shap.append({"feature": "Final", "value": int(estimated_price)})
        features_impact = [
            {
                "feature": item["feature"],
                "pct": int(round((int(item["value"]) / max(estimated_price, 1)) * 100)),
                "amount": int(item["value"]),
            }
            for item in non_zero
        ]
        return shap, features_impact, "true_shap", warnings

    def _fallback(
        self,
        estimated_price: int,
        fused: dict[str, Any],
        model_handle: ModelHandle | None,
        reason: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, list[str]]:
        structured = fused["structured"]
        warnings = ["explainability_fallback", f"explainability_fallback_reason:{reason}"]
        contributions: list[dict[str, Any]] = []

        scale = max(int(estimated_price), 1)
        size_delta = (float(structured.get("surface_m2", 0.0)) - 80.0) / 220.0
        size_boost = int(round(scale * max(min(size_delta, 0.30), -0.15)))
        condition_map = {
            "New": int(round(scale * 0.12)),
            "Excellent": int(round(scale * 0.08)),
            "Good": 0,
            "Fair": -int(round(scale * 0.06)),
            "Needs Renovation": -int(round(scale * 0.12)),
        }
        condition_boost = int(condition_map.get(str(structured["condition"]), 0))
        amenity_boost = int(
            round(
                scale
                * (
                    (0.08 if structured.get("has_pool") else 0.0)
                    + (0.05 if structured.get("has_garden") else 0.0)
                    + (0.04 if structured.get("has_parking") else 0.0)
                    + (0.09 if structured.get("sea_view") else 0.0)
                    + (0.02 if structured.get("elevator") else 0.0)
                )
            )
        )
        text_score = float(fused["nlp"].get("description_score", 0.5))
        text_boost = int(round(scale * ((text_score - 0.5) * 0.20)))
        image_boost = int(round(scale * float(fused["vision"].get("price_band_effect", 0.0)) * 0.08))

        contributions.extend(
            [
                {"feature": "Property Size", "value": size_boost},
                {"feature": "Condition", "value": condition_boost},
                {"feature": "Amenities", "value": amenity_boost},
                {"feature": "Description Quality", "value": text_boost},
                {"feature": "Visual Price Band", "value": image_boost},
            ]
        )
        contributions = [item for item in contributions if item["value"] != 0]

        baseline = estimated_price - sum(int(item["value"]) for item in contributions)
        shap = [{"feature": "Baseline", "value": int(round(baseline))}]
        shap.extend(contributions)
        if model_handle is not None and model_handle.bundle_error:
            shap.append({"feature": "Bundle Gap", "value": 0})
        shap.append({"feature": "Final", "value": int(estimated_price)})

        features_impact = [
            {
                "feature": item["feature"],
                "pct": int(round((item["value"] / max(estimated_price, 1)) * 100)),
                "amount": int(item["value"]),
            }
            for item in contributions
        ]
        features_impact = sorted(features_impact, key=lambda item: abs(int(item["amount"])), reverse=True)[:5]
        return shap, features_impact, "fallback", warnings

    def explain(
        self,
        estimated_price: int,
        fused: dict[str, Any],
        model_handle: ModelHandle | None,
        feature_frame: pd.DataFrame | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, list[str]]:
        """Return feature contributions and an explicit explanation mode."""

        if (
            model_handle is not None
            and model_handle.bundle_available
            and model_handle.model_name.lower() == "catboost"
            and feature_frame is not None
        ):
            try:
                return self._catboost_shap(estimated_price, feature_frame, model_handle)
            except Exception:
                return self._fallback(estimated_price, fused, model_handle, "catboost_shap_runtime_error")

        if model_handle is None:
            reason = "missing_model_handle"
        elif not model_handle.bundle_available:
            reason = "bundle_unavailable"
        elif model_handle.model_name.lower() != "catboost":
            reason = "non_catboost_estimator"
        elif feature_frame is None:
            reason = "missing_feature_frame"
        else:
            reason = "unknown"
        return self._fallback(estimated_price, fused, model_handle, reason)
