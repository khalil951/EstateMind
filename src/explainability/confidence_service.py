"""Confidence interval estimation for valuation responses."""

from __future__ import annotations

from typing import Any

from src.inference.model_registry import ModelHandle


class ConfidenceService:
    """Estimate confidence, uncertainty reasons, and price intervals."""

    def estimate(
        self,
        estimated_price: int,
        fused: dict[str, Any],
        comparables: list[dict[str, Any]],
        model_handle: ModelHandle | None,
        *,
        prediction_mode: str,
        warnings: list[str] | None = None,
        ood_flags: list[str] | None = None,
    ) -> dict[str, Any]:
        completeness = float(fused["summary"].get("input_completeness", 0.0))
        image_score = float(fused["vision"].get("quality", {}).get("coverage_score", 0.0))
        text_score = float(fused["nlp"].get("description_score", 0.0))
        sentiment_mode = str(fused["summary"].get("sentiment_mode", "neutral_fallback"))
        comparable_score = min(len(comparables) / 4.0, 1.0)

        uncertainty_reasons = list(warnings or [])
        uncertainty_reasons.extend([f"ood:{flag}" for flag in (ood_flags or [])])

        base_quality = 0.45
        if prediction_mode != "model":
            base_quality = 0.36
            uncertainty_reasons.append("heuristic_prediction_mode")
        elif model_handle is not None:
            r2 = float(model_handle.metrics.get("test_r2", 0.0) or 0.0)
            overfit_gap = float(model_handle.metrics.get("overfit_gap_rmse", 0.0) or 0.0)
            base_quality = min(max(0.35 + max(r2, 0.0) * 0.5, 0.35), 0.78)
            if overfit_gap > 1_000_000:
                base_quality -= 0.04
                uncertainty_reasons.append("high_overfit_gap")

        if sentiment_mode not in {"tfidf_primary", "transformer"}:
            base_quality -= 0.03
            uncertainty_reasons.append(f"sentiment_mode:{sentiment_mode}")
        if fused["vision"].get("cv_mode") != "resnet50_price_band" and image_score > 0:
            base_quality -= 0.03
            uncertainty_reasons.append(f"cv_mode:{fused['vision'].get('cv_mode')}")
        if not comparables:
            uncertainty_reasons.append("no_comparables")
        if completeness < 0.8:
            uncertainty_reasons.append("low_input_completeness")

        combined = (
            (base_quality * 0.35)
            + (completeness * 0.25)
            + (image_score * 0.15)
            + (text_score * 0.1)
            + (comparable_score * 0.15)
        )
        if ood_flags:
            combined -= min(0.1, 0.03 * len(ood_flags))

        confidence = int(round(max(30.0, min(92.0, combined * 100.0))))
        if confidence >= 80:
            level = "High"
        elif confidence >= 65:
            level = "Medium"
        else:
            level = "Low"

        uncertainty_mode = "calibrated" if model_handle and model_handle.bundle_available else "fallback"
        uncertainty_ratio = max(0.08, min(0.3, 0.34 - (confidence / 500.0)))
        if uncertainty_mode == "fallback":
            uncertainty_ratio = max(uncertainty_ratio, 0.14)

        lower = int(round(estimated_price * (1 - uncertainty_ratio)))
        upper = int(round(estimated_price * (1 + uncertainty_ratio)))
        return {
            "confidence": confidence,
            "confidence_level": level,
            "lower_bound": lower,
            "upper_bound": upper,
            "uncertainty_mode": uncertainty_mode,
            "uncertainty_reasons": sorted(set(uncertainty_reasons)),
        }
