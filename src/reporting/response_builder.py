"""Helpers for assembling the final API response payload.

The response builder keeps the frontend contract centralized in one place so
the orchestration layer can focus on producing intermediate artifacts. It
also makes the output format easier to test independently from the rest of
the serving pipeline.
"""

from __future__ import annotations

from typing import Any


def build_response(
    estimated_price: int,
    price_per_m2: int,
    confidence: dict[str, Any],
    features_impact: list[dict[str, Any]],
    comparables: list[dict[str, Any]],
    ai_explanation: str,
    image_analysis: list[str],
    text_analysis: dict[str, Any],
    market_context: dict[str, Any],
    shap: list[dict[str, Any]],
    prediction_mode: str,
    explanation_mode: str,
    sentiment_mode: str,
    cv_mode: str,
    warnings: list[str],
    model_info: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the normalized response structure expected by the API/UI."""

    return {
        "estimated_price": int(estimated_price),
        "lower_bound": int(confidence["lower_bound"]),
        "upper_bound": int(confidence["upper_bound"]),
        "price_per_m2": int(price_per_m2),
        "confidence": int(confidence["confidence"]),
        "confidence_level": str(confidence["confidence_level"]),
        "features_impact": features_impact,
        "comparables": comparables,
        "ai_explanation": ai_explanation,
        "image_analysis": image_analysis,
        "text_analysis": text_analysis,
        "market_context": market_context,
        "shap": shap,
        "prediction_mode": prediction_mode,
        "explanation_mode": explanation_mode,
        "sentiment_mode": sentiment_mode,
        "cv_mode": cv_mode,
        "warnings": warnings,
        "uncertainty_reasons": confidence.get("uncertainty_reasons", []),
        "uncertainty_mode": confidence.get("uncertainty_mode", "fallback"),
        "model_info": model_info,
    }
