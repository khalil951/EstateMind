"""Narrative explanation generation for valuation results."""

from __future__ import annotations

from typing import Any

from src.inference.model_registry import ModelHandle


class ExplanationService:
    """Build a concise natural-language explanation for the final estimate."""

    def build(
        self,
        estimated_price: int,
        confidence: dict[str, Any],
        comparables: list[dict[str, Any]],
        features_impact: list[dict[str, Any]],
        model_handle: ModelHandle | None,
        fused: dict[str, Any],
        *,
        prediction_mode: str,
        explanation_mode: str,
        warnings: list[str] | None = None,
    ) -> str:
        lines: list[str] = [
            f"The estimated value is {estimated_price:,} TND with {confidence['confidence_level'].lower()} confidence.",
        ]
        if prediction_mode == "model":
            scope = "property-specific" if model_handle and model_handle.scope == "by_type" else "global"
            lines.append(f"The valuation used the {scope} serving model as the primary estimator.")
        else:
            lines.append("The valuation is running in heuristic fallback mode because a compatible serving bundle was unavailable.")

        if features_impact:
            top = features_impact[0]
            direction = "increased" if int(top["amount"]) >= 0 else "reduced"
            lines.append(f"The strongest driver that {direction} the estimate was {top['feature'].lower()}.")
        if comparables:
            lines.append(f"The estimate is grounded against {len(comparables)} comparable listing(s) from the local dataset.")
        if explanation_mode != "true_shap":
            lines.append("Feature attribution is in fallback mode, so the SHAP waterfall should be read as an approximate explanation.")
        if fused["summary"].get("sentiment_mode") not in {"tfidf_primary", "transformer"}:
            lines.append("Description sentiment used a fallback runtime, which slightly lowers certainty.")
        if fused["vision"].get("cv_mode") == "resnet50_price_band":
            band = fused["vision"].get("dominant_price_band", "")
            if band:
                lines.append(f"The image model placed the listing photos in the {str(band).replace('_', ' ')} bucket.")
        elif fused["vision"].get("quality", {}).get("image_count", 0) == 0:
            lines.append("No property photos were available, so visual evidence did not strengthen the estimate.")

        reason_strings = confidence.get("uncertainty_reasons", [])
        if reason_strings:
            lines.append("Main uncertainty sources: " + ", ".join(sorted(set(reason_strings))[:4]) + ".")
        extra_warnings = [item for item in (warnings or []) if item]
        if extra_warnings:
            lines.append("Runtime warnings: " + ", ".join(sorted(set(extra_warnings))[:4]) + ".")
        return " ".join(lines)
