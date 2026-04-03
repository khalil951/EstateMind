"""High-level valuation orchestration for EstateMind inference."""

from __future__ import annotations

from typing import Any

from src.explainability.comparables_service import ComparablesService
from src.explainability.confidence_service import ConfidenceService
from src.explainability.explanation_service import ExplanationService
from src.explainability.shap_service import ShapService
from src.inference.feature_fusion import fuse_features
from src.inference.fallback_model import FallbackTabularModelService
from src.inference.model_registry import ModelRegistry
from src.inference.request_mapper import map_request
from src.nlp.description_analysis import DescriptionAnalysisService
from src.nlp.location_sentiment import LocationSentimentService
from src.nlp.sentiment_service import DescriptionSentimentService
from src.reporting.response_builder import build_response
from src.vision.feature_aggregation import VisionFeatureAggregationService
from src.vision.image_quality import ImageQualityService
from src.vision.type_classifier import ImageTypeClassifierService


class ValuationService:
    """Coordinate the end-to-end valuation workflow for one request."""

    def __init__(self) -> None:
        self.registry = ModelRegistry()
        self.image_type = ImageTypeClassifierService()
        self.image_quality = ImageQualityService()
        self.vision_agg = VisionFeatureAggregationService()
        self.fallback_model = FallbackTabularModelService()
        self.description_analysis = DescriptionAnalysisService()
        self.description_sentiment = DescriptionSentimentService()
        self.location_sentiment = LocationSentimentService()
        self.comparables = ComparablesService()
        self.confidence = ConfidenceService()
        self.shap = ShapService()
        self.explanation = ExplanationService()

    def _heuristic_estimate(self, mapped: dict[str, Any], market_context: dict[str, Any]) -> tuple[int, int, list[str]]:
        base_per_m2 = max(int(market_context.get("avg_m2", 1450)), 1)
        type_factor = {"Terrain": 0.82, "Maison": 1.0, "Appartement": 1.08}.get(mapped["property_type"], 1.0)
        condition_factor = {
            "New": 1.16,
            "Excellent": 1.1,
            "Good": 1.0,
            "Fair": 0.92,
            "Needs Renovation": 0.8,
        }.get(mapped["condition"], 1.0)
        amenity_factor = 1.0
        amenity_factor += 0.06 if mapped["sea_view"] else 0.0
        amenity_factor += 0.04 if mapped["has_pool"] else 0.0
        amenity_factor += 0.03 if mapped["has_garden"] else 0.0
        amenity_factor += 0.02 if mapped["has_parking"] else 0.0
        amenity_factor += 0.015 if mapped["elevator"] else 0.0
        room_factor = 1.0 + min((mapped["bedrooms"] * 0.01) + (mapped["bathrooms"] * 0.008), 0.06)
        estimated_per_m2 = int(round(base_per_m2 * type_factor * condition_factor * amenity_factor * room_factor))
        estimated_price = int(round(estimated_per_m2 * mapped["surface_m2"]))
        return estimated_price, estimated_per_m2, ["heuristic_prediction_mode"]

    @staticmethod
    def _refine_with_multimodal_signals(
        estimated_price: int,
        price_per_m2: int,
        vision: dict[str, Any],
        sentiment: dict[str, Any],
    ) -> tuple[int, int]:
        cv_signal = float(vision.get("price_band_effect", 0.0)) * float(vision.get("price_band_confidence", 0.0)) * 0.04
        sentiment_signal = (float(sentiment.get("description_sentiment", 0.5)) - 0.5) * 0.03
        factor = max(0.92, min(1.08, 1.0 + cv_signal + sentiment_signal))
        refined_price = int(round(estimated_price * factor))
        refined_ppm = int(round(price_per_m2 * factor))
        return max(refined_price, 1), max(refined_ppm, 1)

    @staticmethod
    def _apply_cv_autofill(mapped: dict[str, Any], vision: dict[str, Any]) -> None:
        """Fill missing structured inputs from CLIP inference when applicable."""

        auto_property = str(vision.get("auto_filled_property_type", "")).strip()
        if auto_property and str(mapped.get("property_type", "")).strip().lower() in {"", "unknown", "auto"}:
            mapped["property_type"] = auto_property

        auto_amenities = dict(vision.get("auto_filled_amenities") or {})
        for field in ("has_pool", "has_garden", "has_parking", "sea_view", "elevator"):
            if field in auto_amenities and not bool(mapped.get(field)):
                mapped[field] = bool(auto_amenities[field])

    def estimate(self, payload: Any, external_warnings: list[str] | None = None) -> dict[str, Any]:
        mapped = map_request(payload)
        base_warnings = list(external_warnings or [])

        image_rows = self.image_type.classify_many(mapped["image_refs"])
        quality = self.image_quality.score(mapped["image_refs"], mapped["uploaded_images_count"])
        vision = self.vision_agg.aggregate(image_rows, quality, mapped)
        self._apply_cv_autofill(mapped, vision)

        description = self.description_analysis.analyze(mapped["description"])
        description_sentiment = self.description_sentiment.analyze(mapped["description"])
        location_sentiment = self.location_sentiment.analyze(mapped["city"], mapped["neighborhood"])
        fused = fuse_features(mapped, vision, description, description_sentiment, location_sentiment)

        comparables, market_context = self.comparables.find(mapped)
        handle = self.registry.maybe_load_bundle(self.registry.get_property_handle(mapped["property_type"]))
        property_handle_unavailable = handle is not None and not handle.bundle_available
        allow_global_for_type = str(mapped.get("property_type", "")).strip().lower() != "terrain"
        if allow_global_for_type and (handle is None or not handle.bundle_available) and (handle is None or handle.scope != "global"):
            global_handle = self.registry.maybe_load_bundle(self.registry.get_global_handle())
            if global_handle is not None and global_handle.bundle_available:
                handle = global_handle
        prediction_warnings: list[str] = []
        if property_handle_unavailable and handle is not None and handle.scope == "global":
            prediction_warnings.append("property_specific_bundle_unavailable")
        feature_frame = None

        if handle is not None and handle.bundle_available and handle.bundle is not None:
            try:
                prediction = handle.bundle.predict(mapped, market_context)
                estimated_price = prediction.estimated_price
                price_per_m2 = prediction.price_per_m2
                prediction_mode = prediction.prediction_mode
                prediction_warnings.extend(prediction.warnings)
                prediction_warnings.extend(prediction.uncertainty_reasons)
                feature_frame = prediction.feature_frame
                ood_flags = prediction.ood_flags
                model_info = prediction.model_info
            except Exception as exc:
                fallback_prediction = self.fallback_model.predict(mapped)
                if fallback_prediction is not None:
                    estimated_price = fallback_prediction.estimated_price
                    price_per_m2 = fallback_prediction.price_per_m2
                    prediction_mode = fallback_prediction.prediction_mode
                    prediction_warnings.extend(fallback_prediction.warnings)
                    prediction_warnings.extend(fallback_prediction.uncertainty_reasons)
                    prediction_warnings.append(f"bundle_predict_failed:{exc.__class__.__name__}")
                    ood_flags = []
                    model_info = fallback_prediction.model_info
                else:
                    estimated_price, price_per_m2, fallback_warnings = self._heuristic_estimate(mapped, market_context)
                    prediction_mode = "heuristic"
                    prediction_warnings.extend(fallback_warnings)
                    prediction_warnings.append(f"bundle_predict_failed:{exc.__class__.__name__}")
                    ood_flags = []
                    model_info = {"bundle_error": str(exc)}
        else:
            fallback_prediction = self.fallback_model.predict(mapped)
            if fallback_prediction is not None:
                estimated_price = fallback_prediction.estimated_price
                price_per_m2 = fallback_prediction.price_per_m2
                prediction_mode = fallback_prediction.prediction_mode
                prediction_warnings.extend(fallback_prediction.warnings)
                prediction_warnings.extend(fallback_prediction.uncertainty_reasons)
                if handle is not None and handle.bundle_error:
                    prediction_warnings.append(handle.bundle_error)
                ood_flags = []
                model_info = fallback_prediction.model_info
            else:
                estimated_price, price_per_m2, fallback_warnings = self._heuristic_estimate(mapped, market_context)
                prediction_mode = "heuristic"
                prediction_warnings.extend(fallback_warnings)
                if handle is not None and handle.bundle_error:
                    prediction_warnings.append(handle.bundle_error)
                ood_flags = []
                model_info = {"bundle_error": handle.bundle_error if handle else "no_model_handle"}

        estimated_price, price_per_m2 = self._refine_with_multimodal_signals(
            estimated_price,
            price_per_m2,
            vision,
            description_sentiment,
        )
        market_context["property_m2"] = price_per_m2
        market_context["delta_pct"] = round(((price_per_m2 / max(int(market_context["avg_m2"]), 1)) - 1) * 100)

        confidence = self.confidence.estimate(
            estimated_price,
            fused,
            comparables,
            handle,
            prediction_mode=prediction_mode,
            warnings=base_warnings + prediction_warnings + vision.get("warnings", []),
            ood_flags=ood_flags,
        )
        shap, features_impact, explanation_mode, shap_warnings = self.shap.explain(
            estimated_price,
            fused,
            handle,
            feature_frame=feature_frame,
        )

        all_warnings = sorted(set(base_warnings + prediction_warnings + vision.get("warnings", []) + shap_warnings))
        ai_explanation = self.explanation.build(
            estimated_price=estimated_price,
            confidence=confidence,
            comparables=comparables,
            features_impact=features_impact,
            model_handle=handle,
            fused=fused,
            prediction_mode=prediction_mode,
            explanation_mode=explanation_mode,
            warnings=all_warnings,
        )
        text_analysis = {
            "description_quality": description["description_quality"],
            "description_sentiment": description_sentiment["description_sentiment"],
            "description_sentiment_label": description_sentiment["description_sentiment_label"],
            "location_sentiment": location_sentiment["sentiment"],
            "location_sentiment_label": location_sentiment["sentiment_label"],
            "marketing_effectiveness": description["marketing_effectiveness"],
            "key_phrases": description["key_phrases"],
        }

        return build_response(
            estimated_price=estimated_price,
            price_per_m2=price_per_m2,
            confidence=confidence,
            features_impact=features_impact,
            comparables=comparables,
            ai_explanation=ai_explanation,
            image_analysis=vision["image_analysis"],
            text_analysis=text_analysis,
            market_context=market_context,
            shap=shap,
            prediction_mode=prediction_mode,
            explanation_mode=explanation_mode,
            sentiment_mode=description_sentiment["sentiment_mode"],
            cv_mode=vision.get("cv_mode", "no_images"),
            warnings=all_warnings,
            model_info=model_info,
        )
