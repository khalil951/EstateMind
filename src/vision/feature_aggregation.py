"""Aggregate image-level CV outputs into listing-level vision features."""

from __future__ import annotations

from collections import Counter
from typing import Any


PRICE_BAND_EFFECT = {
    "low_price_band": -1.0,
    "mid_price_band": 0.0,
    "high_price_band": 1.0,
}

PROPERTY_HINT_MAP = {
    "terrain": "Terrain",
    "maison": "Maison",
    "appartement": "Appartement",
    "property_type_terrain": "Terrain",
    "land_plot": "Terrain",
    "agricultural_land": "Terrain",
    "property_type_maison": "Maison",
    "villa_exterior": "Maison",
    "house_exterior": "Maison",
    "property_type_appartement": "Appartement",
    "apartment_building_exterior": "Appartement",
    "residential_building_exterior": "Appartement",
}

AMENITY_LABEL_MAP = {
    "amenity_pool": "has_pool",
    "swimming_pool": "has_pool",
    "amenity_garden": "has_garden",
    "garden_or_yard": "has_garden",
    "amenity_parking": "has_parking",
    "garage_or_parking": "has_parking",
    "amenity_sea_view": "sea_view",
    "amenity_elevator": "elevator",
    "elevator": "elevator",
}


class VisionFeatureAggregationService:
    """Merge primary CV outputs and quality metrics into one vision payload."""

    def aggregate(
        self,
        image_rows: list[dict[str, Any]],
        quality: dict[str, Any],
        mapped: dict[str, Any],
    ) -> dict[str, Any]:
        labels: list[str] = []
        cv_modes: list[str] = []
        confidences: list[float] = []
        for row in image_rows:
            top = row.get("top_prediction") or {}
            label = str(top.get("label", "")).strip()
            if label:
                labels.append(label)
                confidences.append(float(top.get("score", 0.0)))
            mode = str(row.get("cv_mode", "")).strip()
            if mode:
                cv_modes.append(mode)

        counts = Counter(labels)
        detected = [label for label, _ in counts.most_common(3)]
        all_detected = [label for label, _ in counts.most_common(8)]
        image_count = int(quality.get("image_count", 0))
        primary_mode = cv_modes[0] if cv_modes else "no_images"
        dominant_band = detected[0] if detected else ""
        band_effect = PRICE_BAND_EFFECT.get(dominant_band, 0.0)
        band_confidence = max(confidences) if confidences else 0.0

        has_pool_evidence = "swimming_pool" in counts or bool(mapped.get("has_pool"))
        has_garden_evidence = "garden_or_yard" in counts or bool(mapped.get("has_garden"))
        has_parking_evidence = "garage_or_parking" in counts or bool(mapped.get("has_parking"))

        inferred_property_type = ""
        for label in all_detected:
            inferred = PROPERTY_HINT_MAP.get(label)
            if inferred:
                inferred_property_type = inferred
                break

        inferred_amenities = {
            "has_pool": False,
            "has_garden": False,
            "has_parking": False,
            "sea_view": False,
            "elevator": False,
        }
        for label in all_detected:
            amenity = AMENITY_LABEL_MAP.get(label)
            if amenity:
                inferred_amenities[amenity] = True

        input_property = str(mapped.get("property_type", "")).strip()
        property_unfilled = input_property.lower() in {"", "unknown", "auto"}
        auto_filled_property = inferred_property_type if property_unfilled and inferred_property_type else ""

        amenity_keys = ["has_pool", "has_garden", "has_parking", "sea_view", "elevator"]
        all_amenities_false = not any(bool(mapped.get(key)) for key in amenity_keys)
        auto_filled_amenities = {
            key: bool(inferred_amenities[key])
            for key in amenity_keys
            if all_amenities_false and bool(inferred_amenities[key])
        }

        analysis: list[str] = []
        warnings: list[str] = []
        if image_count == 0:
            analysis.append("No images were provided, so visual explainability is limited.")
        else:
            analysis.append(f"{image_count} image(s) received for visual analysis.")
            if primary_mode == "clip_feature_inference" and all_detected:
                analysis.append("CLIP feature inference tags: " + ", ".join(all_detected[:6]) + ".")
                if auto_filled_property:
                    analysis.append(f"Property type auto-filled from images: {auto_filled_property}.")
                if auto_filled_amenities:
                    amenity_text = ", ".join(sorted(auto_filled_amenities.keys()))
                    analysis.append(f"Amenities auto-filled from images: {amenity_text}.")
            elif primary_mode == "notebook_property_type_fallback" and all_detected:
                analysis.append("Notebook fallback visual tags: " + ", ".join(all_detected[:6]) + ".")
                warnings.append("clip_runtime_unavailable")
                if auto_filled_property:
                    analysis.append(f"Property type auto-filled from fallback model: {auto_filled_property}.")
                if auto_filled_amenities:
                    amenity_text = ", ".join(sorted(auto_filled_amenities.keys()))
                    analysis.append(f"Amenities auto-filled from fallback model: {amenity_text}.")
            elif primary_mode == "resnet50_price_band" and dominant_band:
                analysis.append(
                    f"Primary CV model mapped the photos to the {dominant_band.replace('_', ' ')} bucket "
                    f"with {int(round(band_confidence * 100))}% confidence."
                )
            else:
                analysis.append("Visual model could not derive a stable label from the uploaded photos.")
            analysis.append(f"Estimated image quality score: {int(float(quality.get('quality_score', 0.0)) * 100)}/100.")

        return {
            "labels": detected,
            "dominant_price_band": dominant_band,
            "price_band_confidence": round(band_confidence, 4),
            "price_band_effect": band_effect,
            "has_pool_evidence": has_pool_evidence,
            "has_garden_evidence": has_garden_evidence,
            "has_parking_evidence": has_parking_evidence,
            "aesthetic_score": round(float(quality.get("quality_score", 0.0)), 3),
            "inferred_property_type": inferred_property_type,
            "inferred_amenities": inferred_amenities,
            "auto_filled_property_type": auto_filled_property,
            "auto_filled_amenities": auto_filled_amenities,
            "image_analysis": analysis,
            "quality": quality,
            "cv_mode": primary_mode,
            "warnings": warnings,
        }
