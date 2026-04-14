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


def _prompt_item(code: str, severity: str, field: str, message: str, **extra: Any) -> dict[str, Any]:
    item = {
        "code": code,
        "severity": severity,
        "field": field,
        "message": message,
    }
    item.update(extra)
    return item


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
        clip_amenity_labels: set[str] = set()
        clip_predictions_available = False
        for row in image_rows:
            top = row.get("top_prediction") or {}
            label = str(top.get("label", "")).strip()
            if label:
                labels.append(label)
                confidences.append(float(top.get("score", 0.0)))
            mode = str(row.get("cv_mode", "")).strip()
            if mode:
                cv_modes.append(mode)
            clip_predictions = list(row.get("clip_predictions") or [])
            if clip_predictions:
                clip_predictions_available = True
            for pred in clip_predictions:
                clip_label = str(pred.get("label", "")).strip()
                clip_score = float(pred.get("score", 0.0) or 0.0)
                if clip_label in AMENITY_LABEL_MAP and clip_score >= 0.08:
                    clip_amenity_labels.add(clip_label)

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
        for label in clip_amenity_labels:
            amenity = AMENITY_LABEL_MAP.get(label)
            if amenity:
                inferred_amenities[amenity] = True
        clip_detected_amenities = sorted({AMENITY_LABEL_MAP[label] for label in clip_amenity_labels if label in AMENITY_LABEL_MAP})

        input_property = str(mapped.get("property_type", "")).strip()
        property_unfilled = input_property.lower() in {"", "unknown", "auto"}
        auto_filled_property = inferred_property_type if property_unfilled and inferred_property_type else ""
        effective_property_type = auto_filled_property or input_property
        terrain_effective = effective_property_type.lower() == "terrain"

        guidance: list[dict[str, Any]] = []
        requires_confirmation = False
        if image_count > 0:
            guidance.append(
                _prompt_item(
                    "image_guidance_active",
                    "info",
                    "images",
                    "Image-based validation and autofill checks are active for this request.",
                    action="review_vision_guidance",
                )
            )
            guidance.append(
                _prompt_item(
                    "image_anomaly_detection_ran",
                    "info",
                    "images",
                    "Image anomaly checks ran to validate property type and detect missing amenities.",
                    action="review_anomaly_checks",
                )
            )
        if auto_filled_property:
            requires_confirmation = True
            guidance.append(
                _prompt_item(
                    "property_type_autofilled",
                    "warning",
                    "property_type",
                    f"Property type inferred from the image as {auto_filled_property}; please confirm it before continuing.",
                    inferred=auto_filled_property,
                    action="confirm_inferred_property_type",
                )
            )
        elif input_property and inferred_property_type and input_property.title() != inferred_property_type:
            requires_confirmation = True
            guidance.append(
                _prompt_item(
                    "property_type_mismatch",
                    "warning",
                    "property_type",
                    f"Selected property type '{input_property}' conflicts with the image, which looks like {inferred_property_type}.",
                    selected=input_property,
                    inferred=inferred_property_type,
                    action="choose_correct_property_type",
                )
            )

        amenity_keys = ["has_pool", "has_garden", "has_parking", "sea_view", "elevator"]
        if terrain_effective:
            selected_terrain_amenities = [key for key in amenity_keys if bool(mapped.get(key))]
            if selected_terrain_amenities:
                guidance.append(
                    _prompt_item(
                        "terrain_amenities_disabled",
                        "warning",
                        "amenities",
                        "Terrain listings do not support residential amenities; selected amenities will be ignored.",
                        selected=selected_terrain_amenities,
                        action="remove_terrain_amenities",
                    )
                )

        all_amenities_false = not any(bool(mapped.get(key)) for key in amenity_keys)
        auto_filled_amenities = {}
        if not terrain_effective:
            for key in amenity_keys:
                image_says_present = bool(inferred_amenities[key])
                user_selected = bool(mapped.get(key))
                if image_says_present and not user_selected:
                    auto_filled_amenities[key] = True
                    guidance.append(
                        _prompt_item(
                            "amenity_detected_from_image",
                            "info",
                            key,
                            f"The image suggests {key.replace('_', ' ')} is present. Add it if it is part of the listing.",
                            inferred=True,
                            action="add_amenity",
                        )
                    )
                elif user_selected and not image_says_present:
                    guidance.append(
                        _prompt_item(
                            "amenity_not_visible_in_images",
                            "warning",
                            key,
                            f"{key.replace('_', ' ').title()} is selected but not visible in the uploaded photos. Consider adding another image or removing it.",
                            selected=True,
                            inferred=False,
                            action="add_more_images_or_review_selection",
                        )
                    )
        if not terrain_effective and all_amenities_false and auto_filled_amenities:
            guidance.append(
                _prompt_item(
                    "amenity_autofill_summary",
                    "info",
                    "amenities",
                    "The uploaded image suggests additional amenities that can be added to improve the estimate.",
                    inferred=list(sorted(auto_filled_amenities.keys())),
                    action="review_suggested_amenities",
                )
            )

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
                if requires_confirmation:
                    analysis.append("Please confirm the property type because the uploaded photo evidence conflicts with the manual selection.")
            elif primary_mode == "notebook_property_type_primary" and all_detected:
                analysis.append("Notebook model inferred property type from images.")
                if clip_amenity_labels:
                    analysis.append("CLIP amenity tags: " + ", ".join(sorted(clip_amenity_labels)[:6]) + ".")
                elif not clip_predictions_available:
                    warnings.append("clip_amenity_runtime_unavailable")
                if auto_filled_property:
                    analysis.append(f"Property type auto-filled from notebook model: {auto_filled_property}.")
                if auto_filled_amenities:
                    amenity_text = ", ".join(sorted(auto_filled_amenities.keys()))
                    analysis.append(f"Amenities inferred from CLIP image signals: {amenity_text}.")
                if requires_confirmation:
                    analysis.append("Please confirm the property type because the uploaded photo evidence conflicts with the manual selection.")
            elif primary_mode == "clip_property_type_fallback" and all_detected:
                analysis.append("Notebook property classifier was unavailable; CLIP fallback inferred property type.")
                analysis.append("CLIP feature inference tags: " + ", ".join(all_detected[:6]) + ".")
                warnings.append("notebook_property_type_unavailable")
                if auto_filled_property:
                    analysis.append(f"Property type auto-filled from CLIP fallback: {auto_filled_property}.")
                if auto_filled_amenities:
                    amenity_text = ", ".join(sorted(auto_filled_amenities.keys()))
                    analysis.append(f"Amenities inferred from CLIP image signals: {amenity_text}.")
                if requires_confirmation:
                    analysis.append("Please confirm the property type because the uploaded photo evidence conflicts with the manual selection.")
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
            "clip_detected_amenities": clip_detected_amenities,
            "vision_guidance": guidance,
            "requires_user_confirmation": requires_confirmation,
            "image_analysis": analysis,
            "quality": quality,
            "cv_mode": primary_mode,
            "warnings": warnings,
        }
