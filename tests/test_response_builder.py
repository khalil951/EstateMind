from src.reporting.response_builder import build_response


def test_response_builder_preserves_contract() -> None:
    payload = build_response(
        estimated_price=450000,
        price_per_m2=3750,
        confidence={
            "confidence": 74,
            "confidence_level": "Medium",
            "lower_bound": 405000,
            "upper_bound": 495000,
            "uncertainty_reasons": ["proxy_price_features_used"],
            "uncertainty_mode": "fallback",
        },
        features_impact=[{"feature": "Condition", "pct": 5, "amount": 20000}],
        comparables=[{"address": "La Marsa, Tunis", "price": 440000, "size": 118, "transaction_type": "sale", "similarity": 93, "difference": "2 m2 smaller"}],
        ai_explanation="Sample explanation.",
        image_analysis=["2 image(s) received for visual analysis."],
        text_analysis={
            "description_quality": "Good (7.2/10)",
            "description_sentiment": 0.5,
            "description_sentiment_label": "neutral",
            "location_sentiment": 0.6,
            "location_sentiment_label": "positive",
            "marketing_effectiveness": "Medium",
            "key_phrases": ["vue", "mer"],
        },
        market_context={"avg_m2": 3600, "property_m2": 3750, "delta_pct": 4, "trend": "Unavailable", "demand": "10 comparable listing(s) in scope", "city": "La Marsa"},
        shap=[{"feature": "Baseline", "value": 400000}, {"feature": "Final", "value": 450000}],
        prediction_mode="model",
        explanation_mode="true_shap",
        sentiment_mode="transformer",
        cv_mode="resnet50_price_band",
        warnings=["proxy_price_features_used"],
        model_info={"model_name": "catboost"},
    )
    assert payload["estimated_price"] == 450000
    assert payload["confidence_level"] == "Medium"
    assert payload["comparables"][0]["similarity"] == 93
    assert payload["prediction_mode"] == "model"
    assert payload["warnings"] == ["proxy_price_features_used"]
