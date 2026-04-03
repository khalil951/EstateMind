from src.explainability.confidence_service import ConfidenceService


def test_confidence_service_rewards_richer_evidence() -> None:
    service = ConfidenceService()
    sparse = service.estimate(
        estimated_price=300000,
        fused={
            "summary": {"input_completeness": 0.5, "sentiment_mode": "neutral_fallback"},
            "vision": {"quality": {"coverage_score": 0.0}, "cv_mode": "no_images"},
            "nlp": {"description_score": 0.2},
        },
        comparables=[],
        model_handle=None,
        prediction_mode="heuristic",
        warnings=["heuristic_prediction_mode"],
        ood_flags=["unknown_city"],
    )
    rich = service.estimate(
        estimated_price=300000,
        fused={
            "summary": {"input_completeness": 1.0, "sentiment_mode": "transformer"},
            "vision": {"quality": {"coverage_score": 1.0}, "cv_mode": "resnet50_price_band"},
            "nlp": {"description_score": 0.9},
        },
        comparables=[{"similarity": 90}] * 4,
        model_handle=None,
        prediction_mode="model",
        warnings=[],
        ood_flags=[],
    )
    assert rich["confidence"] > sparse["confidence"]
    assert (rich["upper_bound"] - rich["lower_bound"]) < (sparse["upper_bound"] - sparse["lower_bound"])
    assert "heuristic_prediction_mode" in sparse["uncertainty_reasons"]
