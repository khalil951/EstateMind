from src.nlp.nlp_explainability import NLPExplainabilityService


def test_nlp_explainability_returns_token_contributions() -> None:
    service = NLPExplainabilityService()
    description = "Bright renovated sea view apartment with parking and modern finishes"
    description_analysis = {"description_score": 0.74, "key_phrases": ["bright", "renovated"]}
    description_sentiment = {"description_sentiment": 0.8}
    location_sentiment = {"sentiment": 0.64, "sentiment_label": "positive"}
    comparables = [{"address": "La Marsa, Tunis"}, {"address": "Mutuelleville, Tunis"}]

    result = service.analyze(
        description=description,
        description_analysis=description_analysis,
        description_sentiment=description_sentiment,
        location_sentiment=location_sentiment,
        comparables=comparables,
    )

    assert result["description_sentiment_tokens"]
    assert result["description_quality_tokens"]
    assert result["marketing_quality_tokens"]
    assert result["location_comparison"]["location_sentiment_label"] == "positive"
    assert "similar_locations" in result["location_comparison"]
