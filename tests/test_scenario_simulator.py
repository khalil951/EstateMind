from src.explainability.scenario_simulator import ScenarioSimulator


def test_scenario_simulator_generates_ranked_recommendations() -> None:
    simulator = ScenarioSimulator()
    mapped = {
        "property_type": "Maison",
        "city": "Tunis",
        "neighborhood": "Marsa",
        "condition": "Good",
        "surface_m2": 120,
        "has_pool": False,
        "has_garden": False,
        "has_parking": False,
        "sea_view": False,
        "elevator": False,
    }
    market_context = {"avg_m2": 1500, "trend": "Stable"}
    features_impact = [{"feature": "Condition", "amount": 12000}]
    comparables = [{"address": "La Marsa, Tunis"}]
    confidence = {"confidence": 78}
    description_analysis = {"description_score": 0.65, "key_phrases": ["bright", "renovated"]}
    description_sentiment = {"description_sentiment": 0.7}
    location_sentiment = {"sentiment": 0.62, "sentiment_label": "positive"}

    scenarios, recommendations = simulator.generate(
        mapped=mapped,
        estimated_price=250000,
        market_context=market_context,
        features_impact=features_impact,
        comparables=comparables,
        confidence=confidence,
        description_analysis=description_analysis,
        description_sentiment=description_sentiment,
        location_sentiment=location_sentiment,
    )

    assert scenarios
    assert recommendations
    assert len(scenarios) <= 5
    assert scenarios[0]["price_delta"] >= scenarios[-1]["price_delta"]
    assert all("scenario_name" in item for item in scenarios)
    assert all("predicted_impact_tnd" in item for item in recommendations)
