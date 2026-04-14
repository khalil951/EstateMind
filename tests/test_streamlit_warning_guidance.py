from streamlit_app import warning_explanation


def test_warning_explanation_has_enriched_geo_and_prior_guidance() -> None:
    assert "coordinates" in warning_explanation("geo_lookup_missing").lower()
    assert "governorate" in warning_explanation("local_price_prior_fallback_governorate").lower()
    assert "coverage" in warning_explanation("ood:local_price_prior_city_data_coverage").lower()
    assert "city" in warning_explanation("ood:unknown_city_data_coverage").lower()
