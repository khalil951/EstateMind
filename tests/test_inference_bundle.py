import pandas as pd

from src.inference.inference_bundle import _ServingProcessor


def test_serving_processor_emits_governorate_prior_column() -> None:
    reference_df = pd.DataFrame(
        [
            {
                "transaction_type": "sale",
                "property_type": "appartement",
                "price_tnd": 200000,
                "surface_m2": 100,
                "rooms": 4,
                "bedrooms": 3,
                "bathrooms": 2,
                "governorate": "tunis",
                "city": "la marsa",
                "latitude": 36.88,
                "longitude": 10.33,
            },
            {
                "transaction_type": "sale",
                "property_type": "appartement",
                "price_tnd": 260000,
                "surface_m2": 120,
                "rooms": 4,
                "bedrooms": 3,
                "bathrooms": 2,
                "governorate": "tunis",
                "city": "la marsa",
                "latitude": 36.87,
                "longitude": 10.31,
            },
        ]
    )

    processor = _ServingProcessor(reference_df)
    transformed, warnings, _ = processor.transform_request(
        {
            "transaction_type": "sale",
            "property_type": "appartement",
            "price_tnd": 240000,
            "surface_m2": 110,
            "price_per_m2": 2181,
            "rooms": 4,
            "bedrooms": 3,
            "bathrooms": 2,
            "governorate": "tunis",
            "city": "la marsa",
            "latitude": 36.875,
            "longitude": 10.32,
        }
    )

    assert "gov_avg_price_m2" in transformed.columns
    assert transformed["gov_avg_price_m2"].notna().all()
    assert "gov_price_prior_fallback" not in warnings
