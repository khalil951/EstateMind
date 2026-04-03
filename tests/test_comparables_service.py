from pathlib import Path

import pandas as pd

from src.explainability.comparables_service import ComparablesService


def test_comparables_service_ranks_matches() -> None:
    df = pd.DataFrame(
        [
            {
                "property_type": "Appartement",
                "transaction_type": "sale",
                "governorate": "Tunis",
                "city": "La Marsa",
                "price_tnd": 500000,
                "surface_m2": 120,
                "bedrooms": 3,
                "bathrooms": 2,
                "price_per_m2": 4166.67,
                "listing_date": "2026-01-10",
            },
            {
                "property_type": "Appartement",
                "transaction_type": "sale",
                "governorate": "Tunis",
                "city": "La Marsa",
                "price_tnd": 450000,
                "surface_m2": 110,
                "bedrooms": 3,
                "bathrooms": 2,
                "price_per_m2": 4090.91,
                "listing_date": "2026-02-15",
            },
            {
                "property_type": "Maison",
                "transaction_type": "rent",
                "governorate": "Sousse",
                "city": "Sousse",
                "price_tnd": 800000,
                "surface_m2": 250,
                "bedrooms": 4,
                "bathrooms": 3,
                "price_per_m2": 3200.0,
                "listing_date": "2026-03-01",
            },
        ]
    )
    csv_path = Path("artifacts") / "test_comparables_fixture.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    service = ComparablesService(listings_path=csv_path)
    mapped = {
        "property_type": "Appartement",
        "transaction_type": "sale",
        "governorate": "Tunis",
        "city": "La Marsa",
        "surface_m2": 118.0,
        "bedrooms": 3,
        "bathrooms": 2,
    }
    comparables, market_context = service.find(mapped)
    assert len(comparables) >= 2
    assert comparables[0]["similarity"] >= comparables[-1]["similarity"]
    assert comparables[0]["transaction_type"] == "sale"
    assert market_context["city"] == "La Marsa"
    assert "trend" in market_context
    assert "trend_reason" in market_context
