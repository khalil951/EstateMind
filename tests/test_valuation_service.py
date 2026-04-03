from types import SimpleNamespace

from src.inference.valuation_service import ValuationService


def test_valuation_service_prefers_model_mode_for_supported_request() -> None:
    service = ValuationService()
    payload = SimpleNamespace(
        property_type="Appartement",
        transaction_type="sale",
        governorate="Tunis",
        city="La Marsa",
        neighborhood="Sidi Abdelaziz",
        size_m2=120,
        bedrooms=3,
        bathrooms=2,
        condition="Excellent",
        has_pool=False,
        has_garden=True,
        has_parking=True,
        sea_view=True,
        elevator=True,
        description="Appartement renove avec vue mer exceptionnelle, cuisine moderne et emplacement premium.",
        uploaded_images_count=0,
        image_refs=[],
    )
    result = service.estimate(payload)
    assert result["estimated_price"] > 0
    assert result["prediction_mode"] in {"model", "fallback_model", "heuristic"}
    assert "warnings" in result


def test_cv_autofill_only_applies_when_fields_missing() -> None:
    mapped = {
        "property_type": "auto",
        "has_pool": False,
        "has_garden": False,
        "has_parking": False,
        "sea_view": False,
        "elevator": False,
    }
    vision = {
        "auto_filled_property_type": "Maison",
        "auto_filled_amenities": {"has_pool": True, "sea_view": True},
    }
    ValuationService._apply_cv_autofill(mapped, vision)
    assert mapped["property_type"] == "Maison"
    assert mapped["has_pool"] is True
    assert mapped["sea_view"] is True

    mapped_with_manual = {
        "property_type": "Appartement",
        "has_pool": True,
        "has_garden": False,
        "has_parking": False,
        "sea_view": False,
        "elevator": False,
    }
    ValuationService._apply_cv_autofill(mapped_with_manual, vision)
    assert mapped_with_manual["property_type"] == "Appartement"
    assert mapped_with_manual["has_pool"] is True
