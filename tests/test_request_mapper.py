from types import SimpleNamespace

from src.inference.request_mapper import map_request, to_feature_frame


def test_map_request_builds_expected_fields() -> None:
    payload = SimpleNamespace(
        property_type="Maison",
        governorate="Sousse",
        city="Sousse",
        neighborhood="Khzema",
        size_m2=180,
        bedrooms=4,
        bathrooms=2,
        condition="Good",
        has_pool=True,
        has_garden=False,
        has_parking=True,
        sea_view=False,
        elevator=False,
        description="Maison familiale proche commodites avec piscine.",
        uploaded_images_count=3,
        image_refs=["a.jpg", "b.jpg"],
    )
    mapped = map_request(payload)
    assert mapped["surface_m2"] == 180.0
    assert mapped["uploaded_images_count"] == 3
    assert mapped["image_refs"] == ["a.jpg", "b.jpg"]
    assert mapped["transaction_type"] == "sale"
    assert mapped["rooms"] == 5
    assert 0.0 < mapped["input_completeness"] <= 1.0


def test_to_feature_frame_is_single_row() -> None:
    mapped = {
        "property_type": "Terrain",
        "governorate": "Ariana",
        "city": "Ariana",
        "surface_m2": 500.0,
        "bedrooms": 0,
        "bathrooms": 0,
        "condition": "New",
        "has_pool": False,
        "has_garden": False,
        "has_parking": False,
        "sea_view": False,
        "elevator": False,
        "description": "Terrain urbain",
        "uploaded_images_count": 1,
    }
    frame = to_feature_frame(mapped)
    assert frame.shape == (1, 16)
    assert float(frame.iloc[0]["surface_m2"]) == 500.0
