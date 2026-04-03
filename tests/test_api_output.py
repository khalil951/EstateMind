from fastapi.testclient import TestClient
from pathlib import Path

from PIL import Image

from src.api import _valuation_service, app


def test_estimate_returns_complete_payload() -> None:
    client = TestClient(app)
    payload = {
        "property_type": "Appartement",
        "governorate": "Tunis",
        "city": "La Marsa",
        "neighborhood": "Sidi Abdelaziz",
        "size_m2": 120,
        "bedrooms": 3,
        "bathrooms": 2,
        "condition": "Excellent",
        "has_pool": False,
        "has_garden": True,
        "has_parking": True,
        "sea_view": True,
        "elevator": True,
        "description": "Appartement renove avec vue mer exceptionnelle, cuisine moderne et emplacement premium a La Marsa.",
        "uploaded_images_count": 2,
    }
    response = client.post("/estimate", json=payload)
    assert response.status_code == 200
    body = response.json()

    expected_fields = {
        "estimated_price",
        "lower_bound",
        "upper_bound",
        "price_per_m2",
        "confidence",
        "confidence_level",
        "features_impact",
        "comparables",
        "ai_explanation",
        "image_analysis",
        "text_analysis",
        "market_context",
        "shap",
        "prediction_mode",
        "explanation_mode",
        "sentiment_mode",
        "cv_mode",
        "warnings",
        "uncertainty_reasons",
        "uncertainty_mode",
        "model_info",
    }
    assert expected_fields.issubset(body.keys())
    assert isinstance(body["features_impact"], list)
    assert isinstance(body["comparables"], list)
    assert isinstance(body["image_analysis"], list)
    assert isinstance(body["text_analysis"], dict)
    assert isinstance(body["market_context"], dict)
    assert isinstance(body["shap"], list)
    assert body["estimated_price"] > 0
    assert body["lower_bound"] < body["upper_bound"]


def test_estimate_upload_accepts_images() -> None:
    client = TestClient(app)
    image_path = Path("artifacts/test_assets/test_upload_sample.png")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=(180, 180, 200)).save(image_path)
    with image_path.open("rb") as handle:
        response = client.post(
            "/estimate-upload",
            data={
                "property_type": "Appartement",
                "governorate": "Tunis",
                "city": "La Marsa",
                "neighborhood": "Sidi Abdelaziz",
                "size_m2": "120",
                "bedrooms": "3",
                "bathrooms": "2",
                "condition": "Excellent",
                "has_pool": "false",
                "has_garden": "true",
                "has_parking": "true",
                "sea_view": "true",
                "elevator": "true",
                "description": "Appartement renove avec vue mer exceptionnelle.",
                "transaction_type": "sale",
            },
            files=[("images", ("sample.png", handle.read(), "image/png"))],
        )
    assert response.status_code == 200
    body = response.json()
    assert body["cv_mode"] in {"clip_feature_inference", "notebook_property_type_fallback", "no_images"}


def test_estimate_upload_adds_clip_property_mismatch_warning(monkeypatch) -> None:
    def _fake_classify_many(image_refs):
        return [
            {
                "cv_mode": "clip_feature_inference",
                "top_prediction": {"label": "villa_exterior", "score": 0.93},
            }
        ]

    monkeypatch.setattr(_valuation_service.image_type, "classify_many", _fake_classify_many)

    client = TestClient(app)
    image_path = Path("artifacts/test_assets/test_upload_mismatch.png")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=(180, 180, 200)).save(image_path)
    with image_path.open("rb") as handle:
        response = client.post(
            "/estimate-upload",
            data={
                "property_type": "Appartement",
                "governorate": "Tunis",
                "city": "La Marsa",
                "neighborhood": "Sidi Abdelaziz",
                "size_m2": "120",
                "bedrooms": "3",
                "bathrooms": "2",
                "condition": "Excellent",
                "has_pool": "false",
                "has_garden": "true",
                "has_parking": "true",
                "sea_view": "true",
                "elevator": "true",
                "description": "Appartement renove avec vue mer exceptionnelle.",
                "transaction_type": "sale",
            },
            files=[("images", ("sample.png", handle.read(), "image/png"))],
        )
    assert response.status_code == 200
    body = response.json()
    assert any(str(item).startswith("clip_property_type_mismatch:") for item in body.get("warnings", []))
