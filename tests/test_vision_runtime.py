from pathlib import Path

from PIL import Image

from src.vision.feature_aggregation import VisionFeatureAggregationService
from src.vision.image_quality import ImageQualityService
from src.vision.type_classifier import ImageTypeClassifierService


class _StubPrimary:
    def classify_path(self, image_path: str):
        return {
            "image_ref": image_path,
            "top_prediction": {"id": 1, "label": "maison", "score": 0.87},
            "predictions": [{"id": 1, "label": "maison", "score": 0.87}],
        }


class _StubClip:
    def classify_image_path(self, image_path: str, top_k: int = 8):
        return {
            "image_ref": image_path,
            "top_prediction": {"label": "property_type_appartement", "score": 0.78},
            "predictions": [{"label": "property_type_appartement", "score": 0.78}],
        }


def test_image_quality_scores_real_file() -> None:
    image_path = Path("artifacts/test_assets/test_quality_img.png")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (128, 128), color=(180, 190, 200)).save(image_path)
    service = ImageQualityService()
    result = service.score([str(image_path)], 1)
    assert result["image_count"] == 1
    assert result["quality_score"] > 0.0


def test_clip_primary_mode_is_exposed() -> None:
    image_path = Path("artifacts/test_assets/test_primary_cv_img.png")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (128, 128), color=(200, 150, 150)).save(image_path)
    service = ImageTypeClassifierService()
    service._clip_classifier = _StubClip()
    rows = service.classify_many([str(image_path)])
    assert rows
    assert rows[0]["cv_mode"] == "clip_feature_inference"


def test_notebook_fallback_mode_is_exposed(monkeypatch) -> None:
    image_path = Path("artifacts/test_assets/test_fallback_cv_img.png")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (128, 128), color=(120, 130, 140)).save(image_path)
    service = ImageTypeClassifierService()
    monkeypatch.setattr(service, "_ensure_clip", lambda: None)
    service._fallback = _StubPrimary()
    rows = service.classify_many([str(image_path)])
    assert rows
    assert rows[0]["cv_mode"] == "notebook_property_type_fallback"


def test_vision_aggregation_infers_property_type_from_clip_tags() -> None:
    aggregator = VisionFeatureAggregationService()
    result = aggregator.aggregate(
        image_rows=[{"top_prediction": {"label": "property_type_maison", "score": 0.9}, "cv_mode": "clip_feature_inference"}],
        quality={"image_count": 1, "quality_score": 0.8, "coverage_score": 0.5},
        mapped={"property_type": "auto", "has_pool": False, "has_garden": False, "has_parking": False},
    )
    assert result["cv_mode"] == "clip_feature_inference"
    assert result["auto_filled_property_type"] == "Maison"
