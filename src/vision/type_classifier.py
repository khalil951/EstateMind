"""Primary and fallback image classification services for EstateMind.

The primary runtime uses the CLIP zero-shot classifier for semantic property
and amenity tagging. When CLIP runtime cannot load, the service falls back to
the notebook-trained CNN classifier exported from the CV notebook.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from PIL import Image
    from torchvision.models import ResNet50_Weights, resnet50
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    torch = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]
    ResNet50_Weights = None  # type: ignore[assignment]
    resnet50 = None  # type: ignore[assignment]

try:
    from src.image_type_classifier import CLIPImageTypeClassifier
except ModuleNotFoundError:  # pragma: no cover
    CLIPImageTypeClassifier = None  # type: ignore[assignment]


class _NotebookFallbackClassifier:
    """Load the notebook-exported fallback checkpoint and map logits to classes."""

    def __init__(self, checkpoint_path: Path, labels_path: Path) -> None:
        if torch is None or Image is None or resnet50 is None or ResNet50_Weights is None:
            raise ModuleNotFoundError("torch, torchvision, and pillow are required for the primary CV runtime")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"CV checkpoint not found: {checkpoint_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"CV label manifest not found: {labels_path}")

        payload = json.loads(labels_path.read_text(encoding="utf-8"))
        classes = payload.get("classes", [])
        if not isinstance(classes, list) or not classes:
            raise ValueError("CV label manifest must contain a non-empty 'classes' list")
        self.classes = classes
        num_classes = len(classes)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
        if not isinstance(state_dict, dict):
            raise ValueError("CV checkpoint does not expose a 'state_dict'")

        model = resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = torch.nn.Sequential(torch.nn.Dropout(p=0.2), torch.nn.Linear(in_features, num_classes))  # type: ignore[call-overload]
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        self.model = model
        self.transforms = ResNet50_Weights.IMAGENET1K_V2.transforms()

    def classify_path(self, image_path: str) -> dict[str, Any]:
        with Image.open(image_path) as raw: #type: ignore 
            image = raw.convert("RGB")
        tensor = self.transforms(image).unsqueeze(0)
        with torch.no_grad():#type: ignore 
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()#type: ignore 
        top_idx = int(np.argmax(probs))
        top = self.classes[top_idx]
        predictions: list[dict[str, Any]] = []
        for idx, score in enumerate(probs.tolist()):
            item = dict(self.classes[idx])
            item["score"] = float(score)
            predictions.append(item)
        predictions.sort(key=lambda item: float(item["score"]), reverse=True)
        return {
            "image_ref": image_path,
            "model_family": "notebook_cnn_fallback",
            "top_prediction": predictions[0],
            "predictions": predictions,
        }


class ImageTypeClassifierService:
    """Primary CLIP classifier with notebook CNN fallback for resilience."""

    def __init__(
        self,
        fallback_checkpoint_path: str | Path | None = None,
        fallback_labels_path: str | Path | None = None,
    ) -> None:
        root = Path(__file__).resolve().parents[2]
        self.fallback_checkpoint_path = (
            Path(fallback_checkpoint_path)
            if fallback_checkpoint_path
            else root / "artifacts" / "models" / "image_property_type_fallback.pt"
        )
        self.fallback_labels_path = (
            Path(fallback_labels_path)
            if fallback_labels_path
            else root / "artifacts" / "models" / "image_property_type_fallback.labels.json"
        )
        # Keep backward compatibility with older notebook export naming.
        if not self.fallback_checkpoint_path.exists():
            legacy_checkpoint = root / "artifacts" / "models" / "best_model.pt"
            if legacy_checkpoint.exists():
                self.fallback_checkpoint_path = legacy_checkpoint
        if not self.fallback_labels_path.exists():
            legacy_labels = root / "artifacts" / "models" / "best_model.labels.json"
            if legacy_labels.exists():
                self.fallback_labels_path = legacy_labels

        self._fallback: _NotebookFallbackClassifier | None = None
        self._fallback_error: str | None = None
        self._clip_classifier: Any | None = None

    def _ensure_fallback(self) -> _NotebookFallbackClassifier | None:
        if self._fallback is not None or self._fallback_error is not None:
            return self._fallback
        try:
            self._fallback = _NotebookFallbackClassifier(self.fallback_checkpoint_path, self.fallback_labels_path)
        except Exception as exc:
            self._fallback_error = str(exc)
        return self._fallback

    def _ensure_clip(self) -> Any | None:
        if self._clip_classifier is not None:
            return self._clip_classifier
        if CLIPImageTypeClassifier is None:
            return None
        try:
            self._clip_classifier = CLIPImageTypeClassifier()
        except Exception:
            return None
        return self._clip_classifier

    def classify_many(self, image_refs: list[str]) -> list[dict[str, Any]]:
        """Classify up to five images with CLIP or notebook fallback."""

        if not image_refs:
            return []

        rows: list[dict[str, Any]] = []
        clip_classifier = self._ensure_clip()
        if clip_classifier is not None:
            for ref in image_refs[:5]:
                try:
                    row = clip_classifier.classify_image_path(ref, top_k=8)
                    row["cv_mode"] = "clip_feature_inference"
                    rows.append(row)
                except Exception:
                    continue
            if rows:
                return rows

        fallback = self._ensure_fallback()
        if fallback is None:
            return []
        for ref in image_refs[:5]:
            try:
                row = fallback.classify_path(ref)
                row["cv_mode"] = "notebook_property_type_fallback"
                rows.append(row)
            except Exception:
                continue
        return rows

    @property
    def primary_error(self) -> str | None:
        """Expose CLIP/fallback runtime load errors for warnings/reporting."""

        if self._ensure_clip() is not None:
            return None
        self._ensure_fallback()
        return self._fallback_error or "clip_runtime_unavailable"
