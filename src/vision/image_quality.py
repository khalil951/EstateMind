"""Image quality heuristics for uploaded property photos."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    from PIL import Image, ImageStat
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    Image = None  # type: ignore[assignment]
    ImageStat = None  # type: ignore[assignment]


class ImageQualityService:
    """Estimate basic quality/coverage from real image files."""

    def _score_one(self, image_path: str) -> float:
        if Image is None or ImageStat is None:
            return 0.5
        with Image.open(Path(image_path)) as raw:
            image = raw.convert("RGB")
            width, height = image.size
            brightness = float(np.mean(ImageStat.Stat(image).mean)) / 255.0
        resolution_score = min((width * height) / (1280 * 720), 1.0)
        brightness_score = 1.0 - min(abs(brightness - 0.55) / 0.55, 1.0)
        return float(max(0.1, min(1.0, (resolution_score * 0.6) + (brightness_score * 0.4))))

    def score(self, image_refs: list[str], uploaded_images_count: int) -> dict[str, Any]:
        """Estimate image coverage and a coarse quality score for a listing."""

        count = max(len(image_refs), int(uploaded_images_count))
        if count <= 0:
            return {
                "image_count": 0,
                "quality_score": 0.0,
                "coverage_score": 0.0,
                "status": "no_images",
            }

        scores: list[float] = []
        for ref in image_refs[:5]:
            try:
                scores.append(self._score_one(ref))
            except Exception:
                continue
        if not scores:
            quality_score = min(0.4 + (count * 0.1), 0.8)
            status = "count_only"
        else:
            quality_score = float(np.mean(scores))
            status = "ok"
        coverage_score = min(count / 4.0, 1.0)
        return {
            "image_count": count,
            "quality_score": round(quality_score, 3),
            "coverage_score": round(coverage_score, 3),
            "status": status,
        }
