"""Feature fusion helpers for the valuation orchestration layer.

The serving pipeline produces signals from several domains:

- structured request data
- vision-derived evidence
- description and location NLP outputs

This module combines them into a single nested payload that downstream
services can inspect without needing to know which subsystem produced each
signal.
"""

from __future__ import annotations

from typing import Any


def fuse_features(
    mapped: dict[str, Any],
    vision: dict[str, Any],
    description: dict[str, Any],
    sentiment: dict[str, Any],
    location_sentiment: dict[str, Any],
) -> dict[str, Any]:
    """Combine structured, vision, and NLP signals into a fused payload."""

    return {
        "structured": mapped,
        "vision": vision,
        "nlp": {
            **description,
            **sentiment,
            **location_sentiment,
        },
        "summary": {
            "input_completeness": mapped.get("input_completeness", 0.0),
            "image_count": vision.get("quality", {}).get("image_count", 0),
            "description_score": description.get("description_score", 0.0),
            "location_sentiment": location_sentiment.get("sentiment", 0.5),
            "description_sentiment": sentiment.get("description_sentiment", 0.5),
            "sentiment_mode": sentiment.get("sentiment_mode", "neutral_fallback"),
            "cv_mode": vision.get("cv_mode", "no_images"),
        },
    }
