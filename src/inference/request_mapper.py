"""Map validated API payloads into serving-friendly feature structures.

This module converts request objects into:

- a normalized dictionary used by the orchestration layer
- a lightweight tabular frame that mirrors the structured features expected
  by the eventual exported valuation pipeline

The mapping is intentionally explicit so it is easy to review, test, and
keep aligned with both the API contract and the training schema.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


KEY_FIELDS = (
    "property_type",
    "governorate",
    "city",
    "size_m2",
    "condition",
    "description",
)


def map_request(payload: Any) -> dict[str, Any]:
    """Normalize an API request object into a plain serving dictionary.

    The mapper trims text fields, standardizes image references, and adds an
    ``input_completeness`` score used later by confidence estimation.
    """

    image_refs = [str(v) for v in getattr(payload, "image_refs", []) if str(v).strip()]
    mapped = {
        "property_type": str(payload.property_type),
        "transaction_type": str(getattr(payload, "transaction_type", "sale") or "sale").strip().lower(),
        "governorate": str(payload.governorate).strip(),
        "city": str(payload.city).strip(),
        "neighborhood": str(payload.neighborhood).strip(),
        "surface_m2": float(payload.size_m2),
        "size_m2": float(payload.size_m2),
        "rooms": int(getattr(payload, "bedrooms", 0) or 0) + (1 if str(payload.property_type).strip().lower() != "terrain" else 0),
        "bedrooms": int(payload.bedrooms),
        "bathrooms": int(payload.bathrooms),
        "condition": str(payload.condition),
        "has_pool": bool(payload.has_pool),
        "has_garden": bool(payload.has_garden),
        "has_parking": bool(payload.has_parking),
        "sea_view": bool(payload.sea_view),
        "elevator": bool(payload.elevator),
        "description": str(payload.description or "").strip(),
        "uploaded_images_count": int(payload.uploaded_images_count),
        "image_refs": image_refs,
        "latitude": None,
        "longitude": None,
    }
    present = sum(1 for field in KEY_FIELDS if str(mapped.get(field, "")).strip())
    mapped["input_completeness"] = round(present / len(KEY_FIELDS), 3)
    return mapped


def to_feature_frame(mapped: dict[str, Any]) -> pd.DataFrame:
    """Project mapped request data into a one-row structured feature frame."""

    return pd.DataFrame(
        [
            {
                "property_type": mapped["property_type"],
                "transaction_type": mapped.get("transaction_type", "sale"),
                "governorate": mapped["governorate"],
                "city": mapped["city"],
                "surface_m2": mapped["surface_m2"],
                "rooms": mapped.get("rooms", 0),
                "bedrooms": mapped["bedrooms"],
                "bathrooms": mapped["bathrooms"],
                "condition": mapped["condition"],
                "has_pool": mapped["has_pool"],
                "has_garden": mapped["has_garden"],
                "has_parking": mapped["has_parking"],
                "sea_view": mapped["sea_view"],
                "elevator": mapped["elevator"],
                "description_length": len(mapped["description"]),
                "uploaded_images_count": mapped["uploaded_images_count"],
            }
        ]
    )
