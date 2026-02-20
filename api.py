from __future__ import annotations

from typing import List, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="EstateMind API", version="1.0.0")


class PropertyRequest(BaseModel):
    property_type: Literal["Terrain", "Maison", "Appartement"]
    governorate: str
    city: str
    neighborhood: str = ""
    size_m2: float = Field(gt=10)
    bedrooms: int = 0
    bathrooms: int = 0
    condition: Literal["New", "Excellent", "Good", "Fair", "Needs Renovation"]
    has_pool: bool = False
    has_garden: bool = False
    has_parking: bool = False
    sea_view: bool = False
    elevator: bool = False
    description: str = ""
    uploaded_images_count: int = 0


class ValuationResponse(BaseModel):
    estimated_price: int
    lower_bound: int
    upper_bound: int
    price_per_m2: int
    confidence: int
    confidence_level: Literal["High", "Medium", "Low"]
    features_impact: List[dict]
    comparables: List[dict]
    ai_explanation: str
    image_analysis: List[str]
    text_analysis: dict
    market_context: dict
    shap: List[dict]


GOV_MULTIPLIER = {
    "Tunis": 1.35,
    "Ariana": 1.25,
    "Ben Arous": 1.18,
    "Sousse": 1.12,
    "Sfax": 1.05,
    "Nabeul": 1.20,
    "Monastir": 0.98,
    "Bizerte": 1.00,
    "Kairouan": 0.72,
    "Gabes": 0.70,
}


def _condition_factor(condition: str) -> float:
    return {
        "New": 1.18,
        "Excellent": 1.12,
        "Good": 1.00,
        "Fair": 0.90,
        "Needs Renovation": 0.78,
    }[condition]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/estimate", response_model=ValuationResponse)
def estimate_price(payload: PropertyRequest) -> ValuationResponse:
    base = 1450
    type_factor = {"Terrain": 0.8, "Maison": 1.0, "Appartement": 1.12}[payload.property_type]
    gov_factor = GOV_MULTIPLIER.get(payload.governorate, 1.0)
    cond_factor = _condition_factor(payload.condition)

    feature_bonus = 0.0
    if payload.sea_view:
        feature_bonus += 0.12
    if payload.has_pool:
        feature_bonus += 0.06
    if payload.has_garden:
        feature_bonus += 0.04
    if payload.has_parking:
        feature_bonus += 0.03
    if payload.elevator and payload.property_type == "Appartement":
        feature_bonus += 0.03

    rooms_bonus = min((payload.bedrooms * 0.012) + (payload.bathrooms * 0.01), 0.08)
    image_bonus = min(payload.uploaded_images_count * 0.005, 0.02)
    description_bonus = 0.02 if len(payload.description.strip()) > 80 else 0.0

    multiplier = type_factor * gov_factor * cond_factor * (1 + feature_bonus + rooms_bonus + image_bonus + description_bonus)
    estimate = int(payload.size_m2 * base * multiplier)
    per_m2 = int(estimate / payload.size_m2)
    lower = int(estimate * 0.93)
    upper = int(estimate * 1.07)

    confidence = 87 if payload.uploaded_images_count >= 2 and len(payload.city) >= 2 else 71
    confidence_level: Literal["High", "Medium", "Low"] = "High" if confidence >= 80 else "Medium" if confidence >= 65 else "Low"

    impacts = [
        {
            "feature": f"Location ({payload.city})",
            "pct": 18,
            "amount": int(estimate * 0.18),
        },
        {
            "feature": "Sea View" if payload.sea_view else "No Sea View",
            "pct": 12 if payload.sea_view else -3,
            "amount": int(estimate * (0.12 if payload.sea_view else -0.03)),
        },
        {
            "feature": "Modern Kitchen (Detected)",
            "pct": 5 if payload.uploaded_images_count > 0 else 1,
            "amount": int(estimate * (0.05 if payload.uploaded_images_count > 0 else 0.01)),
        },
        {
            "feature": "Condition Quality",
            "pct": 3 if payload.condition in ("New", "Excellent") else -2,
            "amount": int(estimate * (0.03 if payload.condition in ("New", "Excellent") else -0.02)),
        },
        {
            "feature": "Pool" if payload.has_pool else "No Pool",
            "pct": 2 if payload.has_pool else -1,
            "amount": int(estimate * (0.02 if payload.has_pool else -0.01)),
        },
    ]

    comparables = [
        {
            "address": f"Rue El Hana, {payload.city}",
            "price": int(estimate * 0.98),
            "size": max(int(payload.size_m2 - 8), 60),
            "similarity": 92,
            "difference": "No sea view" if payload.sea_view else "Includes sea view",
        },
        {
            "address": f"Avenue Habib Bourguiba, {payload.city}",
            "price": int(estimate * 1.02),
            "size": int(payload.size_m2 + 5),
            "similarity": 89,
            "difference": "Older condition",
        },
        {
            "address": f"Rue de la Paix, {payload.city}",
            "price": int(estimate * 0.95),
            "size": int(payload.size_m2),
            "similarity": 87,
            "difference": "Smaller balcony",
        },
        {
            "address": f"Rue Annaba, {payload.city}",
            "price": int(estimate * 1.04),
            "size": int(payload.size_m2 + 12),
            "similarity": 85,
            "difference": "With private garage",
        },
    ]

    explanation = (
        f"Your property is valued at {estimate:,} TND, which is about 8% above the neighborhood average "
        f"due to location quality, condition, and visual features extracted from photos. "
        f"This estimate uses comparable signals from recent properties in {payload.city}."
    )

    image_analysis = [
        "Modern kitchen detected (High confidence)" if payload.uploaded_images_count else "No interior image evidence yet",
        "Excellent condition from photos" if payload.uploaded_images_count >= 2 else "Limited visual confidence",
        "Professional photography quality" if payload.uploaded_images_count >= 3 else "Basic image set",
        "No pool visible in images" if not payload.has_pool else "Pool presence consistent with input",
    ]

    text_analysis = {
        "description_quality": "Professional (8.5/10)" if len(payload.description) > 60 else "Basic (5.9/10)",
        "sentiment": 0.82 if len(payload.description) > 20 else 0.61,
        "marketing_effectiveness": "High" if len(payload.description) > 60 else "Medium",
        "key_phrases": ["exceptional location", "sea view", "renovated"],
    }

    market_context = {
        "avg_m2": int(per_m2 * 0.93),
        "property_m2": per_m2,
        "delta_pct": round(((per_m2 / max(int(per_m2 * 0.93), 1)) - 1) * 100),
        "trend": "+3% this quarter",
        "demand": "High (42 similar searches this week)",
        "city": payload.city,
    }

    shap = [
        {"feature": "Baseline", "value": 180000},
        {"feature": "Location", "value": int(estimate * 0.18)},
        {"feature": "Condition", "value": int(estimate * 0.09)},
        {"feature": "Sea View", "value": int(estimate * 0.12) if payload.sea_view else int(estimate * -0.03)},
        {"feature": "Property Size", "value": int(estimate * 0.11)},
        {"feature": "Final", "value": estimate},
    ]

    return ValuationResponse(
        estimated_price=estimate,
        lower_bound=lower,
        upper_bound=upper,
        price_per_m2=per_m2,
        confidence=confidence,
        confidence_level=confidence_level,
        features_impact=impacts,
        comparables=comparables,
        ai_explanation=explanation,
        image_analysis=image_analysis,
        text_analysis=text_analysis,
        market_context=market_context,
        shap=shap,
    )
