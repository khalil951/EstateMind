"""FastAPI entrypoints and request/response contracts for EstateMind."""

from __future__ import annotations

import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, List, Literal

from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel, Field

from src.inference.valuation_service import ValuationService
from src.vision.feature_aggregation import PROPERTY_HINT_MAP

app = FastAPI(title="EstateMind API", version="3.0.0")
_valuation_service = ValuationService()


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _build_upload_consistency_warnings(selected_property_type: str, image_refs: list[str]) -> list[str]:
    if not image_refs:
        return []

    rows = _valuation_service.image_type.classify_many(image_refs)
    if not rows:
        return []

    clip_rows = [row for row in rows if str(row.get("cv_mode", "")).strip() == "clip_feature_inference"]
    if not clip_rows:
        return []

    inferred_candidates: list[str] = []
    for row in clip_rows:
        top = row.get("top_prediction") or {}
        label = str(top.get("label", "")).strip()
        inferred = PROPERTY_HINT_MAP.get(label)
        if inferred:
            inferred_candidates.append(inferred)

    if not inferred_candidates:
        return []

    inferred_property_type = Counter(inferred_candidates).most_common(1)[0][0]
    if inferred_property_type == selected_property_type:
        return []

    return [
        f"clip_property_type_mismatch:selected={selected_property_type},inferred={inferred_property_type}"
    ]


class PropertyRequest(BaseModel):
    """Validated property payload accepted by the JSON and upload endpoints."""

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
    image_refs: list[str] = Field(default_factory=list)
    transaction_type: Literal["sale", "rent"] = "sale"


class ValuationResponse(BaseModel):
    """Normalized valuation response returned to the frontend."""

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
    prediction_mode: str
    explanation_mode: str
    sentiment_mode: str
    cv_mode: str
    warnings: List[str]
    uncertainty_reasons: List[str]
    uncertainty_mode: str
    model_info: dict


@app.get("/health")
def health() -> dict[str, str]:
    """Return a lightweight health check for service monitoring."""

    return {"status": "ok"}


@app.post("/estimate", response_model=ValuationResponse)
def estimate_price(payload: PropertyRequest) -> ValuationResponse:
    """Run the full valuation pipeline for a JSON property request."""

    result = _valuation_service.estimate(payload)
    return ValuationResponse(**result)


@app.post("/estimate-upload", response_model=ValuationResponse)
def estimate_price_upload(
    property_type: Literal["Terrain", "Maison", "Appartement"] = Form(...),
    governorate: str = Form(...),
    city: str = Form(...),
    neighborhood: str = Form(""),
    size_m2: float = Form(...),
    bedrooms: int = Form(0),
    bathrooms: int = Form(0),
    condition: Literal["New", "Excellent", "Good", "Fair", "Needs Renovation"] = Form(...),
    has_pool: str = Form("false"),
    has_garden: str = Form("false"),
    has_parking: str = Form("false"),
    sea_view: str = Form("false"),
    elevator: str = Form("false"),
    description: str = Form(""),
    transaction_type: Literal["sale", "rent"] = Form("sale"),
    images: list[UploadFile] | None = File(default=None),
) -> ValuationResponse:
    """Accept multipart uploads, persist images temporarily, and estimate value."""

    upload_dir = Path("artifacts") / "tmp" / "api_uploads" / str(uuid.uuid4())
    upload_dir.mkdir(parents=True, exist_ok=True)
    image_refs: list[str] = []
    try:
        for upload in images or []:
            suffix = Path(upload.filename or "image.jpg").suffix or ".jpg"
            target = upload_dir / f"{uuid.uuid4().hex}{suffix}"
            with target.open("wb") as handle:
                shutil.copyfileobj(upload.file, handle)
            image_refs.append(str(target))

        payload = PropertyRequest(
            property_type=property_type,
            governorate=governorate,
            city=city,
            neighborhood=neighborhood,
            size_m2=size_m2,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            condition=condition,
            has_pool=_parse_bool(has_pool),
            has_garden=_parse_bool(has_garden),
            has_parking=_parse_bool(has_parking),
            sea_view=_parse_bool(sea_view),
            elevator=_parse_bool(elevator),
            description=description,
            uploaded_images_count=len(image_refs),
            image_refs=image_refs,
            transaction_type=transaction_type,
        )
        external_warnings = _build_upload_consistency_warnings(property_type, image_refs)
        result = _valuation_service.estimate(payload, external_warnings=external_warnings)
        return ValuationResponse(**result)
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)
