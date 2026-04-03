import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.inference.valuation_service import ValuationService


@pytest.mark.skipif(os.getenv("ESTATEMIND_RUN_SMOKE") != "1", reason="smoke suite is opt-in")
def test_real_artifacts_smoke_run_against_synthetic_rows() -> None:
    import pandas as pd

    service = ValuationService()
    df = pd.read_csv(Path("tests/fixtures/synthetic_properties.csv"))
    for _, row in df.iterrows():
        as_bool = lambda value: str(value).strip().lower() == "true"
        payload = SimpleNamespace(
            property_type=row["property_type"],
            transaction_type=row["transaction_type"],
            governorate=row["governorate"],
            city=row["city"],
            neighborhood=row["neighborhood"],
            size_m2=float(row["size_m2"]),
            bedrooms=int(row["bedrooms"]),
            bathrooms=int(row["bathrooms"]),
            condition=row["condition"],
            has_pool=as_bool(row["has_pool"]),
            has_garden=as_bool(row["has_garden"]),
            has_parking=as_bool(row["has_parking"]),
            sea_view=as_bool(row["sea_view"]),
            elevator=as_bool(row["elevator"]),
            description=row["description"],
            uploaded_images_count=1 if Path(str(row.get("image_ref", ""))).exists() else 0,
            image_refs=[str(row["image_ref"])] if Path(str(row.get("image_ref", ""))).exists() else [],
        )
        result = service.estimate(payload)
        assert result["estimated_price"] > 0
