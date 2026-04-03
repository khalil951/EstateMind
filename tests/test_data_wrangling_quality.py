from pathlib import Path

from src.data_wrangling_pipeline import CANONICAL_COLUMNS, build_unified_dataset, quality_report


def test_wrangled_dataset_not_empty() -> None:
    df = build_unified_dataset(Path("data/csv"))
    assert len(df) > 0


def test_wrangled_schema_contains_required_columns() -> None:
    df = build_unified_dataset(Path("data/csv"))
    for col in CANONICAL_COLUMNS:
        assert col in df.columns


def test_quality_consistency_rules() -> None:
    df = build_unified_dataset(Path("data/csv"))
    report = quality_report(df)

    assert report["row_count"] > 100
    assert report["invalid_price_rows"] == 0
    assert report["invalid_surface_rows"] == 0
    assert report["url_present_ratio"] >= 0.2
    assert report["duplicate_ratio"] <= 0.1


def test_price_per_m2_logic() -> None:
    df = build_unified_dataset(Path("data/csv"))
    sample = df.dropna(subset=["price_tnd", "surface_m2", "price_per_m2"]).head(100)
    if len(sample) == 0:
        assert True
        return
    recomputed = (sample["price_tnd"] / sample["surface_m2"]).round(2)
    assert (recomputed == sample["price_per_m2"]).mean() >= 0.95
