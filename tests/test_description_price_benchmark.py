import shutil
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.description_price_benchmark import (
    build_summary,
    load_split,
    parse_surface_m2,
    run_benchmark,
)


def test_parse_surface_m2_handles_common_formats() -> None:
    assert parse_surface_m2("Surface: 112 m2 | Prix: 315000 TND") == 112.0
    assert parse_surface_m2("terrain 301 m2 prix 195000 tnd") == 301.0
    assert parse_surface_m2("aucune surface") is None


def test_run_benchmark_compares_target_strategies() -> None:
    rows = [
        {
            "sample_id": "a",
            "source": "x",
            "raw_text": "Appartement a vendre a Tunis. Surface: 100 m2 | Prix: 200000 TND.",
            "target_price_tnd": 200000,
            "price_label_source": "col",
            "clean_text": "appartement vendre tunis surface 100 m2 prix 200000 tnd",
            "token_count": 10,
            "char_count": 60,
            "has_price_label": True,
        },
        {
            "sample_id": "b",
            "source": "x",
            "raw_text": "Appartement a vendre a Tunis. Surface: 120 m2 | Prix: 240000 TND.",
            "target_price_tnd": 240000,
            "price_label_source": "col",
            "clean_text": "appartement vendre tunis surface 120 m2 prix 240000 tnd",
            "token_count": 10,
            "char_count": 60,
            "has_price_label": True,
        },
        {
            "sample_id": "c",
            "source": "x",
            "raw_text": "Maison a vendre a Sousse. Surface: 200 m2 | Prix: 400000 TND.",
            "target_price_tnd": 400000,
            "price_label_source": "col",
            "clean_text": "maison vendre sousse surface 200 m2 prix 400000 tnd",
            "token_count": 10,
            "char_count": 60,
            "has_price_label": True,
        },
        {
            "sample_id": "d",
            "source": "x",
            "raw_text": "Terrain a vendre a Nabeul. Surface: 300 m2 | Prix: 210000 TND.",
            "target_price_tnd": 210000,
            "price_label_source": "col",
            "clean_text": "terrain vendre nabeul surface 300 m2 prix 210000 tnd",
            "token_count": 10,
            "char_count": 60,
            "has_price_label": True,
        },
        {
            "sample_id": "e",
            "source": "x",
            "raw_text": "Appartement a louer a Tunis. Surface: 80 m2 | Prix: 1500 TND.",
            "target_price_tnd": 1500,
            "price_label_source": "col",
            "clean_text": "appartement louer tunis surface 80 m2 prix 1500 tnd",
            "token_count": 10,
            "char_count": 60,
            "has_price_label": True,
        },
        {
            "sample_id": "f",
            "source": "x",
            "raw_text": "Maison a vendre a Ariana. Surface: 150 m2 | Prix: 300000 TND.",
            "target_price_tnd": 300000,
            "price_label_source": "col",
            "clean_text": "maison vendre ariana surface 150 m2 prix 300000 tnd",
            "token_count": 10,
            "char_count": 60,
            "has_price_label": True,
        },
    ]
    df = pd.DataFrame(rows)
    base_dir = Path("artifacts") / "test_description_price_benchmark" / str(uuid4())
    if base_dir.exists():
        shutil.rmtree(base_dir, ignore_errors=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    train_path = base_dir / "train.csv"
    val_path = base_dir / "val.csv"
    test_path = base_dir / "test.csv"
    df.iloc[:4].to_csv(train_path, index=False)
    df.iloc[4:5].to_csv(val_path, index=False)
    df.iloc[5:6].to_csv(test_path, index=False)

    train_df = load_split(train_path)
    val_df = load_split(val_path)
    test_df = load_split(test_path)
    metrics_df = run_benchmark(train_df, val_df, test_df)
    summary = build_summary(metrics_df)

    assert set(metrics_df["target_strategy"].unique()) == {"price_tnd", "log_price_tnd", "price_per_m2"}
    assert {"mae_price", "rmse_price", "r2_price", "rows_used", "coverage_ratio"}.issubset(metrics_df.columns)
    assert "val" in summary["best_by_split"]
    assert "test" in summary["best_by_split"]
    assert summary["rows"] == len(metrics_df)
