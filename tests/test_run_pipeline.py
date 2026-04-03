import argparse
import json
import shutil
from uuid import uuid4
from pathlib import Path

import pandas as pd

from src.run_pipeline import run_full_pipeline


def test_run_full_pipeline_writes_outputs() -> None:
    base_dir = Path("artifacts") / "test_run_pipeline" / str(uuid4())
    if base_dir.exists():
        shutil.rmtree(base_dir, ignore_errors=True)
    input_dir = base_dir / "input_csv"
    out_dir = base_dir / "out"
    input_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.DataFrame(
        [
            {
                "source": "Test",
                "listing_url": "http://x/1",
                "title": "A louer appartement S+1",
                "price": "1200 DT",
                "city": "Tunis",
                "governorate": "Tunis",
            },
            {
                "source": "Test",
                "listing_url": "http://x/2",
                "title": "A vendre villa",
                "price": "350000 DT",
                "city": "Sousse",
                "governorate": "Sousse",
            },
        ]
    )
    raw.to_csv(input_dir / "sample.csv", index=False, encoding="utf-8")

    args = argparse.Namespace(
        input_dir=str(input_dir),
        wrangled_csv=str(out_dir / "wrangled.csv"),
        wrangling_report_json=str(out_dir / "wrangling_report.json"),
        preprocessed_csv=str(out_dir / "preprocessed.csv"),
        preprocessing_report_json=str(out_dir / "preprocessing_report.json"),
        pipeline_report_json=str(out_dir / "pipeline_report.json"),
        geocode_max_cities=0,  # offline-safe for tests
        geocode_sleep_sec=0.0,
        knn_neighbors=2,
    )

    report = run_full_pipeline(args)

    assert (out_dir / "wrangled.csv").exists()
    assert (out_dir / "wrangling_report.json").exists()
    assert (out_dir / "preprocessed.csv").exists()
    assert (out_dir / "preprocessing_report.json").exists()
    assert (out_dir / "pipeline_report.json").exists()

    preprocessed = pd.read_csv(out_dir / "preprocessed.csv")
    assert len(preprocessed) > 0
    assert "title" not in preprocessed.columns
    assert "transaction_type" in preprocessed.columns
    assert "description" in preprocessed.columns

    persisted = json.loads((out_dir / "pipeline_report.json").read_text(encoding="utf-8"))
    assert report["counts"]["preprocessed_rows"] == persisted["counts"]["preprocessed_rows"]
    assert persisted["counts"]["wrangled_rows"] >= 1
