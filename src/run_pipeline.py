from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

try:
    from src.data_preprocessing import BasePreprocessor, PreprocessConfig
    from src.data_wrangling_pipeline import run_pipeline as run_wrangling_pipeline
except ModuleNotFoundError:  # pragma: no cover
    from data_preprocessing import BasePreprocessor, PreprocessConfig
    from data_wrangling_pipeline import run_pipeline as run_wrangling_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full EstateMind pipeline: wrangling -> preprocessing")
    parser.add_argument("--input-dir", default="data/csv", help="Directory containing source CSV files")
    parser.add_argument("--wrangled-csv", default="data/csv/final_listings_wrangled.csv")
    parser.add_argument("--wrangling-report-json", default="data/csv/final_listings_dq_report.json")
    parser.add_argument("--preprocessed-csv", default="data/csv/final_listings_preprocessed.csv")
    parser.add_argument("--preprocessing-report-json", default="data/csv/final_listings_preprocessing_report.json")
    parser.add_argument("--pipeline-report-json", default="data/csv/final_pipeline_report.json")
    parser.add_argument("--geocode-max-cities", type=int, default=250)
    parser.add_argument("--geocode-sleep-sec", type=float, default=0.25)
    parser.add_argument("--knn-neighbors", type=int, default=5)
    return parser.parse_args()


def run_full_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    input_dir = Path(args.input_dir)
    wrangled_csv = Path(args.wrangled_csv)
    wrangling_report_json = Path(args.wrangling_report_json)
    preprocessed_csv = Path(args.preprocessed_csv)
    preprocessing_report_json = Path(args.preprocessing_report_json)
    pipeline_report_json = Path(args.pipeline_report_json)

    # Step 1: Wrangling
    t_w0 = time.time()
    wrangled_df, wrangling_report = run_wrangling_pipeline(
        input_dir=input_dir,
        output_csv=wrangled_csv,
        dq_json=wrangling_report_json,
    )
    wrangling_seconds = round(time.time() - t_w0, 3)

    # Step 2: Preprocessing
    t_p0 = time.time()
    cfg = PreprocessConfig(
        input_csv=str(wrangled_csv),
        output_csv=str(preprocessed_csv),
        report_json=str(preprocessing_report_json),
        geocode_max_cities=int(args.geocode_max_cities),
        geocode_sleep_sec=float(args.geocode_sleep_sec),
        knn_neighbors=int(args.knn_neighbors),
    )
    preprocessor = BasePreprocessor(cfg)
    preprocessed_df, preprocessing_report = preprocessor.run()
    preprocessing_seconds = round(time.time() - t_p0, 3)

    total_seconds = round(time.time() - t0, 3)
    pipeline_report = {
        "artifacts": {
            "wrangled_csv": str(wrangled_csv),
            "wrangling_report_json": str(wrangling_report_json),
            "preprocessed_csv": str(preprocessed_csv),
            "preprocessing_report_json": str(preprocessing_report_json),
        },
        "timing_seconds": {
            "wrangling": wrangling_seconds,
            "preprocessing": preprocessing_seconds,
            "total": total_seconds,
        },
        "counts": {
            "wrangled_rows": int(len(wrangled_df)),
            "preprocessed_rows": int(len(preprocessed_df)),
            "preprocessed_columns": int(len(preprocessed_df.columns)),
        },
        "wrangling_report": wrangling_report,
        "preprocessing_report": preprocessing_report,
    }

    pipeline_report_json.parent.mkdir(parents=True, exist_ok=True)
    pipeline_report_json.write_text(json.dumps(pipeline_report, indent=2), encoding="utf-8")
    return pipeline_report


def main() -> int:
    args = parse_args()
    report = run_full_pipeline(args)
    print("Full pipeline completed")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
