from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.fallback_model import FallbackTabularModelService


def main() -> None:
    sample_path = Path("data/csv/preprocessed/final_listings_preprocessed.csv")
    df = pd.read_csv(sample_path)

    df = df.dropna(subset=["price_tnd", "surface_m2", "property_type", "governorate", "city"]).copy()
    df = df[df["price_tnd"] > 0].copy()
    if len(df) > 2500:
        df = df.sample(n=2500, random_state=42)

    service = FallbackTabularModelService()
    rows: list[dict[str, object]] = []

    for row in df.itertuples(index=False):
        mapped = {
            "transaction_type": str(getattr(row, "transaction_type", "sale") or "sale"),
            "property_type": str(getattr(row, "property_type")),
            "surface_m2": float(getattr(row, "surface_m2")),
            "rooms": float(getattr(row, "rooms")) if pd.notna(getattr(row, "rooms", np.nan)) else None,
            "bedrooms": float(getattr(row, "bedrooms")) if pd.notna(getattr(row, "bedrooms", np.nan)) else 0.0,
            "bathrooms": float(getattr(row, "bathrooms")) if pd.notna(getattr(row, "bathrooms", np.nan)) else 0.0,
            "governorate": str(getattr(row, "governorate")),
            "city": str(getattr(row, "city")),
            "latitude": float(getattr(row, "latitude")) if pd.notna(getattr(row, "latitude", np.nan)) else None,
            "longitude": float(getattr(row, "longitude")) if pd.notna(getattr(row, "longitude", np.nan)) else None,
        }
        pred = service.predict(mapped)
        rows.append(
            {
                "property_type": mapped["property_type"],
                "transaction_type": mapped["transaction_type"],
                "y_true": float(getattr(row, "price_tnd")),
                "y_pred": float(pred.estimated_price) if pred is not None else np.nan,
                "predicted": pred is not None,
            }
        )

    pred_df = pd.DataFrame(rows)
    covered = pred_df[pred_df["predicted"]].copy()

    if covered.empty:
        print(json.dumps({"error": "No rows could be predicted by fallback model service."}, indent=2))
        return

    y_true = covered["y_true"].to_numpy(dtype=float)
    y_pred = covered["y_pred"].to_numpy(dtype=float)
    abs_err = np.abs(y_pred - y_true)
    mae = float(abs_err.mean())
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mape = float(np.mean(abs_err / np.clip(np.abs(y_true), 1e-9, None)) * 100.0)
    smape = float(np.mean(2.0 * abs_err / np.clip(np.abs(y_true) + np.abs(y_pred), 1e-9, None)) * 100.0)

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else float("nan")

    err_pct = abs_err / np.clip(np.abs(y_true), 1e-9, None) * 100.0

    segments: list[dict[str, object]] = []
    for (ptype, tx), g in covered.groupby(["property_type", "transaction_type"]):
        yt = g["y_true"].to_numpy(dtype=float)
        yp = g["y_pred"].to_numpy(dtype=float)
        ae = np.abs(yp - yt)
        seg_mape = float(np.mean(ae / np.clip(np.abs(yt), 1e-9, None)) * 100.0)
        segments.append(
            {
                "property_type": str(ptype),
                "transaction_type": str(tx),
                "n": int(len(g)),
                "mae": round(float(ae.mean()), 2),
                "mape_pct": round(seg_mape, 2),
            }
        )
    segments = sorted(segments, key=lambda x: (-int(x["n"]), str(x["property_type"]), str(x["transaction_type"])))#type:ignore

    result = {
        "dataset": str(sample_path),
        "sample_rows": int(len(pred_df)),
        "predicted_rows": int(len(covered)),
        "coverage_pct": round(100.0 * len(covered) / max(len(pred_df), 1), 2),
        "metrics": {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "mape_pct": round(mape, 2),
            "smape_pct": round(smape, 2),
            "r2": round(r2, 4),
            "median_abs_error": round(float(np.median(abs_err)), 2),
            "p90_abs_error": round(float(np.percentile(abs_err, 90)), 2),
            "p95_abs_error": round(float(np.percentile(abs_err, 95)), 2),
            "median_abs_pct_error": round(float(np.median(err_pct)), 2),
            "p90_abs_pct_error": round(float(np.percentile(err_pct, 90)), 2),
            "p95_abs_pct_error": round(float(np.percentile(err_pct, 95)), 2),
        },
        "top_segments": segments[:12],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
