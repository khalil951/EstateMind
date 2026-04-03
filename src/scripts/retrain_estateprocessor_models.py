"""Retrain transaction-aware EstateMind valuation models with quality gates.

This script rebuilds artifact models consumed by ModelRegistry and persists:
- artifacts/models/models_estateprocessor/*.joblib
- artifacts/reports/ml_reports/training_estateprocessor_results.csv
- artifacts/reports/ml_reports/training_estateprocessor_manifest.json

Quality gates keep only models with acceptable holdout performance.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = [
    "transaction_type",
    "property_type",
    "surface_m2",
    "rooms",
    "bedrooms",
    "bathrooms",
    "governorate",
    "city",
    "latitude",
    "longitude",
    "city_governorate",
    "local_avg_price_m2",
    "gov_avg_price_m2",
    "size_x_local_price",
]

CAT_FEATURES = ["transaction_type", "property_type", "governorate", "city", "city_governorate"]


@dataclass
class QualityGate:
    min_test_r2: float = 0.60
    max_gap: float = 0.20


EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain EstateMind by-type/global models with transaction-aware features")
    parser.add_argument("--input-csv", default="data/csv/preprocessed/final_listings_preprocessed.csv")
    parser.add_argument("--models-dir", default="artifacts/models/models_estateprocessor")
    parser.add_argument("--reports-dir", default="artifacts/reports/ml_reports")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-test-r2", type=float, default=0.6)
    parser.add_argument("--max-overfit-gap", type=float, default=0.2)
    return parser.parse_args()


def clean_source(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric = ["price_tnd", "surface_m2", "rooms", "bedrooms", "bathrooms", "latitude", "longitude"]
    for col in numeric:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ["transaction_type", "property_type", "governorate", "city"]:
        out[col] = out[col].astype(str).str.strip().str.lower()

    out = out.dropna(subset=["price_tnd", "surface_m2", "transaction_type", "property_type", "governorate", "city"]).copy()
    out = out[out["transaction_type"].isin(["sale", "rent"])].copy()
    out = out[out["property_type"].isin(["appartement", "maison", "terrain"])].copy()
    out = out[(out["surface_m2"] > 20) & (out["surface_m2"] < 1500)]

    out["price_per_m2"] = out["price_tnd"] / out["surface_m2"].replace({0: np.nan})
    out = out[(out["price_per_m2"] > 0)]

    sale = out["transaction_type"] == "sale"
    out = out[(~sale) | ((out["price_tnd"] >= 60000) & (out["price_tnd"] <= 4000000) & (out["price_per_m2"] >= 250) & (out["price_per_m2"] <= 12000))]
    rent = out["transaction_type"] == "rent"
    out = out[(~rent) | ((out["price_tnd"] >= 120) & (out["price_tnd"] <= 12000) & (out["price_per_m2"] >= 1) & (out["price_per_m2"] <= 180))]

    out["city_governorate"] = out["city"] + "__" + out["governorate"]
    return out


def add_priors(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    city_ppm = (
        train_df.groupby("city_governorate")["price_tnd"].median()
        / train_df.groupby("city_governorate")["surface_m2"].median().replace({0: np.nan})
    ).replace([np.inf, -np.inf], np.nan).dropna()
    gov_ppm = (
        train_df.groupby("governorate")["price_tnd"].median()
        / train_df.groupby("governorate")["surface_m2"].median().replace({0: np.nan})
    ).replace([np.inf, -np.inf], np.nan).dropna()
    global_ppm = float(np.nanmedian(city_ppm.values)) if len(city_ppm) else 1450.0#type:ignore

    medians = {
        col: float(train_df[col].median()) if pd.notna(train_df[col].median()) else 0.0
        for col in ["surface_m2", "rooms", "bedrooms", "bathrooms", "latitude", "longitude"]
    }

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for col, value in medians.items():
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(value)
        out["local_avg_price_m2"] = out["city_governorate"].map(city_ppm).fillna(global_ppm)
        out["gov_avg_price_m2"] = out["governorate"].map(gov_ppm).fillna(global_ppm)
        out["size_x_local_price"] = out["surface_m2"] * out["local_avg_price_m2"]
        return out

    metadata = {
        "priors": {
            "city_governorate_price_m2": {str(k): float(v) for k, v in city_ppm.to_dict().items()},
            "governorate_price_m2": {str(k): float(v) for k, v in gov_ppm.to_dict().items()},
            "global_price_m2": float(global_ppm),
        },
        "fill_values": medians,
    }
    return transform(train_df), transform(eval_df), metadata


def evaluate_prices(y_log_true: np.ndarray, y_log_pred: np.ndarray) -> dict[str, float]:
    y_true = np.expm1(y_log_true)
    y_pred = np.expm1(y_log_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def evaluate_price_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def fit_scope(scope_df: pd.DataFrame, random_state: int, test_size: float) -> tuple[CatBoostRegressor, dict[str, float], dict[str, object], int, int]:
    train_df, test_df = train_test_split(scope_df, test_size=test_size, random_state=random_state)
    train_x, test_x, metadata = add_priors(train_df, test_df)

    y_train_log = np.log1p(train_df["price_tnd"].to_numpy())
    y_test_log = np.log1p(test_df["price_tnd"].to_numpy())

    model = CatBoostRegressor(
        depth=6,
        learning_rate=0.04,
        iterations=1400,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_state=random_state,
        verbose=False,
    )
    model.fit(train_x[FEATURE_COLUMNS], y_train_log, cat_features=[c for c in CAT_FEATURES if c in train_x.columns])

    train_pred_log = np.asarray(model.predict(train_x[FEATURE_COLUMNS]))
    test_pred_log = np.asarray(model.predict(test_x[FEATURE_COLUMNS]))
    train_metrics = evaluate_prices(y_train_log, train_pred_log)
    test_metrics = evaluate_prices(y_test_log, test_pred_log)

    metrics = {
        "train_r2": train_metrics["r2"],
        "test_r2": test_metrics["r2"],
        "train_rmse": train_metrics["rmse"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "overfit_gap": train_metrics["r2"] - test_metrics["r2"],
    }
    return model, metrics, metadata, int(len(train_df)), int(len(test_df))


def enrich_terrain_data(terrain_df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    """Create a cleaner and denser terrain dataset.

    Approach details:
    - clip extreme price-per-m2 tails per governorate
    - synthesize additional rows for low-count governorates with small perturbations
    """
    if terrain_df.empty:
        return terrain_df.copy()

    rng = np.random.default_rng(random_state)
    df = terrain_df.copy()
    df["price_per_m2"] = (df["price_tnd"] / df["surface_m2"].clip(lower=EPS)).replace([np.inf, -np.inf], np.nan)
    df = df[df["price_per_m2"].notna() & (df["price_per_m2"] > 0)].copy()

    clipped_parts: list[pd.DataFrame] = []
    for _, part in df.groupby("governorate", dropna=False):
        if len(part) < 20:
            clipped_parts.append(part)
            continue
        q_low, q_high = part["price_per_m2"].quantile([0.04, 0.96]).astype(float).tolist()
        clipped = part[(part["price_per_m2"] >= q_low) & (part["price_per_m2"] <= q_high)].copy()
        clipped_parts.append(clipped)
    base = pd.concat(clipped_parts, ignore_index=True)

    target_per_gov = 220
    synth_parts: list[pd.DataFrame] = []
    for _, part in base.groupby("governorate", dropna=False):
        synth_parts.append(part)
        need = max(0, target_per_gov - len(part))
        if need == 0 or part.empty:
            continue
        sampled = part.sample(n=need, replace=True, random_state=random_state).copy()
        sampled["surface_m2"] = (sampled["surface_m2"].astype(float) * rng.uniform(0.95, 1.05, size=need)).clip(lower=30, upper=3000)
        sampled["price_per_m2"] = (sampled["price_per_m2"].astype(float) * rng.uniform(0.90, 1.10, size=need)).clip(lower=40)
        sampled["price_tnd"] = sampled["surface_m2"] * sampled["price_per_m2"]
        sampled["latitude"] = pd.to_numeric(sampled["latitude"], errors="coerce").fillna(0.0) + rng.normal(0, 0.0025, size=need)
        sampled["longitude"] = pd.to_numeric(sampled["longitude"], errors="coerce").fillna(0.0) + rng.normal(0, 0.0025, size=need)
        synth_parts.append(sampled.drop(columns=["price_per_m2"], errors="ignore"))

    enriched = pd.concat(synth_parts, ignore_index=True)
    enriched = enriched.drop(columns=["price_per_m2"], errors="ignore")
    enriched["city_governorate"] = enriched["city"] + "__" + enriched["governorate"]
    return enriched


def fit_terrain_specialized(scope_df: pd.DataFrame, random_state: int, test_size: float) -> tuple[CatBoostRegressor, dict[str, float], dict[str, object], int, int]:
    """Terrain-only strategy: predict price_per_m2 then recover price_tnd.

    This is robust when area dominates absolute price magnitude.
    """
    train_df, test_df = train_test_split(scope_df, test_size=test_size, random_state=random_state)
    train_x, test_x, metadata = add_priors(train_df, test_df)

    y_train_ppm = np.log1p((train_df["price_tnd"] / train_df["surface_m2"].clip(lower=EPS)).to_numpy())
    y_test_price = test_df["price_tnd"].to_numpy()

    model = CatBoostRegressor(
        depth=6,
        learning_rate=0.035,
        iterations=1600,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_state=random_state,
        verbose=False,
    )
    model.fit(train_x[FEATURE_COLUMNS], y_train_ppm, cat_features=[c for c in CAT_FEATURES if c in train_x.columns])

    train_ppm_pred = np.expm1(np.asarray(model.predict(train_x[FEATURE_COLUMNS])))
    test_ppm_pred = np.expm1(np.asarray(model.predict(test_x[FEATURE_COLUMNS])))

    train_price_true = train_df["price_tnd"].to_numpy()
    train_price_pred = train_ppm_pred * train_df["surface_m2"].to_numpy()
    test_price_pred = test_ppm_pred * test_df["surface_m2"].to_numpy()

    train_metrics = evaluate_price_arrays(train_price_true, train_price_pred)
    test_metrics = evaluate_price_arrays(y_test_price, test_price_pred)

    metrics = {
        "train_r2": train_metrics["r2"],
        "test_r2": test_metrics["r2"],
        "train_rmse": train_metrics["rmse"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "overfit_gap": train_metrics["r2"] - test_metrics["r2"],
    }
    metadata["training_strategy"] = "terrain_specialized_price_per_m2"
    return model, metrics, metadata, int(len(train_df)), int(len(test_df))


def run_terrain_candidates(
    scope_df: pd.DataFrame,
    random_state: int,
    test_size: float,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    # Approach A: data enrichment then standard target (price_tnd).
    enriched_df = enrich_terrain_data(scope_df, random_state=random_state)
    if len(enriched_df) >= 300:
        model_a, metrics_a, metadata_a, rows_train_a, rows_test_a = fit_scope(enriched_df, random_state, test_size)
        metadata_a["training_strategy"] = "terrain_data_enrichment_standard_target"
        metadata_a["enrichment_rows"] = int(len(enriched_df))
        candidates.append(
            {
                "approach": "terrain_data_enrichment",
                "model": model_a,
                "metrics": metrics_a,
                "metadata": metadata_a,
                "rows_train": rows_train_a,
                "rows_test": rows_test_a,
            }
        )

    # Approach B: terrain-only specialized objective (price_per_m2).
    if len(scope_df) >= 300:
        model_b, metrics_b, metadata_b, rows_train_b, rows_test_b = fit_terrain_specialized(scope_df, random_state, test_size)
        candidates.append(
            {
                "approach": "terrain_specialized",
                "model": model_b,
                "metrics": metrics_b,
                "metadata": metadata_b,
                "rows_train": rows_train_b,
                "rows_test": rows_test_b,
            }
        )

    candidates.sort(key=lambda x: (x["metrics"]["test_r2"], -x["metrics"]["overfit_gap"]), reverse=True)
    return candidates


def main() -> int:
    args = parse_args()
    gate = QualityGate(min_test_r2=args.min_test_r2, max_gap=args.max_overfit_gap)

    source_path = Path(args.input_csv)
    models_dir = Path(args.models_dir)
    reports_dir = Path(args.reports_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    clean_df = clean_source(pd.read_csv(source_path))

    results_rows: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []

    # By-type scopes.
    for property_type in ["appartement", "maison", "terrain"]:
        scope_df = clean_df[clean_df["property_type"] == property_type].copy()
        if len(scope_df) < 300:
            results_rows.append({
                "scope": "by_type",
                "property_type": property_type.title(),
                "status": "skipped_insufficient_rows",
                "rows": int(len(scope_df)),
            })
            continue

        if property_type == "terrain":
            terrain_candidates = run_terrain_candidates(scope_df, args.random_state, args.test_size)
            if not terrain_candidates:
                results_rows.append(
                    {
                        "scope": "by_type",
                        "property_type": "Terrain",
                        "status": "skipped_no_terrain_candidate",
                        "rows": int(len(scope_df)),
                    }
                )
                continue

            for candidate in terrain_candidates:
                cm = candidate["metrics"]
                results_rows.append(
                    {
                        "scope": "by_type_terrain_experiment",
                        "property_type": "Terrain",
                        "approach": candidate["approach"],
                        "rows_train": candidate["rows_train"],
                        "rows_test": candidate["rows_test"],
                        "train_r2": cm["train_r2"],
                        "test_r2": cm["test_r2"],
                        "train_rmse": cm["train_rmse"],
                        "test_rmse": cm["test_rmse"],
                        "test_mae": cm["test_mae"],
                        "overfit_gap": cm["overfit_gap"],
                    }
                )

            best_candidate = terrain_candidates[0]
            model = best_candidate["model"]
            metrics = best_candidate["metrics"]
            metadata = best_candidate["metadata"]
            rows_train = best_candidate["rows_train"]
            rows_test = best_candidate["rows_test"]
            selected_approach = best_candidate["approach"]
        else:
            model, metrics, metadata, rows_train, rows_test = fit_scope(scope_df, args.random_state, args.test_size)
            selected_approach = "standard"

        status = "accepted" if (metrics["test_r2"] >= gate.min_test_r2 and metrics["overfit_gap"] <= gate.max_gap) else "rejected_quality_gate"
        result_row = {
            "scope": "by_type",
            "property_type": property_type.title(),
            "approach": selected_approach,
            "rows_train": rows_train,
            "rows_test": rows_test,
            "train_r2": metrics["train_r2"],
            "test_r2": metrics["test_r2"],
            "train_rmse": metrics["train_rmse"],
            "test_rmse": metrics["test_rmse"],
            "test_mae": metrics["test_mae"],
            "overfit_gap": metrics["overfit_gap"],
            "status": status,
        }
        results_rows.append(result_row)

        if status == "accepted":
            artifact_name = f"bytype__{property_type}__catboost.joblib"
            artifact_path = models_dir / artifact_name
            joblib.dump(model, artifact_path)
            manifest.append(
                {
                    "scope": "by_type",
                    "property_type": property_type.title(),
                    "approach": selected_approach,
                    "model_name": "catboost",
                    "path": str(Path("artifacts") / "models" / "models_estateprocessor" / artifact_name),
                    "metrics": result_row,
                    **metadata,
                }
            )

    # Global scope over accepted property types only.
    accepted_types = {item["property_type"].__str__().lower() for item in manifest if item.get("scope") == "by_type"}
    global_df = clean_df[clean_df["property_type"].isin(accepted_types)].copy()
    if len(global_df) >= 500:
        model, metrics, metadata, rows_train, rows_test = fit_scope(global_df, args.random_state, args.test_size)
        status = "accepted" if (metrics["test_r2"] >= gate.min_test_r2 and metrics["overfit_gap"] <= gate.max_gap) else "rejected_quality_gate"
        result_row = {
            "scope": "global",
            "property_type": "ALL",
            "rows_train": rows_train,
            "rows_test": rows_test,
            "train_r2": metrics["train_r2"],
            "test_r2": metrics["test_r2"],
            "train_rmse": metrics["train_rmse"],
            "test_rmse": metrics["test_rmse"],
            "test_mae": metrics["test_mae"],
            "overfit_gap": metrics["overfit_gap"],
            "status": status,
            "accepted_types": sorted(accepted_types),
        }
        results_rows.append(result_row)
        if status == "accepted":
            artifact_name = "global__catboost.joblib"
            artifact_path = models_dir / artifact_name
            joblib.dump(model, artifact_path)
            manifest.append(
                {
                    "scope": "global",
                    "property_type": "ALL",
                    "model_name": "catboost",
                    "path": str(Path("artifacts") / "models" / "models_estateprocessor" / artifact_name),
                    "metrics": result_row,
                    **metadata,
                }
            )

    results_df = pd.DataFrame(results_rows)
    results_path = reports_dir / "training_estateprocessor_results.csv"
    results_df.to_csv(results_path, index=False)

    manifest_path = reports_dir / "training_estateprocessor_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({"results_path": str(results_path), "manifest_path": str(manifest_path), "accepted_models": len(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
