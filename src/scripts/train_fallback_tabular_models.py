"""Train non-leaky fallback CatBoost models for EstateMind serving."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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

CATEGORICAL_COLUMNS = ["transaction_type", "property_type", "governorate", "city", "city_governorate"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fallback CatBoost tabular models for EstateMind.")
    parser.add_argument("--input-csv", default="data/csv/preprocessed/final_listings_preprocessed.csv")
    parser.add_argument("--output-dir", default="artifacts/models/fallback_tabular")
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def prepare_subset(df: pd.DataFrame, property_type: str) -> pd.DataFrame:
    out = df.copy()
    out["transaction_type"] = out["transaction_type"].astype(str).str.strip().str.lower()
    out["property_type"] = out["property_type"].astype(str).str.strip().str.title()
    out = out[(out["transaction_type"] == "sale") & (out["property_type"] == property_type)].copy()
    out["price_tnd"] = pd.to_numeric(out["price_tnd"], errors="coerce")
    for col in ["surface_m2", "rooms", "bedrooms", "bathrooms", "latitude", "longitude"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out[(out["price_tnd"] > 0)].dropna(subset=["governorate", "city", "surface_m2"])
    if out.empty:
        return out
    lo, hi = out["price_tnd"].quantile([0.01, 0.99])
    out = out[(out["price_tnd"] >= lo) & (out["price_tnd"] <= hi)].copy()
    out["city"] = out["city"].astype(str).str.strip().str.lower()
    out["governorate"] = out["governorate"].astype(str).str.strip().str.lower()
    out["property_type"] = out["property_type"].astype(str).str.strip().str.lower()
    out["city_governorate"] = out["city"] + "__" + out["governorate"]
    return out


def build_features(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    priors_city = (train_df.groupby("city_governorate")["price_tnd"].median() / train_df.groupby("city_governorate")["surface_m2"].median().replace({0: np.nan}))
    priors_city = priors_city.replace([np.inf, -np.inf], np.nan).dropna().to_dict()
    priors_gov = (train_df.groupby("governorate")["price_tnd"].median() / train_df.groupby("governorate")["surface_m2"].median().replace({0: np.nan}))
    priors_gov = priors_gov.replace([np.inf, -np.inf], np.nan).dropna().to_dict()
    global_price_m2 = float(np.nanmedian(list(priors_city.values()))) if priors_city else float((train_df["price_tnd"] / train_df["surface_m2"].replace({0: np.nan})).median())

    medians = {
        col: float(train_df[col].median())
        for col in ["surface_m2", "rooms", "bedrooms", "bathrooms", "latitude", "longitude"]
    }

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for col, value in medians.items():
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(value)
        out["local_avg_price_m2"] = out["city_governorate"].map(priors_city).fillna(global_price_m2)
        out["gov_avg_price_m2"] = out["governorate"].map(priors_gov).fillna(global_price_m2)
        out["size_x_local_price"] = out["surface_m2"] * out["local_avg_price_m2"]
        return out[FEATURE_COLUMNS].copy()

    metadata = {
        "priors": {
            "city_governorate_price_m2": {str(k): float(v) for k, v in priors_city.items()},
            "governorate_price_m2": {str(k): float(v) for k, v in priors_gov.items()},
            "global_price_m2": float(global_price_m2),
        },
        "fill_values": medians,
    }
    return transform(train_df), transform(eval_df), metadata


def metrics_on_price(y_true_log: pd.Series, y_pred_log: np.ndarray) -> dict[str, float]:
    y_true = np.expm1(y_true_log.to_numpy())
    y_pred = np.expm1(np.asarray(y_pred_log))
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_one(df: pd.DataFrame, property_type: str, output_dir: Path, test_size: float) -> dict | None:
    subset = prepare_subset(df, property_type)
    if len(subset) < 300:
        return None

    train_df, test_df = train_test_split(subset, test_size=test_size, random_state=42)
    X_train, X_test, metadata = build_features(train_df, test_df)
    y_train = np.log1p(train_df["price_tnd"])
    y_test = np.log1p(test_df["price_tnd"])

    model = CatBoostRegressor(
        depth=6,
        learning_rate=0.05,
        iterations=500,
        l2_leaf_reg=5,
        loss_function="RMSE",
        verbose=False,
        random_state=42,
    )
    model.fit(X_train, y_train, cat_features=[col for col in CATEGORICAL_COLUMNS if col in X_train.columns])
    pred = model.predict(X_test)
    metrics = metrics_on_price(y_test, pred)

    artifact_name = f"{property_type.lower()}__sale__catboost.joblib"
    artifact_path = output_dir / artifact_name
    joblib.dump(model, artifact_path)
    return {
        "artifact": artifact_name,
        "model_name": "catboost",
        "scope": property_type.lower(),
        "feature_columns": FEATURE_COLUMNS,
        "cat_features": CATEGORICAL_COLUMNS,
        "metrics": metrics,
        **metadata,
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
    }


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    manifest = {"models": {}}
    for property_type in ["Appartement", "Maison", "Terrain"]:
        spec = train_one(df, property_type, output_dir, args.test_size)
        if spec is not None:
            manifest["models"][f"{property_type.lower()}__sale"] = spec

    if not manifest["models"]:
        raise RuntimeError("No fallback models were trained")

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
