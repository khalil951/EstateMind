"""Dynamic serving bundles for EstateMind valuation artifacts.

The saved estate estimators in the repository are not yet packaged with the
full preprocessing context required for direct online serving. This module
reconstructs the minimum processor state needed for inference from the
preprocessed listings dataset and wraps compatible estimators behind a stable
``predict`` interface.

The current implementation supports CatBoost artifacts with explicit feature
names. Non-CatBoost artifacts are reported as incompatible until their
serving schema is exported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from sklearn.neighbors import BallTree

if TYPE_CHECKING:
    from src.inference.model_registry import ModelHandle


REFERENCE_DATASET_CANDIDATES = (
    Path("data/csv/preprocessed/final_listings_preprocessed.csv"),
    Path("data/csv/final_listings_preprocessed.csv"),
)

CATEGORICAL_COLUMNS = ("city", "governorate", "property_type", "transaction_type")
NUMERIC_COLUMNS = (
    "price_tnd",
    "surface_m2",
    "price_per_m2",
    "rooms",
    "bedrooms",
    "bathrooms",
    "latitude",
    "longitude",
)


@dataclass
class PredictionResult:
    """Outcome of one bundle-backed valuation inference."""

    estimated_price: int
    price_per_m2: int
    prediction_mode: str
    warnings: list[str]
    model_info: dict[str, Any]
    feature_frame: pd.DataFrame | None = None
    uncertainty_reasons: list[str] = field(default_factory=list)
    ood_flags: list[str] = field(default_factory=list)


class _ServingProcessor:
    """Rebuild the CatBoost preprocessing path used during training."""

    def __init__(self, reference_df: pd.DataFrame, *, radius_km: float = 1.0, n_geo_clusters: int = 20) -> None:
        self.radius_km = radius_km
        self.n_geo_clusters = n_geo_clusters
        self.reference_df = self._clean(reference_df)
        self.numeric_columns = [col for col in NUMERIC_COLUMNS if col in self.reference_df.columns and self.reference_df[col].notna().any()]
        self.imputer = KNNImputer(n_neighbors=5, weights="distance")
        if self.numeric_columns:
            self.reference_df[self.numeric_columns] = self.imputer.fit_transform(self.reference_df[self.numeric_columns])

        self.categorical_fill: dict[str, str] = {}
        for col in list(CATEGORICAL_COLUMNS) + ["city_governorate"]:
            if col in self.reference_df.columns:
                mode = self.reference_df[col].mode(dropna=True)
                fill_val = str(mode.iat[0]) if not mode.empty else "unknown"
                self.categorical_fill[col] = fill_val
                self.reference_df[col] = self.reference_df[col].fillna(fill_val)

        coords = self.reference_df[["latitude", "longitude"]].apply(pd.to_numeric, errors="coerce")
        valid = coords.notna().all(axis=1)
        coords_valid = coords[valid]
        if len(coords_valid) == 0:
            raise ValueError("Reference dataset does not contain any usable latitude/longitude rows")

        self.kmeans = KMeans(
            n_clusters=min(self.n_geo_clusters, len(coords_valid)),
            random_state=42,
            n_init=10,
        ).fit(coords_valid.values)
        self.tree = BallTree(np.radians(coords_valid.values), metric="haversine")

        self.train_prices = pd.to_numeric(self.reference_df["price_tnd"], errors="coerce").fillna(0.0).to_numpy()
        self.global_price = float(np.nanmean(self.train_prices)) if len(self.train_prices) else 0.0

        ppm2 = self.reference_df["price_tnd"] / self.reference_df["surface_m2"].replace({0: np.nan})
        local_key = self.reference_df["city_governorate"] if "city_governorate" in self.reference_df.columns else self.reference_df["city"]
        self.local_avg_price_m2 = (
            pd.DataFrame({"key": local_key, "ppm2": ppm2})
            .dropna(subset=["key", "ppm2"])
            .groupby("key", dropna=True)["ppm2"]
            .median()
            .to_dict()
        )

        self.gov_avg_price_m2 = (
            pd.DataFrame({"key": self.reference_df["governorate"], "ppm2": ppm2})
            .dropna(subset=["key", "ppm2"])
            .groupby("key", dropna=True)["ppm2"]
            .median()
            .to_dict()
        )

        city_geo = (
            self.reference_df.dropna(subset=["city", "latitude", "longitude"])
            .groupby("city", dropna=True)[["latitude", "longitude"]]
            .median()
            .reset_index()
        )
        self.city_geo_lookup = {
            str(row["city"]): (float(row["latitude"]), float(row["longitude"]))
            for _, row in city_geo.iterrows()
        }

        self.quantiles: dict[str, tuple[float, float]] = {}
        for col in ("price_tnd", "surface_m2"):
            if col in self.reference_df.columns and self.reference_df[col].notna().any():
                self.quantiles[col] = (
                    float(self.reference_df[col].quantile(0.01)),
                    float(self.reference_df[col].quantile(0.99)),
                )

    @staticmethod
    def _to_num(value: Any) -> float:
        if pd.isna(value):
            return float("nan")
        text = str(value).upper().replace("DT", "").replace("TND", "")
        text = re.sub(r"\s+", "", text).replace(",", ".")
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        return float(match.group()) if match else float("nan")

    @classmethod
    def _clean(cls, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.drop(columns=[c for c in ("record_id", "description", "location_raw") if c in out.columns], inplace=True, errors="ignore")
        for col in NUMERIC_COLUMNS:
            if col in out.columns:
                out[col] = out[col].map(cls._to_num)
        for col in CATEGORICAL_COLUMNS:
            if col in out.columns:
                out[col] = out[col].astype(str).str.strip().str.lower().replace({"nan": np.nan, "": np.nan})
        if "city" in out.columns and "governorate" in out.columns:
            out["city_governorate"] = out["city"].fillna("unknown") + "__" + out["governorate"].fillna("unknown")
        return out

    def transform_request(self, row: dict[str, Any]) -> tuple[pd.DataFrame, list[str], list[str]]:
        """Transform one request row into CatBoost-ready features."""

        warnings: list[str] = []
        ood_flags: list[str] = []
        frame = pd.DataFrame([row])
        frame = self._clean(frame)

        if frame["city"].iat[0] not in self.city_geo_lookup and frame.get("latitude", pd.Series([np.nan])).isna().iat[0]:
            ood_flags.append("unknown_city")
        if "surface_m2" in self.quantiles:
            lo, hi = self.quantiles["surface_m2"]
            surface = float(frame["surface_m2"].iat[0])
            if surface < lo or surface > hi:
                ood_flags.append("surface_out_of_range")

        if "latitude" not in frame.columns:
            frame["latitude"] = np.nan
        if "longitude" not in frame.columns:
            frame["longitude"] = np.nan
        if pd.isna(frame["latitude"].iat[0]) or pd.isna(frame["longitude"].iat[0]):
            coords = self.city_geo_lookup.get(str(frame["city"].iat[0]))
            if coords:
                frame.at[0, "latitude"] = coords[0]
                frame.at[0, "longitude"] = coords[1]
            else:
                warnings.append("geo_lookup_missing")

        for col in self.numeric_columns:
            if col not in frame.columns:
                frame[col] = np.nan
        if self.numeric_columns:
            frame[self.numeric_columns] = self.imputer.transform(frame[self.numeric_columns])

        for col, fill_val in self.categorical_fill.items():
            if col in frame.columns:
                frame[col] = frame[col].fillna(fill_val)
            elif col != "city_governorate":
                frame[col] = fill_val

        frame["city_governorate"] = frame["city"].fillna("unknown") + "__" + frame["governorate"].fillna("unknown")
        frame["local_avg_price_m2"] = frame["city_governorate"].map(self.local_avg_price_m2)
        if frame["local_avg_price_m2"].isna().any():
            frame["local_avg_price_m2"] = frame["local_avg_price_m2"].fillna(np.nanmedian(list(self.local_avg_price_m2.values())) if self.local_avg_price_m2 else 0.0)
            warnings.append("local_price_prior_fallback")
            ood_flags.append("local_price_prior_fallback")

        frame["gov_avg_price_m2"] = frame["governorate"].map(self.gov_avg_price_m2)
        if frame["gov_avg_price_m2"].isna().any():
            frame["gov_avg_price_m2"] = frame["gov_avg_price_m2"].fillna(
                np.nanmedian(list(self.gov_avg_price_m2.values())) if self.gov_avg_price_m2 else 0.0
            )
            warnings.append("gov_price_prior_fallback")
            ood_flags.append("gov_price_prior_fallback")

        coords = frame[["latitude", "longitude"]].apply(pd.to_numeric, errors="coerce")
        valid = coords.notna().all(axis=1)
        frame["geo_cluster_id"] = np.nan
        frame["avg_price_1km_radius"] = np.nan
        frame["listings_density_in_area"] = np.nan
        if valid.any():
            values = coords[valid].values
            frame.loc[valid, "geo_cluster_id"] = self.kmeans.predict(values)
            query_rad = np.radians(values)
            neighbors = self.tree.query_radius(query_rad, r=self.radius_km / 6371.0)
            avg_prices: list[float] = []
            densities: list[float] = []
            area = np.pi * (self.radius_km**2)
            for idx in neighbors:
                idx_list = idx.tolist()
                if not idx_list:
                    avg_prices.append(self.global_price)
                    densities.append(0.0)
                else:
                    avg_prices.append(float(np.nanmean(self.train_prices[idx_list])))
                    densities.append(len(idx_list) / area)
            frame.loc[valid, "avg_price_1km_radius"] = avg_prices
            frame.loc[valid, "listings_density_in_area"] = densities
        else:
            warnings.append("spatial_features_missing")

        frame["size_x_local_price"] = frame["surface_m2"] * frame["local_avg_price_m2"]
        return frame, warnings, ood_flags


@dataclass
class InferenceBundle:
    """Serve-compatible wrapper around one estimator and its processor state."""

    estimator: Any
    model_name: str
    property_scope: str
    reference_rows: int
    feature_columns: list[str]
    processor: _ServingProcessor
    source_path: Path
    uses_proxy_price_features: bool = False
    version: str = "estatebundle-v1"

    @classmethod
    def from_handle(cls, handle: "ModelHandle", reference_df: pd.DataFrame) -> "InferenceBundle":
        """Build a serving bundle from a compatible model handle."""

        estimator = handle.estimator
        if estimator is None:
            raise ValueError("Estimator must be loaded before bundle creation")
        feature_columns = list(getattr(estimator, "feature_names_", []) or [])
        if not feature_columns:
            raise ValueError("Unsupported artifact: estimator does not expose feature names")
        if handle.model_name.lower() != "catboost":
            raise ValueError("Unsupported artifact: only catboost serving bundles are implemented")

        subset = reference_df.copy()
        if handle.scope == "by_type" and handle.property_type and handle.property_type.upper() != "ALL":
            mask = subset["property_type"].astype(str).str.strip().str.lower() == handle.property_type.strip().lower()
            typed = subset[mask].copy()
            if not typed.empty:
                subset = typed
        processor = _ServingProcessor(subset)
        return cls(
            estimator=estimator,
            model_name=handle.model_name,
            property_scope=handle.property_type,
            reference_rows=int(len(subset)),
            feature_columns=feature_columns,
            processor=processor,
            source_path=handle.path,
            uses_proxy_price_features=("price_tnd" in feature_columns or "price_per_m2" in feature_columns),
        )

    def predict(self, mapped: dict[str, Any], market_context: dict[str, Any]) -> PredictionResult:
        """Run estimator-backed inference for one mapped property request."""

        seed_ppm = max(float(market_context.get("avg_m2", 1450) or 1450), 1.0)
        proxy_price = float(mapped["surface_m2"]) * seed_ppm
        proxy_rooms = int(mapped.get("bedrooms", 0)) + (1 if str(mapped.get("property_type", "")).lower() != "terrain" else 0)
        request_row = {
            "transaction_type": str(mapped.get("transaction_type") or "sale"),
            "property_type": str(mapped["property_type"]),
            "price_tnd": proxy_price,
            "surface_m2": float(mapped["surface_m2"]),
            "price_per_m2": seed_ppm,
            "rooms": float(mapped.get("rooms") or proxy_rooms),
            "bedrooms": float(mapped.get("bedrooms", 0)),
            "bathrooms": float(mapped.get("bathrooms", 0)),
            "governorate": str(mapped["governorate"]),
            "city": str(mapped["city"]),
            "latitude": float(mapped["latitude"]) if mapped.get("latitude") is not None else np.nan,
            "longitude": float(mapped["longitude"]) if mapped.get("longitude") is not None else np.nan,
        }

        transformed, warnings, ood_flags = self.processor.transform_request(request_row)
        missing_columns = [col for col in self.feature_columns if col not in transformed.columns]
        if missing_columns:
            raise ValueError(f"schema mismatch: missing transformed columns {missing_columns}")
        features = transformed[self.feature_columns].copy()
        pred_log = float(self.estimator.predict(features)[0])
        pred_price = int(round(float(np.expm1(pred_log))))
        pred_ppm = int(round(pred_price / max(float(mapped["surface_m2"]), 1.0)))

        if self.uses_proxy_price_features:
            warnings.append("proxy_price_features_used")

        uncertainty_reasons = []
        if ood_flags:
            uncertainty_reasons.extend([f"ood:{flag}" for flag in ood_flags])
        if warnings:
            uncertainty_reasons.extend(warnings)

        return PredictionResult(
            estimated_price=max(pred_price, 1),
            price_per_m2=max(pred_ppm, 1),
            prediction_mode="model",
            warnings=sorted(set(warnings)),
            model_info={
                "bundle_version": self.version,
                "model_name": self.model_name,
                "property_scope": self.property_scope,
                "source_path": str(self.source_path),
                "reference_rows": self.reference_rows,
            },
            feature_frame=features,
            uncertainty_reasons=sorted(set(uncertainty_reasons)),
            ood_flags=sorted(set(ood_flags)),
        )


def load_reference_dataset(reference_path: str | Path | None = None) -> pd.DataFrame:
    """Load the preprocessed listings dataset used to reconstruct processor state."""

    root = Path(__file__).resolve().parents[2]
    if reference_path is not None:
        path = Path(reference_path)
        resolved = path if path.is_absolute() else root / path
        if not resolved.exists():
            raise FileNotFoundError(f"Reference dataset not found: {resolved}")
        return pd.read_csv(resolved)

    for rel_path in REFERENCE_DATASET_CANDIDATES:
        resolved = root / rel_path
        if resolved.exists():
            return pd.read_csv(resolved)
    raise FileNotFoundError("Could not locate a preprocessed listings dataset for serving bundle reconstruction")
