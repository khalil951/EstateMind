from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


SURFACE_PATTERNS = (
    r"surface\s*[:\-]?\s*(\d+(?:[.,]\d+)?)\s*m2",
    r"(\d+(?:[.,]\d+)?)\s*m2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark text-only price regression on raw price, log(price), and price_per_m2 targets."
    )
    parser.add_argument("--train-csv", default="data/nlp/prepared/model1_train.csv")
    parser.add_argument("--val-csv", default="data/nlp/prepared/model1_val.csv")
    parser.add_argument("--test-csv", default="data/nlp/prepared/model1_test.csv")
    parser.add_argument("--output-dir", default="artifacts/reports/nlp_description_benchmark")
    return parser.parse_args()


def parse_surface_m2(text: str) -> float | None:
    blob = str(text or "").lower()
    for pattern in SURFACE_PATTERNS:
        match = re.search(pattern, blob)
        if match:
            raw = match.group(1).replace(",", ".")
            try:
                value = float(raw)
            except ValueError:
                continue
            if value > 0:
                return value
    return None


def load_split(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()
    df["clean_text"] = df["clean_text"].fillna("").astype(str)
    df["target_price_tnd"] = pd.to_numeric(df["target_price_tnd"], errors="coerce")
    raw_text = df["raw_text"].fillna("").astype(str)
    clean_text = df["clean_text"].fillna("").astype(str)
    df["surface_m2"] = [
        parse_surface_m2(raw) if parse_surface_m2(raw) is not None else parse_surface_m2(clean)
        for raw, clean in zip(raw_text.tolist(), clean_text.tolist())
    ]
    df["surface_m2"] = pd.to_numeric(df["surface_m2"], errors="coerce")
    df["price_per_m2"] = df["target_price_tnd"] / df["surface_m2"]
    return df


def get_xy_price(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    X = df["clean_text"].fillna("").astype(str)
    y = pd.to_numeric(df["target_price_tnd"], errors="coerce")
    mask = X.str.len().gt(0) & y.notna() & y.gt(0)
    return X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)


def get_xy_log_price(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    X, y = get_xy_price(df)
    return X, np.log1p(y).reset_index(drop=True)


def get_xy_price_per_m2(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    X = df["clean_text"].fillna("").astype(str)
    ppm = pd.to_numeric(df["price_per_m2"], errors="coerce")
    surface = pd.to_numeric(df["surface_m2"], errors="coerce")
    mask = X.str.len().gt(0) & ppm.notna() & ppm.gt(0) & surface.notna() & surface.gt(0)
    return (
        X[mask].reset_index(drop=True),
        ppm[mask].reset_index(drop=True),
        surface[mask].reset_index(drop=True),
    )


def regression_metrics(y_true: pd.Series | np.ndarray, pred: pd.Series | np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "r2": float(r2_score(y_true, pred)),
    }


@dataclass
class TargetSpec:
    name: str
    prepare_train: Callable[[pd.DataFrame], tuple]
    prepare_eval: Callable[[pd.DataFrame], tuple]
    predict_to_price: Callable[[np.ndarray, tuple], np.ndarray]
    compare_target: str


def predict_raw_price(pred: np.ndarray, _: tuple) -> np.ndarray:
    return np.clip(np.asarray(pred, dtype=float), a_min=0.0, a_max=None)


def predict_log_price(pred: np.ndarray, _: tuple) -> np.ndarray:
    return np.clip(np.expm1(np.asarray(pred, dtype=float)), a_min=0.0, a_max=None)


def predict_ppm_to_price(pred: np.ndarray, eval_data: tuple) -> np.ndarray:
    surfaces = np.asarray(eval_data[2], dtype=float)
    pred_ppm = np.clip(np.asarray(pred, dtype=float), a_min=0.0, a_max=None)
    return pred_ppm * surfaces


TARGET_SPECS = {
    "price_tnd": TargetSpec(
        name="price_tnd",
        prepare_train=get_xy_price,
        prepare_eval=get_xy_price,
        predict_to_price=predict_raw_price,
        compare_target="price_tnd",
    ),
    "log_price_tnd": TargetSpec(
        name="log_price_tnd",
        prepare_train=get_xy_log_price,
        prepare_eval=get_xy_log_price,
        predict_to_price=predict_log_price,
        compare_target="price_tnd",
    ),
    "price_per_m2": TargetSpec(
        name="price_per_m2",
        prepare_train=get_xy_price_per_m2,
        prepare_eval=get_xy_price_per_m2,
        predict_to_price=predict_ppm_to_price,
        compare_target="price_tnd",
    ),
}


def baseline_candidates() -> dict[str, Pipeline]:
    return {
        "tfidf_word_ridge": Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000)),
                ("reg", Ridge(alpha=3.0)),
            ]
        ),
        "tfidf_char_ridge": Pipeline(
            [
                ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=80000)),
                ("reg", Ridge(alpha=3.0)),
            ]
        ),
    }


def evaluate_target_strategy(
    model: Pipeline,
    model_name: str,
    target_spec: TargetSpec,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    split_name: str,
) -> dict[str, Any]:
    train_data = target_spec.prepare_train(train_df)
    eval_data = target_spec.prepare_eval(eval_df)
    X_train = train_data[0]
    y_train = train_data[1]
    X_eval = eval_data[0]
    y_eval_transformed = np.asarray(eval_data[1], dtype=float)

    fitted = clone(model)
    fitted.fit(X_train, y_train)
    pred_transformed = np.asarray(fitted.predict(X_eval), dtype=float)
    pred_price = target_spec.predict_to_price(pred_transformed, eval_data)

    if target_spec.name == "price_per_m2":
        actual_price = np.asarray(eval_data[1], dtype=float) * np.asarray(eval_data[2], dtype=float)
        transformed_metrics = regression_metrics(np.asarray(eval_data[1], dtype=float), np.clip(pred_transformed, a_min=0.0, a_max=None))
    elif target_spec.name == "log_price_tnd":
        actual_price = np.expm1(y_eval_transformed)
        transformed_metrics = regression_metrics(y_eval_transformed, pred_transformed)
    else:
        actual_price = y_eval_transformed
        transformed_metrics = regression_metrics(actual_price, pred_price)

    price_metrics = regression_metrics(actual_price, pred_price)
    return {
        "model": model_name,
        "target_strategy": target_spec.name,
        "split": split_name,
        "rows_used": int(len(X_eval)),
        "coverage_ratio": round(float(len(X_eval)) / max(len(eval_df), 1), 4),
        "mae_price": price_metrics["mae"],
        "rmse_price": price_metrics["rmse"],
        "r2_price": price_metrics["r2"],
        "mae_target": transformed_metrics["mae"],
        "rmse_target": transformed_metrics["rmse"],
        "r2_target": transformed_metrics["r2"],
    }


def run_benchmark(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target_spec in TARGET_SPECS.values():
        for model_name, candidate in baseline_candidates().items():
            for split_name, eval_df in (("val", val_df), ("test", test_df)):
                rows.append(
                    evaluate_target_strategy(
                        model=candidate,
                        model_name=model_name,
                        target_spec=target_spec,
                        train_df=train_df,
                        eval_df=eval_df,
                        split_name=split_name,
                    )
                )
    return pd.DataFrame(rows).sort_values(["split", "rmse_price", "mae_price"]).reset_index(drop=True)


def build_summary(metrics_df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"best_by_split": {}, "rows": int(len(metrics_df))}
    for split_name in sorted(metrics_df["split"].unique()):
        split_df = metrics_df.loc[metrics_df["split"] == split_name].sort_values(["rmse_price", "mae_price"]).reset_index(drop=True)
        if split_df.empty:
            continue
        best = split_df.iloc[0].to_dict()
        summary["best_by_split"][split_name] = best
    return summary


def main() -> int:
    args = parse_args()
    train_df = load_split(args.train_csv)
    val_df = load_split(args.val_csv)
    test_df = load_split(args.test_csv)

    metrics_df = run_benchmark(train_df, val_df, test_df)
    summary = build_summary(metrics_df)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "target_strategy_metrics.csv"
    summary_path = output_dir / "target_strategy_summary.json"

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(metrics_df.to_string(index=False))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
