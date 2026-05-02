from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    pipeline = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    AutoModelForSequenceClassification = None  # type: ignore[assignment]


LABEL_ORDER = ["negative", "neutral", "positive"]
DEFAULT_MULTILINGUAL_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"


def _hf_cache_kwargs() -> dict[str, str]:
    cache_dir = os.environ.get("TRANSFORMERS_CACHE") or os.environ.get("HF_HOME")
    return {"cache_dir": cache_dir} if cache_dir else {}


def make_xy(df: pd.DataFrame, text_col: str = "clean_text", label_col: str = "sentiment_label") -> tuple[pd.Series, pd.Series]:
    X = df[text_col].fillna("").astype(str)
    y = df[label_col].fillna("neutral").astype(str)
    mask = X.str.len().gt(0)
    return X[mask], y[mask]


def evaluate_predictions(y_true: list[str] | pd.Series, y_pred: list[str] | pd.Series, split_name: str = "eval") -> dict[str, Any]:
    return {
        "split": split_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=LABEL_ORDER, zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", labels=LABEL_ORDER, zero_division=0)),
        "report": classification_report(
            y_true,
            y_pred,
            labels=LABEL_ORDER,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=LABEL_ORDER,
        ).tolist(),
    }


def split_summary(df: pd.DataFrame) -> dict[str, Any]:
    out = {
        "rows": int(len(df)),
        "label_distribution": {},
        "language_distribution": {},
    }
    if "sentiment_label" in df.columns and not df.empty:
        out["label_distribution"] = {str(k): int(v) for k, v in df["sentiment_label"].value_counts().to_dict().items()}
    if "language" in df.columns and not df.empty:
        out["language_distribution"] = {str(k): int(v) for k, v in df["language"].value_counts().to_dict().items()}
    if "group_id" in df.columns and not df.empty:
        out["unique_groups"] = int(df["group_id"].astype(str).nunique())
    return out


def dataset_summary(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
    return {
        "train": split_summary(train_df),
        "val": split_summary(val_df),
        "test": split_summary(test_df),
    }


def build_word_tfidf_model() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]
    )


def build_char_tfidf_model() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)),
        ]
    )


def run_majority_baseline(train_df: pd.DataFrame, eval_df: pd.DataFrame, split_name: str) -> tuple[dict[str, Any], pd.DataFrame]:
    X_eval, y_eval = make_xy(eval_df)
    majority_label = train_df["sentiment_label"].astype(str).mode().iloc[0]
    preds = [majority_label] * len(y_eval)
    metrics = evaluate_predictions(y_eval.tolist(), preds, split_name=split_name)
    preds_df = pd.DataFrame({"text": list(X_eval), "y_true": list(y_eval), "y_pred": preds})
    return metrics, preds_df


def fit_and_predict_model(
    model: Pipeline,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    split_name: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    X_train, y_train = make_xy(train_df)
    X_eval, y_eval = make_xy(eval_df)
    model.fit(X_train, y_train)
    preds = model.predict(X_eval)
    metrics = evaluate_predictions(y_eval.tolist(), preds.tolist(), split_name=split_name)#type: ignore[list-arg]
    preds_df = pd.DataFrame({"text": list(X_eval), "y_true": list(y_eval), "y_pred": list(preds)})
    return metrics, preds_df


def _map_pipeline_label(label: str, score: float) -> str:
    norm = str(label).strip().lower()
    if "label_0" in norm or "negative" in norm or "neg" in norm:
        return "negative"
    if "label_1" in norm or "neutral" in norm:
        return "neutral"
    if "label_2" in norm or "positive" in norm or "pos" in norm:
        return "positive"
    return "neutral" if score < 0.60 else "positive"


def run_multilingual_pipeline_baseline(
    eval_df: pd.DataFrame,
    model_name: str = DEFAULT_MULTILINGUAL_MODEL,
    split_name: str = "eval",
) -> tuple[dict[str, Any], pd.DataFrame]:
    if pipeline is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
        raise ModuleNotFoundError("transformers is required for the pretrained multilingual baseline.")

    X_eval, y_eval = make_xy(eval_df)
    cache_kwargs = _hf_cache_kwargs()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            **cache_kwargs,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            **cache_kwargs,
        )
        classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)  # type: ignore[arg-type]
    except Exception as exc:
        msg = str(exc)
        if "Error parsing line" in msg or "sentencepiece" in msg.lower():
            raise RuntimeError(
                "Tokenizer initialization failed. Restart the notebook kernel after installing `sentencepiece`, "
                "or clear the cached model files for the multilingual model. "
                f"Original error: {msg}"
            ) from exc
        raise

    outputs = classifier(X_eval.tolist(), truncation=True, batch_size=32)
    preds = [_map_pipeline_label(str(item.get("label", "")), float(item.get("score", 0.0))) for item in outputs]
    metrics = evaluate_predictions(y_eval.tolist(), preds, split_name=split_name)
    preds_df = pd.DataFrame({"text": list(X_eval), "y_true": list(y_eval), "y_pred": preds})
    return metrics, preds_df


def _resolve_group_cv_splits(df: pd.DataFrame, label_col: str, group_col: str, max_splits: int = 5) -> int:
    if df.empty or group_col not in df.columns or label_col not in df.columns:
        return 0
    group_labels = (
        df.groupby(group_col, dropna=False)[label_col]
        .agg(lambda s: s.mode().iloc[0])
        .astype(str)
    )
    min_groups_per_class = int(group_labels.value_counts().min()) if not group_labels.empty else 0
    if min_groups_per_class < 2:
        return 0
    return max(2, min(max_splits, min_groups_per_class))


def cross_validate_group_model(
    df: pd.DataFrame,
    model_factory: Callable[[], Pipeline],
    label_col: str = "sentiment_label",
    group_col: str = "group_id",
    max_splits: int = 5,
) -> dict[str, Any]:
    n_splits = _resolve_group_cv_splits(df, label_col=label_col, group_col=group_col, max_splits=max_splits)
    if n_splits == 0:
        return {
            "fold_count": 0,
            "f1_macro_mean": None,
            "f1_macro_std": None,
            "fold_scores": [],
            "status": "insufficient_groups",
        }

    X = df["clean_text"].fillna("").astype(str)
    y = df[label_col].fillna("neutral").astype(str)
    groups = df[group_col].astype(str)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scores: list[float] = []
    for train_idx, test_idx in splitter.split(X.tolist(), y, groups):
        model = model_factory()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        score = f1_score(y.iloc[test_idx], preds, average="macro", labels=LABEL_ORDER, zero_division=0)
        scores.append(float(score))

    return {
        "fold_count": int(n_splits),
        "f1_macro_mean": float(np.mean(scores)),
        "f1_macro_std": float(np.std(scores)),
        "fold_scores": [float(s) for s in scores],
        "status": "ok",
    }


def transformer_readiness_summary(
    df: pd.DataFrame,
    label_col: str = "sentiment_label",
    group_col: str = "group_id",
    min_unique_groups: int = 300,
    min_groups_per_class: int = 100,
) -> dict[str, Any]:
    if df.empty:
        return {
            "ready": False,
            "reason": "empty_dataset",
            "unique_groups": 0,
            "groups_per_class": {},
        }

    group_labels = (
        df.groupby(group_col, dropna=False)[label_col]
        .agg(lambda s: s.mode().iloc[0])
        .astype(str)
    )
    counts = {str(k): int(v) for k, v in group_labels.value_counts().to_dict().items()}
    unique_groups = int(group_labels.shape[0])

    ready = unique_groups >= int(min_unique_groups) and all(
        counts.get(label, 0) >= int(min_groups_per_class) for label in LABEL_ORDER
    )
    reason = "ok" if ready else "insufficient_unique_labeled_groups"
    return {
        "ready": bool(ready),
        "reason": reason,
        "unique_groups": unique_groups,
        "groups_per_class": counts,
        "thresholds": {
            "min_unique_groups": int(min_unique_groups),
            "min_groups_per_class": int(min_groups_per_class),
        },
    }
