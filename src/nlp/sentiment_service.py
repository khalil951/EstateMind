"""Hybrid description sentiment runtime for EstateMind.

The service prefers the saved TF-IDF sentiment model for fast, deterministic
serving and falls back to the locally exported DistilBERT checkpoints when
the classical artifact is unavailable. Both paths normalize into the same
compact output schema so the valuation pipeline can reason about sentiment
quality without caring which runtime produced it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    AutoModelForSequenceClassification = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    pipeline = None  # type: ignore[assignment]


class DescriptionSentimentService:
    """Run description sentiment inference with TF-IDF-first fallback logic."""

    def __init__(
        self,
        transformer_root: str | Path | None = None,
        tfidf_path: str | Path | None = None,
        sentiment_report_path: str | Path | None = None,
    ) -> None:
        root = Path(__file__).resolve().parents[2]
        self.transformer_root = Path(transformer_root) if transformer_root else root / "artifacts" / "models" / "distilbert_sentiment"
        self.tfidf_path = Path(tfidf_path) if tfidf_path else root / "artifacts" / "models" / "tfidf_char_sentiment.joblib"
        self.sentiment_report_path = (
            Path(sentiment_report_path)
            if sentiment_report_path
            else root / "artifacts" / "reports" / "nlp_sentiment" / "best_sentiment_model_report.json"
        )
        self._pipeline: Any | None = None
        self._pipeline_error: str | None = None
        self._tfidf_model: Any | None = None
        self._tfidf_error: str | None = None

    def _latest_checkpoint(self) -> Path | None:
        if self.transformer_root.is_file():
            return self.transformer_root
        if not self.transformer_root.exists():
            return None
        checkpoints = sorted(
            [p for p in self.transformer_root.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
        )
        return checkpoints[-1] if checkpoints else self.transformer_root

    def _ensure_pipeline(self) -> Any | None:
        if self._pipeline is not None or self._pipeline_error is not None:
            return self._pipeline
        checkpoint = self._latest_checkpoint()
        if checkpoint is None or pipeline is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
            self._pipeline_error = "transformer checkpoint unavailable"
            return None
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True, use_fast=False)
            model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint), local_files_only=True)
            self._pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
        except Exception as exc:  # pragma: no cover
            self._pipeline_error = str(exc)
        return self._pipeline

    def _ensure_tfidf(self) -> Any | None:
        if self._tfidf_model is not None or self._tfidf_error is not None:
            return self._tfidf_model
        if not self.tfidf_path.exists():
            self._tfidf_error = f"missing tfidf artifact: {self.tfidf_path}"
            return None
        try:
            self._tfidf_model = joblib.load(self.tfidf_path)
        except Exception as exc:  # pragma: no cover
            self._tfidf_error = str(exc)
        return self._tfidf_model

    @staticmethod
    def _label_score_to_polarity(label: str) -> float:
        norm = str(label).strip().lower()
        mapping = {"negative": 0.0, "neutral": 0.5, "positive": 1.0}
        return mapping.get(norm, 0.5)

    def _primary_runtime_order(self) -> tuple[str, str]:
        """Return preferred runtime order using persisted benchmark evidence."""

        selected_model = "tfidf_char"
        if self.sentiment_report_path.exists():
            try:
                payload = json.loads(self.sentiment_report_path.read_text(encoding="utf-8"))
                selected_model = str(payload.get("selected_model", "tfidf_char")).strip().lower()
            except Exception:
                selected_model = "tfidf_char"
        if selected_model.startswith("distilbert"):
            return "transformer", "tfidf"
        return "tfidf", "transformer"

    def _predict_with_tfidf(self, text: str) -> dict[str, Any] | None:
        tfidf_model = self._ensure_tfidf()
        if tfidf_model is None:
            return None
        pred = str(tfidf_model.predict([text])[0]).strip().lower()
        score = 0.0
        if hasattr(tfidf_model, "predict_proba"):
            try:
                probs = tfidf_model.predict_proba([text])[0]
                score = float(np.max(probs))
            except Exception:  # pragma: no cover
                score = 0.0
        return {
            "description_sentiment": self._label_score_to_polarity(pred),
            "description_sentiment_label": pred,
            "description_sentiment_score": round(score, 4),
            "sentiment_mode": "tfidf_primary",
            "sentiment_warning": "",
        }

    def _predict_with_transformer(self, text: str) -> dict[str, Any] | None:
        classifier = self._ensure_pipeline()
        if classifier is None:
            return None
        try:
            output = classifier(text, truncation=True)
            row = output[0] if isinstance(output, list) else output
            label = str(row.get("label", "neutral")).strip().lower()
            score = float(row.get("score", 0.0))
            return {
                "description_sentiment": self._label_score_to_polarity(label),
                "description_sentiment_label": label,
                "description_sentiment_score": round(score, 4),
                "sentiment_mode": "transformer_fallback",
                "sentiment_warning": self._tfidf_error or "",
            }
        except Exception:  # pragma: no cover
            self._pipeline_error = "transformer_inference_failed"
            return None

    def analyze(self, description: str) -> dict[str, Any]:
        """Analyze listing text sentiment and return normalized serving metadata."""

        text = str(description or "").strip()
        if not text:
            return {
                "description_sentiment": 0.5,
                "description_sentiment_label": "neutral",
                "description_sentiment_score": 0.0,
                "sentiment_mode": "neutral_fallback",
                "sentiment_warning": "empty_description",
            }

        primary, secondary = self._primary_runtime_order()
        first = self._predict_with_tfidf(text) if primary == "tfidf" else self._predict_with_transformer(text)
        if first is not None:
            return first
        second = self._predict_with_transformer(text) if secondary == "transformer" else self._predict_with_tfidf(text)
        if second is not None:
            return second

        return {
            "description_sentiment": 0.5,
            "description_sentiment_label": "neutral",
            "description_sentiment_score": 0.0,
            "sentiment_mode": "neutral_fallback",
            "sentiment_warning": self._tfidf_error or self._pipeline_error or "no_sentiment_model",
        }
