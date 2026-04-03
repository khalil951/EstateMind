from src.nlp.sentiment_service import DescriptionSentimentService


class _StubPipeline:
    def __call__(self, text: str, truncation: bool = True):
        return [{"label": "positive", "score": 0.91}]


class _StubTfidfModel:
    def predict(self, rows):
        return ["negative"]

    def predict_proba(self, rows):
        return [[0.72, 0.18, 0.10]]


def test_sentiment_service_uses_transformer_when_available() -> None:
    service = DescriptionSentimentService(transformer_root="artifacts/test_assets/missing_transformer", tfidf_path="artifacts/test_assets/missing.joblib")
    service._pipeline = _StubPipeline()
    service._tfidf_error = "missing tfidf artifact"
    result = service.analyze("Tres bon quartier et appartement lumineux")
    assert result["sentiment_mode"] == "transformer_fallback"
    assert result["description_sentiment_label"] == "positive"


def test_sentiment_service_prefers_tfidf_when_available() -> None:
    service = DescriptionSentimentService(transformer_root="artifacts/test_assets/missing_transformer", tfidf_path="artifacts/test_assets/missing.joblib")
    service._tfidf_model = _StubTfidfModel()
    service._pipeline = _StubPipeline()
    result = service.analyze("Description neutre pour tester le fallback")
    assert result["sentiment_mode"] == "tfidf_primary"
    assert result["description_sentiment_label"] == "negative"


def test_sentiment_service_returns_some_mode_with_real_artifacts() -> None:
    service = DescriptionSentimentService()
    result = service.analyze("Description neutre pour tester le fallback")
    assert result["sentiment_mode"] in {"tfidf_primary", "transformer_fallback", "neutral_fallback"}
    assert "description_sentiment_label" in result
