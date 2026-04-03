import pandas as pd

from src.sentiment_recovery import (
    build_word_tfidf_model,
    cross_validate_group_model,
    evaluate_predictions,
    transformer_readiness_summary,
)


def test_evaluate_predictions_reports_macro_f1() -> None:
    metrics = evaluate_predictions(
        ["negative", "neutral", "positive"],
        ["negative", "positive", "positive"],
        split_name="toy",
    )

    assert metrics["split"] == "toy"
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    assert metrics["confusion_matrix"][0][0] == 1


def test_cross_validate_group_model_uses_group_folds() -> None:
    rows = []
    for label in ["negative", "neutral", "positive"]:
        for group_idx in range(3):
            group_id = f"{label}_{group_idx}"
            for sample_idx in range(2):
                rows.append(
                    {
                        "clean_text": f"{label} sample {group_idx} {sample_idx}",
                        "sentiment_label": label,
                        "group_id": group_id,
                    }
                )
    df = pd.DataFrame(rows)

    report = cross_validate_group_model(df, model_factory=build_word_tfidf_model, max_splits=3)

    assert report["status"] == "ok"
    assert report["fold_count"] == 3
    assert len(report["fold_scores"]) == 3


def test_transformer_readiness_summary_blocks_small_dataset() -> None:
    df = pd.DataFrame(
        [
            {"clean_text": "negative case", "sentiment_label": "negative", "group_id": "n1"},
            {"clean_text": "neutral case", "sentiment_label": "neutral", "group_id": "u1"},
            {"clean_text": "positive case", "sentiment_label": "positive", "group_id": "p1"},
        ]
    )

    readiness = transformer_readiness_summary(df, min_unique_groups=10, min_groups_per_class=4)

    assert readiness["ready"] is False
    assert readiness["reason"] == "insufficient_unique_labeled_groups"
