import pandas as pd

from src.prepare_nlp_data import build_model2_reviews_df, split_grouped_sentiment_df


def test_build_model2_prefers_metadata_and_adds_schema() -> None:
    metadata = pd.DataFrame(
        [
            {"review_text": "Administrative and legal procedures lack transparency.", "sentiment": "Negative", "language": "English"},
            {"review_text": "Prix excessif par rapport aux biens similaires.", "sentiment": "Negative", "language": "French"},
            {"review_text": "Stable environment with balanced supply and demand.", "sentiment": "Neutral", "language": "English"},
        ]
    )

    out = build_model2_reviews_df(
        reviews_txt=["Excellent potential and strong resale value."],
        reviews_metadata_df=metadata,
        lowercase=True,
        min_tokens=3,
        target_groups_per_class=3,
        seed=42,
    )

    assert len(out) >= 9
    assert {"language", "label_source", "group_id", "sentiment_label"}.issubset(out.columns)
    assert {"metadata_label", "synthetic_template_label"}.issubset(set(out["label_source"]))
    assert set(out["language"]) == {"en", "fr"}
    assert set(out["sentiment_label"]) == {"negative", "neutral", "positive"}


def test_split_grouped_sentiment_df_prevents_group_leakage() -> None:
    rows = []
    for label, group_count in [("negative", 4), ("neutral", 4), ("positive", 4)]:
        for g in range(group_count):
            group_id = f"{label}_{g}"
            for idx in range(3):
                rows.append(
                    {
                        "sample_id": f"s_{label}_{g}_{idx}",
                        "source": "reviews_metadata_csv",
                        "raw_text": f"{label} text {g} {idx}",
                        "clean_text": f"{label} text {g} {idx}",
                        "token_count": 4,
                        "char_count": 20,
                        "sentiment_label": label,
                        "language": "en",
                        "label_source": "metadata_label",
                        "group_id": group_id,
                    }
                )
    df = pd.DataFrame(rows)

    train_df, val_df, test_df = split_grouped_sentiment_df(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)

    train_groups = set(train_df["group_id"])
    val_groups = set(val_df["group_id"])
    test_groups = set(test_df["group_id"])

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)
    assert set(val_df["sentiment_label"]) == {"negative", "neutral", "positive"}
    assert set(test_df["sentiment_label"]) == {"negative", "neutral", "positive"}
