import pandas as pd

from src.prepare_nlp_data import make_group_id
from src.sentiment_template_generator import synthesize_sentiment_metadata


def test_synthesize_sentiment_metadata_generates_balanced_groups() -> None:
    df = synthesize_sentiment_metadata(target_groups_per_label=120, seed=42)

    assert len(df) == 360
    assert set(df["language"]) == {"English", "French"}
    assert set(df["sentiment"]) == {"Negative", "Neutral", "Positive"}

    prepared = pd.DataFrame(
        {
            "language": df["language"].map({"English": "en", "French": "fr"}),
            "sentiment_label": df["sentiment"].str.lower(),
            "group_id": [
                make_group_id(text, "en" if lang == "English" else "fr")
                for text, lang in zip(df["review_text"], df["language"])
            ],
        }
    )

    groups_per_class = (
        prepared.groupby("group_id")["sentiment_label"]
        .agg(lambda s: s.mode().iloc[0])
        .value_counts()
        .to_dict()
    )
    assert groups_per_class == {"negative": 120, "neutral": 120, "positive": 120}
