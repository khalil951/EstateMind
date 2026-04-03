from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

try:
    from src.sentiment_template_generator import synthesize_sentiment_metadata
except ModuleNotFoundError:  # pragma: no cover
    from sentiment_template_generator import synthesize_sentiment_metadata

try:
    from transformers import pipeline
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    pipeline = None  # type: ignore[assignment]


DEFAULT_SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
MODEL2_COLUMNS = [
    "sample_id",
    "source",
    "raw_text",
    "clean_text",
    "token_count",
    "char_count",
    "sentiment_label",
    "language",
    "label_source",
    "group_id",
]

POSITIVE_TERMS = {
    "excellent",
    "stable",
    "good",
    "strong",
    "secure",
    "attractive",
    "high rental yield",
    "bon",
    "fiable",
    "fiables",
    "developpe",
    "developpee",
    "potentiel",
    "eleve",
    "elevee",
    "croissante",
    "confort",
}

NEGATIVE_TERMS = {
    "lack transparency",
    "deficiencies",
    "deficiency",
    "weak",
    "poor",
    "risque",
    "risky",
    "excessif",
    "excessive",
    "manquant",
    "insufficient",
    "insuffisant",
    "instable",
    "probleme",
    "deficience",
    "deficiences",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare two NLP datasets: model1 (price from descriptions) and model2 (sentiment from reviews)."
    )
    parser.add_argument("--reviews-txt", default="data/nlp/real_estate_synthesized_reviews.txt")
    parser.add_argument("--reviews-metadata-csv", default="data/nlp/real_estate_reviews_metadata.csv")
    parser.add_argument("--listings-csv", default="data/csv/preprocessed/final_listings_preprocessed.csv")
    parser.add_argument("--description-column", default="description", help="Description column in listings CSV.")
    parser.add_argument("--price-column", default="price_tnd", help="Price column in listings CSV.")
    parser.add_argument("--output-dir", default="data/nlp/prepared")
    parser.add_argument("--lowercase", action="store_true", default=True, help="Lowercase normalized text (default enabled).")
    parser.add_argument("--keep-case", action="store_true", help="Disable lowercase normalization.")
    parser.add_argument("--min-tokens-model1", type=int, default=5, help="Min tokens for model1 descriptions.")
    parser.add_argument("--min-tokens-model2", type=int, default=3, help="Min tokens for model2 reviews.")
    parser.add_argument(
        "--sentiment-model",
        default=DEFAULT_SENTIMENT_MODEL,
        help="Hugging Face sentiment-analysis pipeline model used for the sample audit.",
    )
    parser.add_argument(
        "--neutral-threshold",
        type=float,
        default=0.60,
        help="If the sentiment model does not emit a neutral class, sample predictions below this score become neutral.",
    )
    parser.add_argument("--sentiment-batch-size", type=int, default=32, help="Batch size for sentiment pipeline sample inference.")
    parser.add_argument("--sentiment-sample-size", type=int, default=25, help="Number of model2 rows to score with the sentiment pipeline as a sample audit.")
    parser.add_argument(
        "--target-model2-groups-per-class",
        type=int,
        default=120,
        help="If explicit labels do not cover at least this many unique groups per class, synthesize offline template groups to reach the target.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_text_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            raw = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            raw = ""
            continue
    lines = [line.strip() for line in raw.splitlines()]
    return [line for line in lines if line]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df: pd.DataFrame | None = None
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=encoding)
            break
        except UnicodeDecodeError:
            df = None
            continue
    return df if df is not None else pd.DataFrame()


def maybe_fix_mojibake(text: str) -> str:
    markers = ("Ãƒ", "Ã¢", "Ã°", "Ã‚")
    if not any(m in text for m in markers):
        return text
    try:
        repaired = text.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")
        if repaired and repaired.count("ï¿½") <= text.count("ï¿½"):
            return repaired
    except Exception:
        return text
    return text


def normalize_text(text: str, lowercase: bool = True) -> str:
    t = maybe_fix_mojibake(text)
    t = unicodedata.normalize("NFKC", t)

    t = re.sub(r"\[(phone|email)\]", r"<\1>", t, flags=re.IGNORECASE)
    t = re.sub(r"https?://\S+|www\.\S+", " <url> ", t, flags=re.IGNORECASE)
    t = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", " <email> ", t, flags=re.IGNORECASE)
    t = re.sub(r"\+?\d[\d\-\s()]{6,}\d", " <phone> ", t)
    t = re.sub(r"<\s*(phone|email|url)\s*>", r" <\1> ", t, flags=re.IGNORECASE)

    t = re.sub(r"[^\w\s+<>]", " ", t, flags=re.UNICODE)
    t = re.sub(r"_+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    if lowercase:
        t = t.lower()
    return t


def parse_price_value(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None

    text = text.replace(" ", "")
    text = text.replace(",", ".")
    m = re.search(r"\d+(?:\.\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def parse_price_from_text(text: str) -> float | None:
    t = maybe_fix_mojibake(str(text))
    t_low = t.lower()
    if "[phone]" in t_low or "<phone>" in t_low:
        return None

    md_match = re.search(r"(\d+(?:[.,]\d+)?)\s*(md|mdt)\b", t_low)
    if md_match:
        n = parse_price_value(md_match.group(1))
        if n is not None:
            return n * 1000.0

    m = re.search(r"(\d{2,3}(?:[.,\s]\d{3})+|\d+(?:[.,]\d+)?)\s*(tnd|dt|dinar|dinars)\b", t_low)
    if not m:
        m = re.search(r"\bprix\b[^0-9]{0,12}(\d{2,3}(?:[.,\s]\d{3})+|\d+(?:[.,]\d+)?)", t_low)
    if not m:
        return None

    n = parse_price_value(m.group(1))
    if n is None:
        return None
    return n


def split_df(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}")

    n = len(df)
    if n == 0:
        empty = df.copy()
        return empty, empty, empty

    rng = np.random.default_rng(seed)
    shuffled_idx = rng.permutation(n)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = df.iloc[shuffled_idx[:train_end]].reset_index(drop=True)
    val_df = df.iloc[shuffled_idx[train_end:val_end]].reset_index(drop=True)
    test_df = df.iloc[shuffled_idx[val_end:]].reset_index(drop=True)
    return train_df, val_df, test_df


def remove_accents(text: str) -> str:
    d = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in d if not unicodedata.combining(ch))


def canonicalize_language(value: Any, text: str = "") -> str:
    norm = remove_accents(maybe_fix_mojibake(str(value or "")).strip().lower())
    if norm in {"fr", "fra", "french", "francais", "français"}:
        return "fr"
    if norm in {"en", "eng", "english"}:
        return "en"

    text_norm = remove_accents(normalize_text(text, lowercase=True))
    french_markers = ("prix", "quartier", "ecoles", "transport", "juridiques", "ameliorations")
    if any(marker in text_norm for marker in french_markers):
        return "fr"
    return "en"


def canonicalize_sentiment_label(value: Any) -> Literal["negative", "neutral", "positive"] | None:
    norm = remove_accents(maybe_fix_mojibake(str(value or "")).strip().lower())
    if norm in {"negative", "neg", "bad", "bearish"}:
        return "negative"
    if norm in {"neutral", "neu", "mixed"}:
        return "neutral"
    if norm in {"positive", "pos", "good", "bullish"}:
        return "positive"
    return None


def make_group_id(text: str, language: str) -> str:
    normalized = remove_accents(normalize_text(text, lowercase=True))
    digest = hashlib.sha1(f"{language}|{normalized}".encode("utf-8")).hexdigest()[:12]
    return f"grp_{digest}"


def ensure_model2_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in MODEL2_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    return out[MODEL2_COLUMNS]


def _group_counts_by_label(df: pd.DataFrame) -> dict[str, int]:
    if df.empty or "group_id" not in df.columns or "sentiment_label" not in df.columns:
        return {}
    grouped = (
        df.groupby("group_id", dropna=False)["sentiment_label"]
        .agg(lambda s: s.mode().iloc[0])
        .astype(str)
    )
    return {str(k): int(v) for k, v in grouped.value_counts().to_dict().items()}


def _map_pipeline_label(label: str, score: float, neutral_threshold: float) -> Literal["negative", "neutral", "positive"]:
    """Normalize diverse pipeline labels into negative/neutral/positive."""
    norm = remove_accents(str(label).strip().lower())

    if any(key in norm for key in ["neutral", "3 stars", "3 star", "label_1"]):
        return "neutral"
    if any(key in norm for key in ["positive", "pos", "5 stars", "5 star", "4 stars", "4 star", "label_2"]):
        return "positive"
    if any(key in norm for key in ["negative", "neg", "1 star", "1 stars", "2 stars", "2 star", "label_0"]):
        return "negative"

    if score < neutral_threshold:
        return "neutral"
    return "positive" if "p" in norm else "negative"


def infer_sentiment_label(text: str) -> Literal["negative", "neutral", "positive"]:
    """Infer sentiment with the original fast rule-based heuristic."""
    clean = normalize_text(text, lowercase=True)
    norm = remove_accents(clean)
    pos_score = sum(1 for term in POSITIVE_TERMS if term in norm)
    neg_score = sum(1 for term in NEGATIVE_TERMS if term in norm)
    if pos_score > neg_score:
        return "positive"
    if neg_score > pos_score:
        return "negative"
    return "neutral"


def infer_sentiment_labels_with_pipeline(
    texts: list[str],
    model_name: str,
    neutral_threshold: float,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Infer sentiment labels with a Hugging Face sentiment-analysis pipeline for audit sampling."""
    if pipeline is None:
        raise ModuleNotFoundError(
            "transformers is required for model2 sentiment labeling. Install it before running prepare_nlp_data.py."
        )

    classifier = pipeline("sentiment-analysis", model=model_name)  # type: ignore[arg-type]
    outputs = classifier(texts, truncation=True, batch_size=batch_size)
    rows: list[dict[str, Any]] = []
    for result in outputs:
        raw_label = str(result.get("label", ""))
        score = float(result.get("score", 0.0))
        rows.append(
            {
                "pipeline_label_raw": raw_label,
                "pipeline_score": score,
                "pipeline_label_mapped": _map_pipeline_label(
                    label=raw_label,
                    score=score,
                    neutral_threshold=neutral_threshold,
                ),
            }
        )
    return rows


def build_model1_descriptions_df(
    listings_df: pd.DataFrame,
    description_col: str,
    price_col: str,
    lowercase: bool,
    min_tokens: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []

    if not listings_df.empty and description_col in listings_df.columns:
        for _, row in listings_df.iterrows():
            text = str(row.get(description_col, "")).strip()
            if not text:
                continue
            price = parse_price_value(row.get(price_col))
            rows.append(
                {
                    "source": "listings_csv_description",
                    "raw_text": text,
                    "target_price_tnd": price,
                    "price_label_source": "listings_price_col" if price is not None else "missing",
                }
            )

    all_df = pd.DataFrame(rows)
    if all_df.empty:
        empty_cols = [
            "sample_id",
            "source",
            "raw_text",
            "clean_text",
            "token_count",
            "char_count",
            "target_price_tnd",
            "has_price_label",
            "price_label_source",
        ]
        empty = pd.DataFrame(columns=empty_cols)
        return empty, empty.copy()

    all_df["clean_text"] = all_df["raw_text"].astype(str).map(lambda s: normalize_text(s, lowercase=lowercase))
    all_df["token_count"] = all_df["clean_text"].map(lambda s: len(s.split()))
    all_df["char_count"] = all_df["clean_text"].map(len)
    all_df = all_df[(all_df["clean_text"].str.len() > 0) & (all_df["token_count"] >= int(min_tokens))]
    all_df = all_df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)
    all_df.insert(0, "sample_id", [f"m1_{i:07d}" for i in range(1, len(all_df) + 1)])
    all_df["has_price_label"] = all_df["target_price_tnd"].notna()

    labeled_df = all_df[all_df["has_price_label"]].copy().reset_index(drop=True)
    return all_df, labeled_df


def _build_model2_from_metadata(
    reviews_metadata_df: pd.DataFrame,
    lowercase: bool,
    min_tokens: int,
) -> pd.DataFrame:
    if reviews_metadata_df.empty:
        return pd.DataFrame(columns=MODEL2_COLUMNS)

    review_col = "review_text" if "review_text" in reviews_metadata_df.columns else None
    label_col = "sentiment" if "sentiment" in reviews_metadata_df.columns else None
    if review_col is None or label_col is None:
        return pd.DataFrame(columns=MODEL2_COLUMNS)

    rows: list[dict[str, Any]] = []
    for _, row in reviews_metadata_df.iterrows():
        raw_text = maybe_fix_mojibake(str(row.get(review_col, ""))).strip()
        if not raw_text:
            continue
        sentiment_label = canonicalize_sentiment_label(row.get(label_col))
        if sentiment_label is None:
            continue
        language = canonicalize_language(row.get("language"), raw_text)
        clean_text = normalize_text(raw_text, lowercase=lowercase)
        token_count = len(clean_text.split())
        if not clean_text or token_count < int(min_tokens):
            continue
        rows.append(
            {
                "source": str(row.get("source") or "reviews_metadata_csv"),
                "raw_text": raw_text,
                "clean_text": clean_text,
                "token_count": token_count,
                "char_count": len(clean_text),
                "sentiment_label": sentiment_label,
                "language": language,
                "label_source": str(row.get("label_source") or "metadata_label"),
                "group_id": make_group_id(raw_text, language),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=MODEL2_COLUMNS)
    out.insert(0, "sample_id", [f"m2_{i:07d}" for i in range(1, len(out) + 1)])
    return ensure_model2_schema(out)


def _build_model2_from_reviews_txt(
    reviews_txt: Iterable[str],
    lowercase: bool,
    min_tokens: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for text in reviews_txt:
        raw_text = maybe_fix_mojibake(str(text)).strip()
        if not raw_text:
            continue
        clean_text = normalize_text(raw_text, lowercase=lowercase)
        token_count = len(clean_text.split())
        if not clean_text or token_count < int(min_tokens):
            continue
        language = canonicalize_language("", raw_text)
        rows.append(
            {
                "source": "reviews_txt",
                "raw_text": raw_text,
                "clean_text": clean_text,
                "token_count": token_count,
                "char_count": len(clean_text),
                "sentiment_label": infer_sentiment_label(raw_text),
                "language": language,
                "label_source": "rule_based_bootstrap",
                "group_id": make_group_id(raw_text, language),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=MODEL2_COLUMNS)
    out.insert(0, "sample_id", [f"m2_{i:07d}" for i in range(1, len(out) + 1)])
    return ensure_model2_schema(out)


def build_model2_reviews_df(
    reviews_txt: Iterable[str],
    reviews_metadata_df: pd.DataFrame,
    lowercase: bool,
    min_tokens: int,
    target_groups_per_class: int,
    seed: int,
) -> pd.DataFrame:
    """Build model2 from explicit labels when available, falling back to rule-based bootstrap labels."""
    metadata_df = _build_model2_from_metadata(
        reviews_metadata_df=reviews_metadata_df,
        lowercase=lowercase,
        min_tokens=min_tokens,
    )
    if not metadata_df.empty:
        group_counts = _group_counts_by_label(metadata_df)
        synth_needed = any(group_counts.get(label, 0) < int(target_groups_per_class) for label in ("negative", "neutral", "positive"))
        if synth_needed:
            synth_metadata_df = synthesize_sentiment_metadata(
                target_groups_per_label=int(target_groups_per_class),
                seed=int(seed),
            )
            synth_df = _build_model2_from_metadata(
                reviews_metadata_df=synth_metadata_df,
                lowercase=lowercase,
                min_tokens=min_tokens,
            )
            if not synth_df.empty:
                existing_groups = set(metadata_df["group_id"].astype(str))
                synth_df = synth_df[~synth_df["group_id"].astype(str).isin(existing_groups)].reset_index(drop=True)
                if not synth_df.empty:
                    metadata_df = pd.concat([metadata_df, synth_df], ignore_index=True)
                    metadata_df["sample_id"] = [f"m2_{i:07d}" for i in range(1, len(metadata_df) + 1)]
                    metadata_df = ensure_model2_schema(metadata_df)
        return metadata_df.reset_index(drop=True)

    return _build_model2_from_reviews_txt(
        reviews_txt=reviews_txt,
        lowercase=lowercase,
        min_tokens=min_tokens,
    ).reset_index(drop=True)


def _split_targets_for_label(
    n_groups: int,
    ratios: dict[str, float],
) -> dict[str, int]:
    active_splits = [name for name, ratio in ratios.items() if ratio > 0]
    if n_groups <= 0:
        return {name: 0 for name in ratios}

    targets = {name: 0 for name in ratios}
    if n_groups >= len(active_splits):
        for name in active_splits:
            targets[name] = 1
        remaining = n_groups - len(active_splits)
    else:
        remaining = n_groups

    if remaining > 0:
        weights = np.array([ratios[name] for name in active_splits], dtype=float)
        weights = weights / weights.sum()
        raw = remaining * weights
        base = np.floor(raw).astype(int)
        remainder = remaining - int(base.sum())

        for idx, name in enumerate(active_splits):
            targets[name] += int(base[idx])

        if remainder > 0:
            order = np.argsort(-(raw - base))
            for idx in order[:remainder]:
                targets[active_splits[int(idx)]] += 1

    return targets


def split_grouped_sentiment_df(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    label_col: str = "sentiment_label",
    group_col: str = "group_id",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}")

    if df.empty:
        empty = df.copy()
        return empty, empty, empty

    if group_col not in df.columns or label_col not in df.columns:
        return split_df(df, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)

    group_stats = (
        df.groupby(group_col, dropna=False)
        .agg(
            group_label=(label_col, lambda s: s.mode().iloc[0]),
        )
        .reset_index()
    )

    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    rng = np.random.default_rng(seed)
    assigned_groups: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for label in sorted(group_stats["group_label"].astype(str).unique().tolist()):
        label_groups = group_stats.loc[group_stats["group_label"] == label, group_col].astype(str).tolist()
        rng.shuffle(label_groups)
        targets = _split_targets_for_label(len(label_groups), ratios)

        cursor = 0
        for split_name in ("train", "val", "test"):
            take = targets[split_name]
            assigned_groups[split_name].extend(label_groups[cursor:cursor + take])
            cursor += take

    train_df = df[df[group_col].astype(str).isin(assigned_groups["train"])].reset_index(drop=True)
    val_df = df[df[group_col].astype(str).isin(assigned_groups["val"])].reset_index(drop=True)
    test_df = df[df[group_col].astype(str).isin(assigned_groups["test"])].reset_index(drop=True)
    return train_df, val_df, test_df


def _distribution(df: pd.DataFrame, column: str) -> dict[str, int]:
    if df.empty or column not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df[column].value_counts().to_dict().items()}


def split_summary(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rows": int(len(df)),
        "label_distribution": _distribution(df, "sentiment_label"),
        "language_distribution": _distribution(df, "language"),
    }
    if "group_id" in df.columns:
        summary["unique_groups"] = int(df["group_id"].astype(str).nunique())
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    lowercase = bool(args.lowercase and not args.keep_case)

    reviews_path = Path(args.reviews_txt)
    reviews_metadata_path = Path(args.reviews_metadata_csv)
    listings_csv_path = Path(args.listings_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reviews = read_text_lines(reviews_path)
    reviews_metadata_df = read_csv(reviews_metadata_path)
    if not reviews and reviews_metadata_df.empty:
        raise FileNotFoundError(
            "No review text was loaded for model2. Check --reviews-txt and --reviews-metadata-csv inputs."
        )
    listings_df = read_csv(listings_csv_path)

    model1_all_df, model1_labeled_df = build_model1_descriptions_df(
        listings_df=listings_df,
        description_col=str(args.description_column),
        price_col=str(args.price_column),
        lowercase=lowercase,
        min_tokens=int(args.min_tokens_model1),
    )
    model2_df = build_model2_reviews_df(
        reviews_txt=reviews,
        reviews_metadata_df=reviews_metadata_df,
        lowercase=lowercase,
        min_tokens=int(args.min_tokens_model2),
        target_groups_per_class=int(args.target_model2_groups_per_class),
        seed=int(args.seed),
    )

    m1_train, m1_val, m1_test = split_df(
        model1_labeled_df,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
    )
    m2_train, m2_val, m2_test = split_grouped_sentiment_df(
        model2_df,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
    )

    m1_all_out = output_dir / "model1_descriptions_all.csv"
    m1_labeled_out = output_dir / "model1_price_labeled.csv"
    m1_train_out = output_dir / "model1_train.csv"
    m1_val_out = output_dir / "model1_val.csv"
    m1_test_out = output_dir / "model1_test.csv"

    m2_all_out = output_dir / "model2_reviews_labeled.csv"
    m2_train_out = output_dir / "model2_train.csv"
    m2_val_out = output_dir / "model2_val.csv"
    m2_test_out = output_dir / "model2_test.csv"
    m2_synth_out = output_dir / "model2_synthetic_reviews.csv"
    m2_sample_out = output_dir / "model2_sentiment_pipeline_sample.csv"

    report_out = output_dir / "prep_report.json"

    model1_all_df.to_csv(m1_all_out, index=False, encoding="utf-8")
    model1_labeled_df.to_csv(m1_labeled_out, index=False, encoding="utf-8")
    m1_train.to_csv(m1_train_out, index=False, encoding="utf-8")
    m1_val.to_csv(m1_val_out, index=False, encoding="utf-8")
    m1_test.to_csv(m1_test_out, index=False, encoding="utf-8")

    model2_df.to_csv(m2_all_out, index=False, encoding="utf-8")
    m2_train.to_csv(m2_train_out, index=False, encoding="utf-8")
    m2_val.to_csv(m2_val_out, index=False, encoding="utf-8")
    m2_test.to_csv(m2_test_out, index=False, encoding="utf-8")

    generated_rows_df = synthesize_sentiment_metadata(
        target_groups_per_label=int(args.target_model2_groups_per_class),
        seed=int(args.seed),
    )
    generated_rows_df.to_csv(m2_synth_out, index=False, encoding="utf-8")

    sample_report: dict[str, Any] = {
        "enabled": int(args.sentiment_sample_size) > 0,
        "rows_written": 0,
        "output_csv": str(m2_sample_out),
        "status": "not_run",
    }
    if int(args.sentiment_sample_size) > 0 and not model2_df.empty:
        sample_df = model2_df.drop_duplicates(subset=["group_id"]).head(int(args.sentiment_sample_size)).copy()
        sample_df = sample_df[["sample_id", "raw_text", "sentiment_label", "language", "group_id"]].copy()
        try:
            pipeline_rows = infer_sentiment_labels_with_pipeline(
                sample_df["raw_text"].astype(str).tolist(),
                model_name=str(args.sentiment_model),
                neutral_threshold=float(args.neutral_threshold),
                batch_size=int(args.sentiment_batch_size),
            )
            pipeline_df = pd.DataFrame(pipeline_rows)
            sample_df = pd.concat([sample_df.reset_index(drop=True), pipeline_df.reset_index(drop=True)], axis=1)
            sample_df.to_csv(m2_sample_out, index=False, encoding="utf-8")
            sample_report["rows_written"] = int(len(sample_df))
            sample_report["status"] = "ok"
        except Exception as exc:
            sample_report["status"] = f"failed: {exc}"

    all_groups = set(model2_df["group_id"].astype(str)) if not model2_df.empty else set()
    train_groups = set(m2_train["group_id"].astype(str)) if not m2_train.empty else set()
    val_groups = set(m2_val["group_id"].astype(str)) if not m2_val.empty else set()
    test_groups = set(m2_test["group_id"].astype(str)) if not m2_test.empty else set()

    report: dict[str, Any] = {
        "inputs": {
            "reviews_txt": str(reviews_path),
            "reviews_metadata_csv": str(reviews_metadata_path),
            "listings_csv": str(listings_csv_path),
            "description_column": str(args.description_column),
            "price_column": str(args.price_column),
        },
        "outputs": {
            "model1_descriptions_all_csv": str(m1_all_out),
            "model1_price_labeled_csv": str(m1_labeled_out),
            "model1_train_csv": str(m1_train_out),
            "model1_val_csv": str(m1_val_out),
            "model1_test_csv": str(m1_test_out),
            "model2_reviews_labeled_csv": str(m2_all_out),
            "model2_train_csv": str(m2_train_out),
            "model2_val_csv": str(m2_val_out),
            "model2_test_csv": str(m2_test_out),
            "model2_synthetic_reviews_csv": str(m2_synth_out),
            "model2_sentiment_pipeline_sample_csv": str(m2_sample_out),
            "report_json": str(report_out),
        },
        "config": {
            "lowercase": lowercase,
            "min_tokens_model1": int(args.min_tokens_model1),
            "min_tokens_model2": int(args.min_tokens_model2),
            "sentiment_model": str(args.sentiment_model),
            "neutral_threshold": float(args.neutral_threshold),
            "sentiment_batch_size": int(args.sentiment_batch_size),
            "sentiment_sample_size": int(args.sentiment_sample_size),
            "target_model2_groups_per_class": int(args.target_model2_groups_per_class),
            "split": {
                "train": float(args.train_ratio),
                "val": float(args.val_ratio),
                "test": float(args.test_ratio),
            },
            "seed": int(args.seed),
        },
        "counts": {
            "model1": {
                "all_rows": int(len(model1_all_df)),
                "labeled_rows": int(len(model1_labeled_df)),
                "train_rows": int(len(m1_train)),
                "val_rows": int(len(m1_val)),
                "test_rows": int(len(m1_test)),
                "source_distribution": _distribution(model1_all_df, "source"),
            },
            "model2": {
                "all_rows": int(len(model2_df)),
                "unique_clean_text": int(model2_df["clean_text"].astype(str).nunique()) if not model2_df.empty else 0,
                "unique_groups": int(model2_df["group_id"].astype(str).nunique()) if not model2_df.empty else 0,
                "groups_per_class": _group_counts_by_label(model2_df),
                "train_rows": int(len(m2_train)),
                "val_rows": int(len(m2_val)),
                "test_rows": int(len(m2_test)),
                "source_distribution": _distribution(model2_df, "source"),
                "label_source_distribution": _distribution(model2_df, "label_source"),
                "sentiment_distribution": _distribution(model2_df, "sentiment_label"),
                "language_distribution": _distribution(model2_df, "language"),
                "splits": {
                    "train": split_summary(m2_train),
                    "val": split_summary(m2_val),
                    "test": split_summary(m2_test),
                },
                "group_leakage": {
                    "train_val_overlap": int(len(train_groups & val_groups)),
                    "train_test_overlap": int(len(train_groups & test_groups)),
                    "val_test_overlap": int(len(val_groups & test_groups)),
                    "covered_groups": int(len(train_groups | val_groups | test_groups)),
                    "all_groups": int(len(all_groups)),
                },
                "synthesis": {
                    "target_groups_per_class": int(args.target_model2_groups_per_class),
                    "generated_rows_csv": str(m2_synth_out),
                    "generated_rows": int(len(generated_rows_df)),
                    "incorporated_rows": int(
                        len(model2_df[model2_df["label_source"].astype(str) == "synthetic_template_label"])
                    ),
                },
                "pipeline_sample": sample_report,
            },
        },
    }
    report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    args = parse_args()
    report = run(args)
    print("NLP data preparation complete.")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
