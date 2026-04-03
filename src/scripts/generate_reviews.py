import argparse
import csv
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import requests


DEFAULT_API_URL = "https://router.huggingface.co/v1/chat/completions"
DEFAULT_MODEL = "HuggingFaceTB/SmolLM3-3B"
DEFAULT_ITERATIONS = 1000
DEFAULT_OUTPUT = "data/reviews_text_dataset.txt"
DEFAULT_DATA_DIR = "data"
DEFAULT_RUNS = 5

LOCATION_COLUMNS = {
    "location",
    "city",
    "governorate",
    "region",
    "listing_location",
    "raw_city",
    "raw_governorate",
}


def build_prompt(location: str, num_reviews: int) -> str:
    return (
        f"Generate {num_reviews} realistic neighborhood reviews for {location}, Tunisia.\n"
        "Include both positive and negative aspects. Write in French.\n"
        "Reviews should mention safety, noise, amenities, schools, and transport.\n"
        "Return reviews only, without analysis tags."
    )


def _headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def _clean_location(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"nan", "none", "null"}:
        return None
    text = re.sub(r"\s+", " ", text)
    if not any(ch.isalpha() for ch in text):
        return None
    if len(text) > 100:
        return None
    return text


def load_locations_from_csv(data_dir: Path) -> list[str]:
    csv_files = sorted(data_dir.rglob("*.csv"))
    locations: set[str] = set()

    for csv_path in csv_files:
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as fp:
                reader = csv.DictReader(fp)
                if not reader.fieldnames:
                    continue
                valid_cols = [
                    c
                    for c in reader.fieldnames
                    if c and c.replace("\ufeff", "").strip().lower() in LOCATION_COLUMNS
                ]
                if not valid_cols:
                    continue
                for row in reader:
                    for col in valid_cols:
                        cleaned = _clean_location(row.get(col))
                        if cleaned:
                            locations.add(cleaned)
        except UnicodeDecodeError:
            with csv_path.open("r", encoding="latin-1", newline="") as fp:
                reader = csv.DictReader(fp)
                if not reader.fieldnames:
                    continue
                valid_cols = [
                    c
                    for c in reader.fieldnames
                    if c and c.replace("\ufeff", "").strip().lower() in LOCATION_COLUMNS
                ]
                if not valid_cols:
                    continue
                for row in reader:
                    for col in valid_cols:
                        cleaned = _clean_location(row.get(col))
                        if cleaned:
                            locations.add(cleaned)
        except Exception:
            continue

    return sorted(locations)


def strip_think_tags(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _extract_generated_text(result: Any) -> str | None:
    if not isinstance(result, dict):
        return None
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    chunks.append(item["text"])
            if chunks:
                return "\n".join(chunks).strip()
    return None


def _generate_synthetic_reviews(
    location: str,
    token: str,
    model_id: str = DEFAULT_MODEL,
    api_url: str = DEFAULT_API_URL,
    seed: int | None = None,
    num_reviews: int = 1,
    timeout_s: int = 60,
    max_retries: int = 2,
) -> str | None:
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": "You generate concise French neighborhood reviews only. Never output <think> tags.",
            },
            {"role": "user", "content": build_prompt(location, num_reviews)},
        ],
        "temperature": 0.8,
        "top_p": 0.9,
        "max_tokens": 220,
        "stream": False,
    }
    if seed is not None:
        payload["seed"] = seed

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                api_url,
                headers=_headers(token),
                json=payload,
                timeout=timeout_s,
            )
        except requests.exceptions.Timeout:
            return None
        except requests.exceptions.RequestException:
            return None

        if response.status_code == 200:
            try:
                result = response.json()
            except ValueError:
                return None
            text = _extract_generated_text(result)
            return strip_think_tags(text) if text else None

        if response.status_code in {429, 503} and attempt < max_retries:
            wait_s = 2
            try:
                body = response.json()
                if isinstance(body, dict):
                    if "estimated_time" in body:
                        wait_s = max(1, int(body["estimated_time"]))
                    elif "retry_after" in body:
                        wait_s = max(1, int(body["retry_after"]))
            except ValueError:
                pass
            time.sleep(wait_s)
            continue

        return None

    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic reviews text dataset using Hugging Face Router API.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory containing CSV datasets.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output TXT dataset path.")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS, help="Total generation calls.")
    parser.add_argument("--model", default=os.getenv("HF_MODEL_ID", DEFAULT_MODEL), help="HF model id.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="HF Router chat completions URL.")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--sleep-ms", type=int, default=150, help="Delay between calls to reduce throttling.")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Repeat full generation loop N times.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for location sampling.")
    parser.add_argument("--append", action="store_true", help="Append to output file instead of overwriting.")
    return parser.parse_args()


def _main() -> int:
    args = _parse_args()
    token = os.getenv("HF_API_TOKEN")
    if not token:
        print("Error: HF_API_TOKEN is not set.")
        print("PowerShell: $env:HF_API_TOKEN='your_token_here'")
        return 1

    locations = load_locations_from_csv(Path(args.data_dir))
    if not locations:
        print(f"Error: no locations found in CSV files under {args.data_dir}")
        return 2

    print("=" * 60)
    print("Synthetic Reviews Text Dataset Generator")
    print("=" * 60)
    print(f"Locations loaded: {len(locations)}")
    print(f"Iterations: {args.iterations}")
    print(f"Runs: {args.runs}")
    print(f"Seed: {args.seed}")
    print(f"Model: {args.model}")
    print(f"API URL: {args.api_url}")
    print()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    total_kept = 0
    total_attempts = args.runs * args.iterations
    with output_path.open(mode, encoding="utf-8") as fp:
        for run in range(args.runs):
            rng = random.Random(args.seed + run)
            run_kept = 0
            for i in range(args.iterations):
                location = rng.choice(locations)
                req_seed = ((args.seed + run) * 1_000_000) + i
                review = _generate_synthetic_reviews(
                    location=location,
                    token=token,
                    model_id=args.model,
                    api_url=args.api_url,
                    seed=req_seed,
                    num_reviews=1,
                    timeout_s=args.timeout,
                    max_retries=args.max_retries,
                )
                if review:
                    cleaned = strip_think_tags(review)
                    if cleaned:
                        fp.write(cleaned + "\n")
                        run_kept += 1
                        total_kept += 1

                global_step = (run * args.iterations) + (i + 1)
                if global_step % 25 == 0:
                    print(
                        f"Progress: {global_step}/{total_attempts} | "
                        f"run {run + 1}/{args.runs} kept: {run_kept} | total kept: {total_kept}"
                    )

                if args.sleep_ms > 0:
                    time.sleep(args.sleep_ms / 1000)

    print("\n" + "=" * 60)
    print(f"Generated reviews kept: {total_kept}")
    print(f"Saved dataset: {output_path}")
    return 0 if total_kept else 3


if __name__ == "__main__":
    raise SystemExit(_main())
