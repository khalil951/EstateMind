from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

import pandas as pd


@dataclass
class EnrichReport:
    rows: int
    filled_rooms: int
    filled_bedrooms: int
    filled_bathrooms: int
    filled_surface_m2: int
    filled_city: int
    filled_governorate: int
    llm_calls: int
    llm_rows_used: int


def _clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _norm(value: str) -> str:
    text = _clean_text(value).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9+\s'-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _to_float(value: str) -> float | None:
    if not value:
        return None
    s = value.strip().replace(" ", "")
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        frac = len(s.split(",")[-1])
        s = s.replace(",", ".") if frac in (1, 2) else s.replace(",", "")
    elif "." in s:
        frac = len(s.split(".")[-1])
        s = s if frac in (1, 2) else s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return None


def _extract_rooms(text: str) -> int | None:
    blob = _norm(text)

    if re.search(r"\b(studio|garconniere)\b", blob):
        return 1

    m = re.search(r"\bs\s*\+\s*(\d+)\b", blob)
    if m:
        return int(m.group(1)) + 1

    m = re.search(r"\bs\s*(\d+)\b", blob)
    if m:
        return int(m.group(1)) + 1

    m = re.search(r"\b(\d+)\s*(?:pieces?|piece|p)\b", blob)
    if m:
        return int(m.group(1))

    return None


def _extract_bedrooms(text: str) -> int | None:
    blob = _norm(text)
    if re.search(r"\b(studio|garconniere)\b", blob):
        return 0

    m = re.search(r"\b(\d+)\s*(?:bedrooms?|chambres?|chb)\b", blob)
    if m:
        return int(m.group(1))

    m = re.search(r"\bs\s*\+\s*(\d+)\b", blob)
    if m:
        return int(m.group(1))

    m = re.search(r"\bs\s*(\d+)\b", blob)
    if m:
        return int(m.group(1))
    return None


def _extract_bathrooms(text: str) -> int | None:
    blob = _norm(text)
    m = re.search(r"\b(\d+)\s*(?:bathrooms?|sdb|salle(?:s)? de bain)\b", blob)
    if m:
        return int(m.group(1))
    return None


def _extract_surface_m2(text: str) -> float | None:
    blob = _norm(text)
    m = re.search(r"\b(\d{1,4}(?:[.,]\d+)?)\s*(?:m2|m\^2|m)\b", blob)
    if not m:
        m = re.search(r"\b(?:surface|superficie|area)\s*[:=-]?\s*(\d{1,4}(?:[.,]\d+)?)\b", blob)
    if not m:
        return None
    n = _to_float(m.group(1))
    if n is None or n <= 0:
        return None
    return n


def _build_city_lookup(df: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    city_col = df["city"] if "city" in df.columns else pd.Series(dtype="object")
    gov_col = df["governorate"] if "governorate" in df.columns else pd.Series(dtype="object")

    pairs = (
        pd.DataFrame({"city": city_col, "governorate": gov_col})
        .dropna(subset=["city"])
        .assign(city_norm=lambda d: d["city"].map(_norm))
    )
    pairs = pairs[pairs["city_norm"].astype(bool)]

    city_names = sorted(pairs["city_norm"].unique().tolist(), key=len, reverse=True)

    city_to_gov: dict[str, str] = {}
    for city_norm, group in pairs.groupby("city_norm", sort=False):
        gov = group["governorate"].dropna()
        if not gov.empty:
            city_to_gov[str(city_norm)] = str(gov.mode().iat[0])

    return city_names, city_to_gov


def _extract_city_and_governorate(
    text: str,
    city_names: list[str],
    city_to_governorate: dict[str, str],
) -> tuple[str | None, str | None]:
    blob = f" {_norm(text)} "
    for city_norm in city_names:
        if not city_norm:
            continue
        pattern = rf"(?<!\w){re.escape(city_norm)}(?!\w)"
        if re.search(pattern, blob):
            governorate = city_to_governorate.get(city_norm)
            return city_norm, governorate
    return None, None


def _to_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(round(float(str(value).strip())))
    except (TypeError, ValueError):
        return None


def _call_llm_extract(
    text: str,
    api_key: str,
    model: str,
    base_url: str,
    timeout_sec: int,
) -> dict[str, Any] | None:
    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint += "/chat/completions"

    prompt = (
        "Extract structured real-estate fields from the FULL listing title text. "
        "Apply these pattern rules strictly when detected in title:\n"
        "1) S+N (or sN / S N) means bedrooms=N and rooms=N+1.\n"
        "2) Studio means rooms=2, bedrooms=1, bathrooms=1 unless title explicitly gives different values.\n"
        "3) If S+N is detected and bathrooms is not explicit, infer bathrooms=1.\n"
        "4) If explicit values (e.g. chambres, SDB, pieces, m2) appear, prefer explicit values over defaults.\n"
        "5)Infer transaction_type from title cues: 'a vendre'/'vente' => sale, 'a louer'/'location' => rent.\n"
        "Return ONLY valid JSON with keys: rooms, bedrooms, bathrooms, surface_m2, city, governorate, transaction_type. "
        "Use null when unknown. No explanation.\n\n"
        f"TITLE: {text}"
    )

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "You are a precise information extraction engine."},
            {"role": "user", "content": prompt},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        endpoint,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with request.urlopen(req, timeout=timeout_sec) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (error.URLError, TimeoutError, json.JSONDecodeError):
        return None

    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None

    if isinstance(content, list):
        content = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    content = str(content)

    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        parsed = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def enrich_from_title(
    df: pd.DataFrame,
    use_llm: bool = False,
    llm_api_key: str | None = None,
    llm_model: str = "meta-llama/llama-3.1-8b-instruct",
    llm_base_url: str = "https://openrouter.ai/api/v1",
    llm_max_rows: int = 500,
    llm_timeout_sec: int = 20,
) -> tuple[pd.DataFrame, EnrichReport]:
    out = df.copy()

    for col in ["title", "description", "location_raw", "neighborhood", "city", "governorate"]:
        if col not in out.columns:
            out[col] = None

    city_names, city_to_governorate = _build_city_lookup(out)
    city_norm_to_raw = {
        _norm(c): str(c)
        for c in out["city"].dropna().astype(str).tolist()
        if _norm(c)
    }

    filled_rooms = filled_bedrooms = filled_bathrooms = 0
    filled_surface_m2 = filled_city = filled_governorate = 0
    llm_calls = llm_rows_used = 0

    for idx, row in out.iterrows():
        title_text = _clean_text(row.get("title"))
        text = " | ".join(
            _clean_text(row.get(c))
            for c in ["title", "description", "location_raw", "neighborhood"]
            if _clean_text(row.get(c))
        )
        if not text:
            continue

        if pd.isna(row.get("rooms")):
            rooms = _extract_rooms(text)
            if rooms is not None:
                out.at[idx, "rooms"] = rooms
                filled_rooms += 1

        if pd.isna(row.get("bedrooms")):
            bedrooms = _extract_bedrooms(text)
            if bedrooms is not None:
                out.at[idx, "bedrooms"] = bedrooms
                filled_bedrooms += 1

        if pd.isna(row.get("bathrooms")):
            bathrooms = _extract_bathrooms(text)
            if bathrooms is not None:
                out.at[idx, "bathrooms"] = bathrooms
                filled_bathrooms += 1

        if pd.isna(row.get("surface_m2")):
            surface = _extract_surface_m2(text)
            if surface is not None:
                out.at[idx, "surface_m2"] = surface
                filled_surface_m2 += 1

        city_missing = pd.isna(row.get("city")) or not _clean_text(row.get("city"))
        gov_missing = pd.isna(row.get("governorate")) or not _clean_text(row.get("governorate"))
        if city_missing or gov_missing:
            city_norm, governorate = _extract_city_and_governorate(text, city_names, city_to_governorate)
            if city_missing and city_norm:
                out.at[idx, "city"] = city_norm_to_raw.get(city_norm, city_norm.title())
                filled_city += 1
            if gov_missing and governorate:
                out.at[idx, "governorate"] = governorate
                filled_governorate += 1

        should_call_llm = (
            use_llm
            and bool(llm_api_key)
            and llm_calls < llm_max_rows
            and (
                pd.isna(out.at[idx, "rooms"])
                or pd.isna(out.at[idx, "bedrooms"])
                or pd.isna(out.at[idx, "bathrooms"])
                or pd.isna(out.at[idx, "surface_m2"])
            )
        )
        if should_call_llm:
            llm_result = _call_llm_extract(
                text=title_text or text,
                api_key=str(llm_api_key),
                model=llm_model,
                base_url=llm_base_url,
                timeout_sec=llm_timeout_sec,
            )
            llm_calls += 1
            row_changed = False

            if llm_result:
                if pd.isna(out.at[idx, "rooms"]):
                    rooms = _to_optional_int(llm_result.get("rooms"))
                    if rooms is not None and rooms > 0:
                        out.at[idx, "rooms"] = rooms
                        filled_rooms += 1
                        row_changed = True

                if pd.isna(out.at[idx, "bedrooms"]):
                    bedrooms = _to_optional_int(llm_result.get("bedrooms"))
                    if bedrooms is not None and bedrooms >= 0:
                        out.at[idx, "bedrooms"] = bedrooms
                        filled_bedrooms += 1
                        row_changed = True

                if pd.isna(out.at[idx, "bathrooms"]):
                    bathrooms = _to_optional_int(llm_result.get("bathrooms"))
                    if bathrooms is not None and bathrooms >= 0:
                        out.at[idx, "bathrooms"] = bathrooms
                        filled_bathrooms += 1
                        row_changed = True

                if pd.isna(out.at[idx, "surface_m2"]):
                    surface = _to_float(str(llm_result.get("surface_m2") or ""))
                    if surface is not None and surface > 0:
                        out.at[idx, "surface_m2"] = surface
                        filled_surface_m2 += 1
                        row_changed = True

            if row_changed:
                llm_rows_used += 1

    report = EnrichReport(
        rows=len(out),
        filled_rooms=filled_rooms,
        filled_bedrooms=filled_bedrooms,
        filled_bathrooms=filled_bathrooms,
        filled_surface_m2=filled_surface_m2,
        filled_city=filled_city,
        filled_governorate=filled_governorate,
        llm_calls=llm_calls,
        llm_rows_used=llm_rows_used,
    )
    return out, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich listing CSV by extracting structured fields from title text.")
    parser.add_argument("--input-csv", default="data/csv/final_listings_wrangled.csv")
    parser.add_argument("--output-csv", default="data/csv/final_listings_wrangled_enriched.csv")
    parser.add_argument("--report-json", default="data/csv/final_listings_title_enrichment_report.json")
    parser.add_argument("--use-llm", action="store_true", help="Enable OpenAI-compatible API fallback extraction.")
    parser.add_argument("--llm-base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--llm-model", default="meta-llama/llama-3.1-8b-instruct")
    parser.add_argument("--llm-api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--llm-max-rows", type=int, default=500)
    parser.add_argument("--llm-timeout-sec", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    report_path = Path(args.report_json)

    df = pd.read_csv(input_path)
    before = {
        "missing_rooms": float(df["rooms"].isna().mean()) if "rooms" in df.columns else None,
        "missing_bedrooms": float(df["bedrooms"].isna().mean()) if "bedrooms" in df.columns else None,
        "missing_bathrooms": float(df["bathrooms"].isna().mean()) if "bathrooms" in df.columns else None,
        "missing_surface_m2": float(df["surface_m2"].isna().mean()) if "surface_m2" in df.columns else None,
        "missing_city": float(df["city"].isna().mean()) if "city" in df.columns else None,
        "missing_governorate": float(df["governorate"].isna().mean()) if "governorate" in df.columns else None,
    }

    llm_api_key = os.getenv(args.llm_api_key_env) if args.use_llm else None
    if args.use_llm and not llm_api_key:
        print(f"Warning: --use-llm enabled but env var {args.llm_api_key_env} is empty. Falling back to regex-only.")

    enriched, report = enrich_from_title(
        df,
        use_llm=args.use_llm and bool(llm_api_key),
        llm_api_key=llm_api_key,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_max_rows=args.llm_max_rows,
        llm_timeout_sec=args.llm_timeout_sec,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_path, index=False, encoding="utf-8")

    after = {
        "missing_rooms": float(enriched["rooms"].isna().mean()) if "rooms" in enriched.columns else None,
        "missing_bedrooms": float(enriched["bedrooms"].isna().mean()) if "bedrooms" in enriched.columns else None,
        "missing_bathrooms": float(enriched["bathrooms"].isna().mean()) if "bathrooms" in enriched.columns else None,
        "missing_surface_m2": float(enriched["surface_m2"].isna().mean()) if "surface_m2" in enriched.columns else None,
        "missing_city": float(enriched["city"].isna().mean()) if "city" in enriched.columns else None,
        "missing_governorate": float(enriched["governorate"].isna().mean()) if "governorate" in enriched.columns else None,
    }

    report_payload = {
        "input_csv": str(input_path),
        "output_csv": str(output_path),
        "summary": report.__dict__,
        "llm": {
            "enabled": bool(args.use_llm and llm_api_key),
            "base_url": args.llm_base_url,
            "model": args.llm_model,
            "max_rows": args.llm_max_rows,
        },
        "before_missing_ratio": before,
        "after_missing_ratio": after,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    print("Title enrichment completed")
    print(json.dumps(report_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
