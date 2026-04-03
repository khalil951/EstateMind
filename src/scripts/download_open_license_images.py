"""Download a cached open-license image set for EstateMind smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

HEADERS = {
    "User-Agent": "EstateMind/1.0 (open-license-cache-builder; https://github.com/openai)"
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download open-license images referenced by a manifest JSON file.")
    parser.add_argument("--manifest", default="artifacts/test_assets/open_license_images/manifest.json")
    parser.add_argument("--output-dir", default="artifacts/test_assets/open_license_images/cache")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    images = payload.get("images", [])
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for item in images:
        url = str(item["url"])
        filename = str(item["filename"])
        target = output_dir / filename
        if target.exists():
            continue
        response = requests.get(url, timeout=30, headers=HEADERS)
        response.raise_for_status()
        target.write_bytes(response.content)
        downloaded += 1

    print(json.dumps({"downloaded": downloaded, "cache_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
