from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.feature_aggregation import AMENITY_LABEL_MAP, PROPERTY_HINT_MAP
from src.vision.type_classifier import ImageTypeClassifierService


REPORT_DIR = Path("artifacts/reports/vision_runtime")
REPORT_JSON = REPORT_DIR / "vision_runtime_report.json"
REPORT_MD = REPORT_DIR / "vision_runtime_report.md"

LOCAL_IMAGE_CASES = [
    {
        "name": "house_facade",
        "path": Path("artifacts/test_assets/open_license_images/cache/house_facade.jpg"),
        "expected_property_type": "Maison",
    },
    {
        "name": "apartment_building",
        "path": Path("artifacts/test_assets/open_license_images/cache/apartment_building.jpg"),
        "expected_property_type": "Appartement",
    },
    {
        "name": "land_plot",
        "path": Path("artifacts/test_assets/open_license_images/cache/land_plot.jpg"),
        "expected_property_type": "Terrain",
    },
]

# Small synthetic probes to exercise amenity prompts in a deterministic way.
SYNTHETIC_CASES = [
    {
        "name": "amenity_pool_probe",
        "path": Path("artifacts/test_assets/synthetic_amenities/pool_blue.jpg"),
        "expected_amenity": "has_pool",
        "rgb": (40, 120, 220),
    },
    {
        "name": "amenity_garden_probe",
        "path": Path("artifacts/test_assets/synthetic_amenities/garden_green.jpg"),
        "expected_amenity": "has_garden",
        "rgb": (70, 170, 80),
    },
    {
        "name": "amenity_parking_probe",
        "path": Path("artifacts/test_assets/synthetic_amenities/parking_gray.jpg"),
        "expected_amenity": "has_parking",
        "rgb": (120, 120, 120),
    },
]


@dataclass
class EvalRow:
    name: str
    model_mode: str
    image_path: str
    top_label: str
    top_score: float
    inferred_property_type: str
    expected_property_type: str
    property_type_hit: bool
    expected_amenity: str
    amenity_hit_top8: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_mode": self.model_mode,
            "image_path": self.image_path,
            "top_label": self.top_label,
            "top_score": round(self.top_score, 4),
            "inferred_property_type": self.inferred_property_type,
            "expected_property_type": self.expected_property_type,
            "property_type_hit": self.property_type_hit,
            "expected_amenity": self.expected_amenity,
            "amenity_hit_top8": self.amenity_hit_top8,
        }


def _ensure_synthetic_images() -> None:
    for case in SYNTHETIC_CASES:
        path = case["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            continue
        img = Image.new("RGB", (224, 224), color=case["rgb"])
        img.save(path)


def _infer_property_type(predictions: list[dict[str, Any]]) -> str:
    for pred in predictions:
        label = str(pred.get("label", "")).strip()
        hint = PROPERTY_HINT_MAP.get(label)
        if hint:
            return hint
    return ""


def _infer_amenities(predictions: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for pred in predictions:
        label = str(pred.get("label", "")).strip()
        mapped = AMENITY_LABEL_MAP.get(label)
        if mapped:
            out.add(mapped)
    return out


def _classify_with_clip_primary(image_paths: list[str]) -> list[dict[str, Any]]:
    service = ImageTypeClassifierService()
    return service.classify_many(image_paths)


def _classify_with_forced_fallback(image_paths: list[str]) -> list[dict[str, Any]]:
    service = ImageTypeClassifierService()
    service._clip_classifier = None
    service._ensure_clip = lambda: None  # type: ignore[method-assign]
    return service.classify_many(image_paths)


def _evaluate_rows(model_mode: str, rows: list[dict[str, Any]], expected_by_path: dict[str, dict[str, Any]]) -> list[EvalRow]:
    out: list[EvalRow] = []
    for row in rows:
        image_path = str(row.get("image_ref", ""))
        preds = list(row.get("predictions") or [])
        top = row.get("top_prediction") or {}

        expected = expected_by_path.get(image_path, {})
        expected_property_type = str(expected.get("expected_property_type", ""))
        expected_amenity = str(expected.get("expected_amenity", ""))

        inferred_property = _infer_property_type(preds)
        inferred_amenities = _infer_amenities(preds)

        property_hit = bool(expected_property_type) and inferred_property == expected_property_type
        amenity_hit = bool(expected_amenity) and expected_amenity in inferred_amenities

        out.append(
            EvalRow(
                name=str(expected.get("name", Path(image_path).stem)),
                model_mode=model_mode,
                image_path=image_path,
                top_label=str(top.get("label", "")),
                top_score=float(top.get("score", 0.0)),
                inferred_property_type=inferred_property,
                expected_property_type=expected_property_type,
                property_type_hit=property_hit,
                expected_amenity=expected_amenity,
                amenity_hit_top8=amenity_hit,
            )
        )
    return out


def _anomaly_scenarios(rows: list[EvalRow]) -> list[dict[str, Any]]:
    by_name = {r.name: r for r in rows}
    scenarios: list[dict[str, Any]] = []

    house = by_name.get("house_facade")
    apartment = by_name.get("apartment_building")
    land = by_name.get("land_plot")

    if house is not None:
        selected = "Appartement"
        mismatch = bool(house.inferred_property_type) and house.inferred_property_type != selected
        scenarios.append(
            {
                "scenario": "wrong_property_type_selected_for_house",
                "selected_property_type": selected,
                "inferred_property_type": house.inferred_property_type,
                "expected_mismatch": True,
                "detected_mismatch": mismatch,
            }
        )

    if apartment is not None:
        selected = "Maison"
        mismatch = bool(apartment.inferred_property_type) and apartment.inferred_property_type != selected
        scenarios.append(
            {
                "scenario": "wrong_property_type_selected_for_apartment",
                "selected_property_type": selected,
                "inferred_property_type": apartment.inferred_property_type,
                "expected_mismatch": True,
                "detected_mismatch": mismatch,
            }
        )

    if land is not None:
        selected_amenities = {"has_pool": True, "has_garden": True}
        predicted_amenities = {"has_pool": land.amenity_hit_top8, "has_garden": land.amenity_hit_top8}
        mismatch = any(selected_amenities[k] and not predicted_amenities[k] for k in selected_amenities)
        scenarios.append(
            {
                "scenario": "wrong_amenities_selected_for_land",
                "selected_amenities": selected_amenities,
                "predicted_positive_amenities": predicted_amenities,
                "expected_mismatch": True,
                "detected_mismatch": mismatch,
            }
        )

    return scenarios


def _summarize(rows: list[EvalRow]) -> dict[str, Any]:
    total_property = sum(1 for r in rows if r.expected_property_type)
    property_hits = sum(1 for r in rows if r.expected_property_type and r.property_type_hit)

    total_amenity = sum(1 for r in rows if r.expected_amenity)
    amenity_hits = sum(1 for r in rows if r.expected_amenity and r.amenity_hit_top8)

    cv_modes = Counter(r.model_mode for r in rows)

    return {
        "property_type_cases": total_property,
        "property_type_hits": property_hits,
        "property_type_accuracy": round((property_hits / total_property) if total_property else 0.0, 4),
        "amenity_cases": total_amenity,
        "amenity_hits_top8": amenity_hits,
        "amenity_hit_rate_top8": round((amenity_hits / total_amenity) if total_amenity else 0.0, 4),
        "mode_counts": dict(cv_modes),
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Vision Runtime Evaluation Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("- CLIP primary model behavior on property-type and amenity probes.")
    lines.append("- Notebook fallback behavior when CLIP is unavailable.")
    lines.append("- Anomaly scenarios for property-type and amenity mismatch detection.")
    lines.append("")

    for section in ["clip_primary", "notebook_fallback"]:
        summary = payload[section]["summary"]
        lines.append(f"## {section.replace('_', ' ').title()}")
        lines.append(f"- Property-type accuracy: {summary['property_type_hits']}/{summary['property_type_cases']} ({summary['property_type_accuracy']:.2%})")
        lines.append(f"- Amenity hit@8: {summary['amenity_hits_top8']}/{summary['amenity_cases']} ({summary['amenity_hit_rate_top8']:.2%})")
        lines.append("")
        lines.append("### Per-case results")
        for row in payload[section]["rows"]:
            lines.append(
                "- "
                f"{row['name']} | mode={row['model_mode']} | top={row['top_label']} ({row['top_score']}) | "
                f"inferred_type={row['inferred_property_type'] or 'n/a'} | expected_type={row['expected_property_type'] or 'n/a'} | "
                f"amenity_target={row['expected_amenity'] or 'n/a'} | amenity_hit_top8={row['amenity_hit_top8']}"
            )
        lines.append("")

    lines.append("## Anomaly Scenarios")
    for scenario in payload["anomaly_scenarios"]:
        lines.append(f"- {json.dumps(scenario, ensure_ascii=True)}")
    lines.append("")

    lines.append("## Notes")
    lines.append("- Amenity probes are synthetic color blocks to test prompt behavior quickly; they are not photoreal samples.")
    lines.append("- For production-level amenity benchmarking, replace synthetic probes with labeled real photos per amenity.")
    return "\n".join(lines) + "\n"


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_synthetic_images()

    cases: list[dict[str, Any]] = []
    cases.extend(LOCAL_IMAGE_CASES)
    for c in SYNTHETIC_CASES:
        cases.append(
            {
                "name": c["name"],
                "path": c["path"],
                "expected_property_type": "",
                "expected_amenity": c["expected_amenity"],
            }
        )

    image_paths = [str(c["path"]) for c in cases if Path(c["path"]).exists()]
    expected_by_path = {str(c["path"]): c for c in cases}

    clip_rows_raw = _classify_with_clip_primary(image_paths)
    clip_eval = _evaluate_rows("clip_feature_inference", clip_rows_raw, expected_by_path)

    fallback_rows_raw = _classify_with_forced_fallback(image_paths)
    fallback_eval = _evaluate_rows("notebook_property_type_fallback", fallback_rows_raw, expected_by_path)

    payload = {
        "clip_primary": {
            "summary": _summarize(clip_eval),
            "rows": [r.to_dict() for r in clip_eval],
        },
        "notebook_fallback": {
            "summary": _summarize(fallback_eval),
            "rows": [r.to_dict() for r in fallback_eval],
        },
        "anomaly_scenarios": _anomaly_scenarios(clip_eval),
    }

    REPORT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    REPORT_MD.write_text(_markdown_report(payload), encoding="utf-8")

    print(f"Saved report JSON to: {REPORT_JSON}")
    print(f"Saved report markdown to: {REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
