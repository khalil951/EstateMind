from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_REPORT_DIR = Path("artifacts/reports/summary_dashboard")
DEFAULT_JSON_PATH = DEFAULT_REPORT_DIR / "summary_dashboard.json"
DEFAULT_MD_PATH = DEFAULT_REPORT_DIR / "summary_dashboard.md"


@dataclass(frozen=True)
class MetricRow:
    section: str
    metric: str
    result: str
    note: str = ""


def _dashboard_rows() -> list[MetricRow]:
    return [
        MetricRow("Model Comparison", "Ridge R2", "0.55", "Baseline regression model from the screenshot."),
        MetricRow("Model Comparison", "Ridge MAPE", "~12%", "Higher error than CatBoost."),
        MetricRow("Model Comparison", "CatBoost R2", "0.72", "Stronger regression fit."),
        MetricRow("Model Comparison", "CatBoost MAPE", "6.5-7.5%", "Best model in the first comparison card."),
        MetricRow("Ensemble Snapshot", "Ensemble R2", "0.76", "Top regression score in the visual."),
        MetricRow("Ensemble Snapshot", "Ensemble MAPE", "<6.5%", "Lowest error in the comparison set."),
        MetricRow("Ensemble Snapshot", "ResNet50 accuracy", "91.8%", "Image model support metric."),
        MetricRow("Ensemble Snapshot", "Sentiment F1", "~0.695", "Text model support metric."),
        MetricRow("Ensemble Snapshot", "Random Forest Regressor", "~90%", "Supporting model metric from the screenshot."),
        MetricRow("Evaluation", "Precision@5", "88%", "Ranking quality metric."),
        MetricRow("Evaluation", "Grounded Answers", "82%", "Answer grounding quality."),
        MetricRow("Evaluation", "Hallucination Reduction", "65%", "Reported reduction value."),
        MetricRow("Evaluation", "Response Time", "~0.013s", "Fast response shown in the visual."),
        MetricRow("Feature Engineering", "Price appreciation forecast", "6-24 months", "Forecast horizon used in the feature set."),
        MetricRow("Feature Engineering", "Rental yield estimation", "Included", "Financial feature present in the visual."),
        MetricRow("Feature Engineering", "Cost-adjusted ROI", "Included", "Financial feature present in the visual."),
        MetricRow("Feature Engineering", "Climate risk score", "RF model", "Risk feature derived from a random forest model."),
        MetricRow("Feature Engineering", "Market volatility index", "Included", "Risk feature present in the visual."),
        MetricRow("Feature Engineering", "Liquidity risk", "Included", "Risk feature present in the visual."),
        MetricRow("Feature Engineering", "Neighborhood demand intensity", "Included", "Spatial feature present in the visual."),
        MetricRow("Feature Engineering", "Infrastructure score", "Included", "Spatial feature present in the visual."),
        MetricRow("Feature Engineering", "Accessibility index", "Included", "Spatial feature present in the visual."),
        MetricRow("Feature Engineering", "Seasonal demand variation", "Included", "Temporal feature present in the visual."),
        MetricRow("Feature Engineering", "Time-on-market trends", "Included", "Temporal feature present in the visual."),
        MetricRow("Performance", "Precision", "85%", "Top-opportunity precision from the evaluation card."),
        MetricRow("Performance", "Recall", "72%", "Top-opportunity recall from the evaluation card."),
        MetricRow("Performance", "ROC-AUC", "0.82", "Model discrimination score."),
        MetricRow("Performance", "Portfolio return uplift", "+18-25% vs baseline", "Reported uplift over the baseline portfolio."),
        MetricRow("Random Forest", "Test R2", "0.73", "Regression performance shown in the visual."),
        MetricRow("Random Forest", "CV R2", "0.71", "Cross-validation score shown in the visual."),
        MetricRow("Random Forest", "Feature stability", "High", "Stability judgment shown in the visual."),
    ]


def _section_titles() -> list[str]:
    return [
        "Model Comparison",
        "Ensemble Snapshot",
        "Evaluation",
        "Feature Engineering",
        "Performance",
        "Random Forest",
    ]


def _build_payload() -> dict[str, Any]:
    rows = _dashboard_rows()
    sections: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        sections.setdefault(row.section, []).append(
            {
                "metric": row.metric,
                "result": row.result,
                "note": row.note,
            }
        )

    return {
        "title": "EstateMind Summary Dashboard",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sections": sections,
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# {payload['title']}")
    lines.append("")
    lines.append(f"Generated at: {payload['generated_at']}")
    lines.append("")
    lines.append("This dashboard is a compact summary of the metrics visible in the provided visualizations.")
    lines.append("")

    for section in _section_titles():
        items = payload["sections"].get(section, [])
        if not items:
            continue
        lines.append(f"## {section}")
        lines.append("| Metric | Result | Note |")
        lines.append("| --- | --- | --- |")
        for item in items:
            note = item["note"] or ""
            lines.append(f"| {item['metric']} | {item['result']} | {note} |")
        lines.append("")

    lines.append("## Quick Read")
    lines.append("- Best regression score in the screenshots: Ensemble R2 at 0.76 with MAPE below 6.5%.")
    lines.append("- Strong supporting metrics include ResNet50 accuracy at 91.8% and Random Forest Test R2 at 0.73.")
    lines.append("- The evaluation card shows Precision@5 at 88% and response time around 0.013s.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a Markdown summary dashboard from screenshot metrics.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR, help="Directory for dashboard outputs.")
    parser.add_argument("--json-path", type=Path, default=DEFAULT_JSON_PATH, help="Path for the JSON payload.")
    parser.add_argument("--md-path", type=Path, default=DEFAULT_MD_PATH, help="Path for the Markdown dashboard.")
    args = parser.parse_args()

    payload = _build_payload()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.md_path.write_text(_render_markdown(payload), encoding="utf-8")

    print(f"Saved JSON dashboard to: {args.json_path}")
    print(f"Saved Markdown dashboard to: {args.md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())