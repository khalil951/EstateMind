from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_SUMMARY_JSON = Path("artifacts/reports/summary_dashboard/summary_dashboard.json")
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/summary_dashboard/visualizations")


def _parse_numeric(value: str) -> float | None:
    text = value.strip().replace("~", "").replace(">", "").replace("<", "")
    if not text:
        return None

    if "-" in text and any(ch.isdigit() for ch in text):
        parts = [part.strip() for part in text.split("-") if part.strip()]
        numbers: list[float] = []
        for part in parts:
            match = re.search(r"[-+]?\d*\.?\d+", part)
            if match:
                numbers.append(float(match.group(0)))
        if numbers:
            return sum(numbers) / len(numbers)

    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return None
    return float(match.group(0))


def _normalize(metric_name: str, raw_value: str) -> float | None:
    number = _parse_numeric(raw_value)
    if number is None:
        return None

    name = metric_name.lower()
    if "mape" in name:
        return max(0.0, 100.0 - number)
    if raw_value.strip().endswith("%"):
        return number
    if "r2" in name or "auc" in name or "f1" in name:
        return number * 100.0 if number <= 1.2 else number
    return number


def _find_metric(sections: dict, metric_name: str) -> str:
    for items in sections.values():
        for item in items:
            if item.get("metric") == metric_name:
                return str(item.get("result", ""))
    return ""


def _save_bar_plot(labels: list[str], values: list[float], title: str, ylabel: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    bars = ax.bar(labels, values, color=["#2f80ed", "#4fd1c5", "#f6b73c", "#ff7a59", "#7c9cff", "#9ad0f5"])
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(values) * 1.2 if values else 100)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.03,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(output_path, format="png")
    plt.close(fig)


def _save_feature_group_plot(output_path: Path) -> None:
    # Feature-group counts come directly from the screenshot bullets.
    labels = ["Financial", "Risk", "Spatial", "Temporal"]
    values = [3, 3, 3, 2]
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    bars = ax.bar(labels, values, color=["#2f80ed", "#4fd1c5", "#f6b73c", "#ff7a59"])
    ax.set_title("Feature Engineering Coverage (from screenshot)", fontsize=14, weight="bold")
    ax.set_ylabel("Feature count")
    ax.set_ylim(0, 4)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.06,
            str(value),
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )

    plt.tight_layout()
    fig.savefig(output_path, format="png")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate PNG summary visualizations from the summary dashboard JSON.")
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON, help="Path to summary_dashboard.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated PNG files")
    args = parser.parse_args()

    if not args.summary_json.exists():
        raise FileNotFoundError(f"Summary JSON not found: {args.summary_json}")

    payload = json.loads(args.summary_json.read_text(encoding="utf-8"))
    sections = payload.get("sections", {})

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_labels = ["Ridge", "CatBoost", "Ensemble"]
    r2_values = [
        _normalize("Ridge R2", _find_metric(sections, "Ridge R2")) or 0.0,
        _normalize("CatBoost R2", _find_metric(sections, "CatBoost R2")) or 0.0,
        _normalize("Ensemble R2", _find_metric(sections, "Ensemble R2")) or 0.0,
    ]
    _save_bar_plot(
        labels=model_labels,
        values=r2_values,
        title="Model Comparison (R2 Scaled to 100)",
        ylabel="Score",
        output_path=args.output_dir / "model_comparison_r2.png",
    )

    eval_metric_names = ["Precision@5", "Grounded Answers", "Hallucination Reduction"]
    eval_values = [
        _normalize(name, _find_metric(sections, name)) or 0.0 for name in eval_metric_names
    ]
    _save_bar_plot(
        labels=eval_metric_names,
        values=eval_values,
        title="Evaluation Metrics",
        ylabel="Percent",
        output_path=args.output_dir / "evaluation_metrics.png",
    )

    perf_metric_names = ["Precision", "Recall", "ROC-AUC", "Portfolio return uplift"]
    perf_values = [
        _normalize(name, _find_metric(sections, name)) or 0.0 for name in perf_metric_names
    ]
    _save_bar_plot(
        labels=perf_metric_names,
        values=perf_values,
        title="Evaluation & Performance (from screenshot)",
        ylabel="Score",
        output_path=args.output_dir / "performance_snapshot.png",
    )

    _save_feature_group_plot(
        output_path=args.output_dir / "feature_engineering_groups.png",
    )

    support_metric_names = ["ResNet50 accuracy", "Sentiment F1", "Random Forest Regressor"]
    support_values = [
        _normalize(name, _find_metric(sections, name)) or 0.0 for name in support_metric_names
    ]
    _save_bar_plot(
        labels=support_metric_names,
        values=support_values,
        title="Supporting Model Metrics",
        ylabel="Score",
        output_path=args.output_dir / "supporting_metrics.png",
    )

    print(f"Saved PNG files to: {args.output_dir}")
    print("- model_comparison_r2.png")
    print("- evaluation_metrics.png")
    print("- performance_snapshot.png")
    print("- feature_engineering_groups.png")
    print("- supporting_metrics.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())