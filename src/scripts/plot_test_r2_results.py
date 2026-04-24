from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT = Path("artifacts/reports/ml_reports/training_estateprocessor_results.csv")
DEFAULT_OUTPUT = Path("artifacts/reports/ml_reports/test_r2_barplot.png")


def _label_row(row: pd.Series) -> str:
    scope = str(row.get("scope", "")).strip()
    property_type = str(row.get("property_type", "")).strip()
    approach = str(row.get("approach", "")).strip()

    if scope == "global":
        return "Global"
    if scope == "by_type" and property_type:
        return property_type
    if scope == "by_type_terrain_experiment" and approach:
        return f"Terrain ({approach})"
    if property_type:
        return property_type
    return scope or "Unknown"


def _prepare_frame(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    frame["label"] = frame.apply(_label_row, axis=1)
    frame["test_r2"] = pd.to_numeric(frame["test_r2"], errors="coerce")
    frame = frame.dropna(subset=["test_r2"])

    # Keep the accepted serving rows for a concise dashboard view.
    accepted = frame[frame["status"].astype(str).str.lower().eq("accepted")].copy()
    accepted = accepted.sort_values("test_r2", ascending=False)
    return accepted


def _plot_test_r2(frame: pd.DataFrame, output_path: Path) -> None:
    labels = frame["label"].tolist()
    values = frame["test_r2"].tolist()

    colors = ["#2f80ed", "#4fd1c5", "#f6b73c", "#ff7a59", "#7c9cff"][: len(values)]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    bars = ax.bar(labels, values, color=colors, edgecolor="#0e1c2f", linewidth=0.8)

    ax.set_title("Training EstateProcessor Test R2", fontsize=15, weight="bold")
    ax.set_ylabel("Test R2")
    ax.set_ylim(0, max(values) * 1.2 if values else 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="png")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot a bar chart of accepted test_r2 results.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to training_estateprocessor_results.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="PNG output path")
    args = parser.parse_args()

    frame = _prepare_frame(args.input)
    if frame.empty:
        raise RuntimeError("No accepted rows with test_r2 values were found in the input CSV.")

    _plot_test_r2(frame, args.output)
    print(f"Saved test_r2 plot to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())