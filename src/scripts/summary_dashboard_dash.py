from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from dash import Dash, dcc, html
import plotly.graph_objects as go


SUMMARY_JSON_PATH = Path("artifacts/reports/summary_dashboard/summary_dashboard.json")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8050


def _default_payload() -> dict[str, Any]:
    return {
        "title": "EstateMind Summary Dashboard",
        "generated_at": "from screenshot summary",
        "sections": {
            "Model Comparison": [
                {"metric": "Ridge R2", "result": "0.55", "note": "Baseline regression model from the screenshot."},
                {"metric": "Ridge MAPE", "result": "~12%", "note": "Higher error than CatBoost."},
                {"metric": "CatBoost R2", "result": "0.72", "note": "Stronger regression fit."},
                {"metric": "CatBoost MAPE", "result": "6.5-7.5%", "note": "Best model in the first comparison card."},
            ],
            "Ensemble Snapshot": [
                {"metric": "Ensemble R2", "result": "0.76", "note": "Top regression score in the visual."},
                {"metric": "Ensemble MAPE", "result": "<6.5%", "note": "Lowest error in the comparison set."},
                {"metric": "ResNet50 accuracy", "result": "91.8%", "note": "Image model support metric."},
                {"metric": "Sentiment F1", "result": "~0.695", "note": "Text model support metric."},
                {"metric": "Random Forest Regressor", "result": "~90%", "note": "Supporting model metric from the screenshot."},
            ],
            "Evaluation": [
                {"metric": "Precision@5", "result": "88%", "note": "Ranking quality metric."},
                {"metric": "Grounded Answers", "result": "82%", "note": "Answer grounding quality."},
                {"metric": "Hallucination Reduction", "result": "65%", "note": "Reported reduction value."},
                {"metric": "Response Time", "result": "~0.013s", "note": "Fast response shown in the visual."},
            ],
            "Feature Engineering": [
                {"metric": "Price appreciation forecast", "result": "6-24 months", "note": "Forecast horizon used in the feature set."},
                {"metric": "Rental yield estimation", "result": "Included", "note": "Financial feature present in the visual."},
                {"metric": "Cost-adjusted ROI", "result": "Included", "note": "Financial feature present in the visual."},
                {"metric": "Climate risk score", "result": "RF model", "note": "Risk feature derived from a random forest model."},
                {"metric": "Market volatility index", "result": "Included", "note": "Risk feature present in the visual."},
                {"metric": "Liquidity risk", "result": "Included", "note": "Risk feature present in the visual."},
                {"metric": "Neighborhood demand intensity", "result": "Included", "note": "Spatial feature present in the visual."},
                {"metric": "Infrastructure score", "result": "Included", "note": "Spatial feature present in the visual."},
                {"metric": "Accessibility index", "result": "Included", "note": "Spatial feature present in the visual."},
                {"metric": "Seasonal demand variation", "result": "Included", "note": "Temporal feature present in the visual."},
                {"metric": "Time-on-market trends", "result": "Included", "note": "Temporal feature present in the visual."},
            ],
            "Performance": [
                {"metric": "Precision", "result": "85%", "note": "Top-opportunity precision from the evaluation card."},
                {"metric": "Recall", "result": "72%", "note": "Top-opportunity recall from the evaluation card."},
                {"metric": "ROC-AUC", "result": "0.82", "note": "Model discrimination score."},
                {"metric": "Portfolio return uplift", "result": "+18-25% vs baseline", "note": "Reported uplift over the baseline portfolio."},
            ],
            "Random Forest": [
                {"metric": "Test R2", "result": "0.73", "note": "Regression performance shown in the visual."},
                {"metric": "CV R2", "result": "0.71", "note": "Cross-validation score shown in the visual."},
                {"metric": "Feature stability", "result": "High", "note": "Stability judgment shown in the visual."},
            ],
        },
    }


def _load_payload(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return _default_payload()


def _find_metric(payload: dict[str, Any], metric_name: str) -> str:
    for section_items in payload.get("sections", {}).values():
        for item in section_items:
            if item.get("metric") == metric_name:
                return str(item.get("result", ""))
    return ""


def _parse_number(value: str) -> float | None:
    text = value.strip().replace("~", "").replace(">", "").replace("<", "")
    if not text:
        return None

    if "-" in text and any(char.isdigit() for char in text):
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


def _chart_score(metric_name: str, value_text: str) -> float | None:
    number = _parse_number(value_text)
    if number is None:
        return None

    normalized = value_text.strip().lower()
    if "mape" in metric_name.lower():
        return max(0.0, 100.0 - number)
    if normalized.endswith("%"):
        return number
    if metric_name.lower() in {"roc-auc", "sentiment f1", "test r2", "cv r2", "r2", "ensemble r2", "ridge r2", "catboost r2"}:
        return number * 100.0 if number <= 1.2 else number
    if number <= 1.2 and any(token in metric_name.lower() for token in ["precision", "recall", "accuracy", "f1", "auc"]):
        return number * 100.0
    if number <= 1.2 and value_text.strip().startswith("0."):
        return number * 100.0
    return number


def _section_counts(payload: dict[str, Any]) -> dict[str, int]:
    return {section: len(items) for section, items in payload.get("sections", {}).items()}


def _metric_card(title: str, value: str, caption: str, accent: str) -> html.Div:
    return html.Div(
        [
            html.Div(title, className="metric-card-title"),
            html.Div(value, className="metric-card-value"),
            html.Div(caption, className="metric-card-caption"),
        ],
        className="metric-card",
        style={"borderTop": f"3px solid {accent}"},
    )


def _build_model_comparison_figure(payload: dict[str, Any]) -> go.Figure:
    models = ["Ridge", "CatBoost", "Ensemble"]
    r2_values = [
        _chart_score("Ridge R2", _find_metric(payload, "Ridge R2")),
        _chart_score("CatBoost R2", _find_metric(payload, "CatBoost R2")),
        _chart_score("Ensemble R2", _find_metric(payload, "Ensemble R2")),
    ]
    mape_scores = [
        _chart_score("Ridge MAPE", _find_metric(payload, "Ridge MAPE")),
        _chart_score("CatBoost MAPE", _find_metric(payload, "CatBoost MAPE")),
        _chart_score("Ensemble MAPE", _find_metric(payload, "Ensemble MAPE")),
    ]

    fig = go.Figure()
    fig.add_bar(
        name="R2",
        x=models,
        y=r2_values,
        marker_color="#4fd1c5",
        text=[_find_metric(payload, f"{model} R2") for model in models],
        textposition="outside",
    )
    fig.add_bar(
        name="Lower-error score",
        x=models,
        y=mape_scores,
        marker_color="#f6b73c",
        text=[_find_metric(payload, f"{model} MAPE") for model in models],
        textposition="outside",
    )
    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,17,31,0.2)",
        font={"family": "Manrope, sans-serif", "color": "#e7eef8"},
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
        height=360,
        yaxis={"range": [0, 110], "gridcolor": "#20324a", "title": "Score"},
        xaxis={"title": "Model"},
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    return fig


def _build_performance_figure(payload: dict[str, Any]) -> go.Figure:
    metric_names = [
        "ResNet50 accuracy",
        "Precision@5",
        "Grounded Answers",
        "Hallucination Reduction",
        "Precision",
        "Recall",
        "ROC-AUC",
        "Test R2",
        "CV R2",
        "Sentiment F1",
        "Random Forest Regressor",
    ]
    rows: list[tuple[str, float]] = []
    for metric_name in metric_names:
        raw_value = _find_metric(payload, metric_name)
        score = _chart_score(metric_name, raw_value)
        if score is not None:
            rows.append((metric_name, score))

    rows.sort(key=lambda item: item[1], reverse=True)
    labels = [item[0] for item in rows]
    values = [item[1] for item in rows]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={
                "color": values,
                "colorscale": [[0.0, "#1d3557"], [0.5, "#4fd1c5"], [1.0, "#f6b73c"]],
                "line": {"color": "rgba(255,255,255,0.12)", "width": 1},
            },
            text=[f"{value:.1f}" for value in values],
            textposition="outside",
            hovertemplate="%{y}<br>Score: %{x:.1f}<extra></extra>",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,17,31,0.2)",
        font={"family": "Manrope, sans-serif", "color": "#e7eef8"},
        margin={"l": 180, "r": 20, "t": 30, "b": 20},
        height=460,
        xaxis={"range": [0, 100], "gridcolor": "#20324a", "title": "Normalized score"},
        yaxis={"autorange": "reversed"},
    )
    return fig


def _build_section_mix_figure(payload: dict[str, Any]) -> go.Figure:
    section_counts = _section_counts(payload)
    fig = go.Figure(
        go.Pie(
            labels=list(section_counts.keys()),
            values=list(section_counts.values()),
            hole=0.58,
            marker={"colors": ["#4fd1c5", "#2f80ed", "#f6b73c", "#ff7a59", "#9ad0f5", "#7c9cff"]},
            textinfo="label+percent",
            textfont={"size": 13},
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Manrope, sans-serif", "color": "#e7eef8"},
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        height=360,
        showlegend=False,
        annotations=[
            {
                "text": f"{sum(section_counts.values())}<br>metrics",
                "showarrow": False,
                "font": {"size": 22, "color": "#ffffff"},
            }
        ],
    )
    return fig


def _build_gauge_figure(payload: dict[str, Any]) -> go.Figure:
    value = _chart_score("Ensemble R2", _find_metric(payload, "Ensemble R2")) or 0.0
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": " / 100", "font": {"size": 38, "color": "#ffffff"}},
            title={"text": "Ensemble R2 strength", "font": {"size": 18, "color": "#c8d7ea"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#93a4bd"},
                "bar": {"color": "#4fd1c5"},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "#14243a"},
                    {"range": [50, 75], "color": "#1b3350"},
                    {"range": [75, 100], "color": "#21496f"},
                ],
                "threshold": {"line": {"color": "#f6b73c", "width": 4}, "thickness": 0.7, "value": 76},
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        height=300,
        font={"family": "Manrope, sans-serif", "color": "#e7eef8"},
    )
    return fig


def _build_section_table(payload: dict[str, Any]) -> html.Div:
    blocks: list[html.Div] = []
    for section_name, items in payload.get("sections", {}).items():
        rows = [
            html.Tr(
                [
                    html.Th("Metric", className="table-head"),
                    html.Th("Result", className="table-head"),
                    html.Th("Note", className="table-head"),
                ]
            )
        ]
        for item in items:
            rows.append(
                html.Tr(
                    [
                        html.Td(item.get("metric", ""), className="table-cell metric-cell"),
                        html.Td(item.get("result", ""), className="table-cell result-cell"),
                        html.Td(item.get("note", ""), className="table-cell note-cell"),
                    ]
                )
            )
        blocks.append(
            html.Div(
                [
                    html.H3(section_name, className="section-title"),
                    html.Table(rows, className="summary-table"),
                ],
                className="section-block",
            )
        )
    return html.Div(blocks, className="section-stack")


def create_app(summary_path: Path = SUMMARY_JSON_PATH) -> Dash:
    payload = _load_payload(summary_path)
    app = Dash(__name__)
    app.title = "EstateMind Summary Dashboard"
    app.index_string = """
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>EstateMind Summary Dashboard</title>
            {%favicon%}
            {%css%}
            <link rel="preconnect" href="https://fonts.googleapis.com">
            <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
            <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap" rel="stylesheet">
            <style>
                * { box-sizing: border-box; }
                body {
                    margin: 0;
                    background:
                        radial-gradient(circle at top left, rgba(79, 209, 197, 0.16), transparent 28%),
                        radial-gradient(circle at top right, rgba(246, 183, 60, 0.16), transparent 26%),
                        linear-gradient(180deg, #05101d 0%, #0b1727 60%, #09111d 100%);
                    color: #e7eef8;
                    font-family: 'Manrope', sans-serif;
                }
                a { color: inherit; }
                .page-shell {
                    max-width: 1480px;
                    margin: 0 auto;
                    padding: 28px 24px 40px;
                }
                .hero {
                    position: relative;
                    overflow: hidden;
                    padding: 28px 28px 24px;
                    border: 1px solid rgba(255,255,255,0.08);
                    border-radius: 24px;
                    background: linear-gradient(135deg, rgba(9, 20, 35, 0.96), rgba(16, 37, 63, 0.9));
                    box-shadow: 0 24px 70px rgba(0, 0, 0, 0.28);
                }
                .hero::after {
                    content: '';
                    position: absolute;
                    inset: 0;
                    background: linear-gradient(120deg, rgba(79,209,197,0.08), transparent 40%, rgba(246,183,60,0.06));
                    pointer-events: none;
                }
                .hero-top {
                    position: relative;
                    z-index: 1;
                    display: flex;
                    align-items: flex-start;
                    justify-content: space-between;
                    gap: 20px;
                }
                .hero h1 {
                    margin: 0;
                    font-size: clamp(2rem, 4vw, 3.6rem);
                    line-height: 1;
                    letter-spacing: -0.04em;
                }
                .hero p {
                    margin: 12px 0 0;
                    max-width: 820px;
                    color: #c7d3e6;
                    line-height: 1.6;
                    font-size: 1rem;
                }
                .hero-badges {
                    display: flex;
                    gap: 10px;
                    flex-wrap: wrap;
                    margin-top: 18px;
                }
                .badge {
                    display: inline-flex;
                    align-items: center;
                    padding: 8px 12px;
                    border-radius: 999px;
                    background: rgba(255,255,255,0.08);
                    border: 1px solid rgba(255,255,255,0.12);
                    color: #f3f7ff;
                    font-size: 0.86rem;
                    font-weight: 700;
                }
                .kpi-grid {
                    margin-top: 18px;
                    display: grid;
                    grid-template-columns: repeat(4, minmax(0, 1fr));
                    gap: 16px;
                }
                .metric-card,
                .panel,
                .section-block {
                    border: 1px solid rgba(255,255,255,0.08);
                    background: linear-gradient(180deg, rgba(12, 24, 40, 0.92), rgba(12, 24, 40, 0.82));
                    border-radius: 22px;
                    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.22);
                }
                .metric-card {
                    padding: 18px 18px 16px;
                    min-height: 112px;
                }
                .metric-card-title {
                    color: #99adc6;
                    font-size: 0.88rem;
                    font-weight: 700;
                    letter-spacing: 0.02em;
                    text-transform: uppercase;
                }
                .metric-card-value {
                    margin-top: 10px;
                    font-size: 2rem;
                    font-weight: 800;
                    color: #ffffff;
                    letter-spacing: -0.03em;
                }
                .metric-card-caption {
                    margin-top: 6px;
                    color: #c7d3e6;
                    font-size: 0.94rem;
                    line-height: 1.45;
                }
                .grid-row {
                    margin-top: 18px;
                    display: grid;
                    grid-template-columns: 2.1fr 1fr;
                    gap: 16px;
                }
                .panel {
                    padding: 18px 18px 8px;
                }
                .panel h2,
                .section-title {
                    margin: 0 0 8px;
                    font-size: 1.1rem;
                    letter-spacing: -0.02em;
                }
                .panel p {
                    margin: 0 0 12px;
                    color: #c7d3e6;
                }
                .section-stack {
                    margin-top: 18px;
                    display: grid;
                    gap: 16px;
                }
                .section-block {
                    padding: 18px;
                }
                .summary-table {
                    width: 100%;
                    border-collapse: collapse;
                    overflow: hidden;
                    border-radius: 18px;
                }
                .table-head {
                    text-align: left;
                    color: #88a0be;
                    font-size: 0.8rem;
                    font-weight: 800;
                    text-transform: uppercase;
                    letter-spacing: 0.04em;
                    padding: 12px 12px 10px;
                    border-bottom: 1px solid rgba(255,255,255,0.08);
                }
                .table-cell {
                    padding: 12px;
                    vertical-align: top;
                    border-bottom: 1px solid rgba(255,255,255,0.06);
                    color: #e7eef8;
                }
                .metric-cell { width: 28%; font-weight: 700; }
                .result-cell { width: 18%; color: #f6b73c; font-weight: 800; }
                .note-cell { color: #c7d3e6; }
                .footer-note {
                    margin-top: 16px;
                    color: #93a4bd;
                    font-size: 0.9rem;
                }
                @media (max-width: 1200px) {
                    .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
                    .grid-row { grid-template-columns: 1fr; }
                }
                @media (max-width: 720px) {
                    .page-shell { padding: 16px; }
                    .hero { padding: 20px; }
                    .kpi-grid { grid-template-columns: 1fr; }
                    .metric-card-value { font-size: 1.7rem; }
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    """

    section_counts = _section_counts(payload)
    total_metrics = sum(section_counts.values())
    key_metrics = [
        ("Ensemble R2", _find_metric(payload, "Ensemble R2"), "Highest regression score in the screenshots.", "#4fd1c5"),
        ("Ensemble MAPE", _find_metric(payload, "Ensemble MAPE"), "Lowest error in the comparison set.", "#f6b73c"),
        ("ResNet50 accuracy", _find_metric(payload, "ResNet50 accuracy"), "Image model support signal.", "#2f80ed"),
        ("Precision@5", _find_metric(payload, "Precision@5"), "Top-opportunity retrieval quality.", "#ff7a59"),
    ]

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H1(payload.get("title", "EstateMind Summary Dashboard")),
                            html.P(
                                "A Dash + Plotly dashboard built directly from the screenshot metrics. It keeps the summary compact, but the visuals make the strongest signals easy to compare at a glance."
                            ),
                            html.Div(
                                [
                                    html.Span(f"{total_metrics} metrics captured", className="badge"),
                                    html.Span(f"Last generated: {payload.get('generated_at', 'n/a')}", className="badge"),
                                    html.Span("Dash + Plotly", className="badge"),
                                ],
                                className="hero-badges",
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Span("Screenshot summary", className="badge"),
                            html.Span(f"{len(section_counts)} sections", className="badge"),
                        ],
                        className="hero-badges",
                    ),
                ],
                className="hero-top",
            ),
            html.Div(
                [
                    _metric_card(title, value, caption, accent)
                    for title, value, caption, accent in key_metrics
                ],
                className="kpi-grid",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Model comparison"),
                            html.P("R2 and lower-error scores for the three model cards shown in the screenshots."),
                            dcc.Graph(figure=_build_model_comparison_figure(payload), config={"displayModeBar": False, "responsive": True}),
                        ],
                        className="panel",
                    ),
                    html.Div(
                        [
                            html.H2("Section mix"),
                            html.P("How many metrics each dashboard section contributes."),
                            dcc.Graph(figure=_build_section_mix_figure(payload), config={"displayModeBar": False, "responsive": True}),
                            dcc.Graph(figure=_build_gauge_figure(payload), config={"displayModeBar": False, "responsive": True}),
                        ],
                        className="panel",
                    ),
                ],
                className="grid-row",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Normalized performance scoreboard"),
                            html.P("All high-signal percentage-like metrics normalized onto a 0-100 scale for quick reading."),
                            dcc.Graph(figure=_build_performance_figure(payload), config={"displayModeBar": False, "responsive": True}),
                        ],
                        className="panel",
                    ),
                ],
                className="section-block",
            ),
            _build_section_table(payload),
            html.Div(
                "The dashboard is intentionally screenshot-driven: it visualizes only the metrics already present in the summary cards and reports.",
                className="footer-note",
            ),
        ],
        className="page-shell",
    )

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the EstateMind summary dashboard.")
    parser.add_argument("--summary-json", type=Path, default=SUMMARY_JSON_PATH, help="Summary JSON generated from the screenshot metrics.")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="Host to bind the Dash server to.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind the Dash server to.")
    parser.add_argument("--debug", action="store_true", help="Enable Dash debug mode.")
    args = parser.parse_args()

    app = create_app(args.summary_json)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())