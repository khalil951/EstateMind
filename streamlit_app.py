from __future__ import annotations

from io import BytesIO
from typing import Dict, List

import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image
from streamlit.errors import StreamlitSecretNotFoundError

st.set_page_config(page_title="EstateMind - Tunisia Real Estate Intelligence Platform", page_icon="🏠", layout="wide")

PRIMARY = "#2563eb"
SECONDARY = "#0d9488"
ACCENT = "#f59e0b"
SUCCESS = "#10b981"
DANGER = "#ef4444"

def resolve_api_url() -> str:
    env_url = os.getenv("ESTATEMIND_API_URL")
    if env_url:
        return env_url
    try:
        return st.secrets["api_url"]
    except (StreamlitSecretNotFoundError, KeyError):
        return "http://127.0.0.1:8000"


API_URL = resolve_api_url()


def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --primary: {PRIMARY};
            --secondary: {SECONDARY};
            --accent: {ACCENT};
            --success: {SUCCESS};
            --danger: {DANGER};
        }}
        .stApp {{
            background: radial-gradient(circle at top right, #dbeafe 0%, #f8fafc 45%, #ffffff 100%);
            color: #0f172a;
        }}
        .brand-header {{
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 18px 22px;
            box-shadow: 0 8px 30px rgba(37, 99, 235, 0.08);
            margin-bottom: 14px;
        }}
        .brand-title {{ font-size: 1.8rem; font-weight: 800; color: var(--primary); }}
        .brand-sub {{ color: #334155; font-size: 1rem; margin-top: 4px; }}
        .card {{
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 16px;
            box-shadow: 0 6px 20px rgba(2, 6, 23, 0.05);
            transition: all 0.25s ease;
        }}
        .card:hover {{ transform: translateY(-2px); box-shadow: 0 10px 24px rgba(2, 6, 23, 0.08); }}
        .metric-main {{ font-size: 2.05rem; font-weight: 800; color: var(--primary); line-height: 1.1; }}
        .metric-sub {{ color: #475569; font-size: 0.96rem; }}
        .conf-wrap {{
            width: 140px; height: 140px; border-radius: 999px; display: flex;
            align-items: center; justify-content: center; flex-direction: column;
            margin: 10px auto; color: white; font-weight: 700;
        }}
        .footer {{
            margin-top: 24px; padding: 14px 0 6px 0; border-top: 1px solid #e2e8f0;
            color: #475569; text-align: center; font-size: 0.9rem;
        }}
        .socials a {{ margin: 0 8px; text-decoration: none; color: var(--primary); font-weight: 600; }}
        @media (max-width: 900px) {{
            .brand-title {{ font-size: 1.45rem; }}
            .metric-main {{ font-size: 1.7rem; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_payload(state: Dict, image_count: int) -> Dict:
    return {
        "property_type": state["property_type"],
        "governorate": state["governorate"],
        "city": state["city"],
        "neighborhood": state["neighborhood"],
        "size_m2": state["size_m2"],
        "bedrooms": state.get("bedrooms", 0),
        "bathrooms": state.get("bathrooms", 0),
        "condition": state["condition"],
        "has_pool": state["has_pool"],
        "has_garden": state["has_garden"],
        "has_parking": state["has_parking"],
        "sea_view": state["sea_view"],
        "elevator": state.get("elevator", False),
        "description": state["description"],
        "uploaded_images_count": image_count,
    }


def try_sample_prefill() -> None:
    st.session_state["property_type"] = "Appartement"
    st.session_state["governorate"] = "Tunis"
    st.session_state["city"] = "La Marsa"
    st.session_state["neighborhood"] = "Sidi Abdelaziz"
    st.session_state["size_m2"] = 120
    st.session_state["bedrooms"] = 3
    st.session_state["bathrooms"] = 2
    st.session_state["condition"] = "Excellent"
    st.session_state["has_pool"] = False
    st.session_state["has_garden"] = True
    st.session_state["has_parking"] = True
    st.session_state["sea_view"] = True
    st.session_state["elevator"] = True
    st.session_state["description"] = "Appartement renove avec vue mer exceptionnelle, cuisine moderne et emplacement premium a La Marsa."


def confidence_color(level: str) -> str:
    if level == "High":
        return SUCCESS
    if level == "Medium":
        return ACCENT
    return DANGER


def format_tnd(value: int) -> str:
    return f"{value:,} TND"


inject_styles()

if "results" not in st.session_state:
    st.session_state["results"] = None
if "carousel_start" not in st.session_state:
    st.session_state["carousel_start"] = 0

st.markdown(
    """
    <div class="brand-header">
        <div class="brand-title">🏠 EstateMind</div>
        <div class="brand-sub">AI-Powered Property Valuation for Tunisia</div>
    </div>
    """,
    unsafe_allow_html=True,
)

progress_col1, progress_col2, progress_col3 = st.columns(3)
with progress_col1:
    st.info("1) Input", icon="🧾")
with progress_col2:
    st.info("2) Analysis", icon="🧠")
with progress_col3:
    st.info("3) Results", icon="📈")

left_col, center_col, right_col = st.columns([3, 4, 3], gap="large")

with left_col:
    st.markdown("### Property Input Form")
    if st.button("Try Sample Property", type="secondary", use_container_width=True):
        try_sample_prefill()
        st.toast("Sample property loaded")

    property_type = st.radio(
        "Property Type",
        ["Terrain", "Maison", "Appartement"],
        horizontal=False,
        key="property_type",
        format_func=lambda x: {"Terrain": "🏗️ Terrain (Land)", "Maison": "🏠 Maison (House)", "Appartement": "🏢 Appartement (Apartment)"}[x],
    )

    governorate = st.selectbox(
        "Governorate",
        ["Tunis", "Ariana", "Ben Arous", "Sousse", "Sfax", "Nabeul", "Monastir", "Bizerte", "Kairouan", "Gabes"],
        key="governorate",
    )
    city = st.text_input("City", key="city", placeholder="e.g., La Marsa")
    neighborhood = st.text_input("Neighborhood (optional)", key="neighborhood", placeholder="e.g., Cite Ennasr")

    size_m2 = st.number_input("Size (m2)", min_value=20, max_value=2000, value=120, step=5, key="size_m2")

    bedrooms = 0
    bathrooms = 0
    if property_type in ["Maison", "Appartement"]:
        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=12, value=3, step=1, key="bedrooms")
        bathrooms = st.number_input("Bathrooms", min_value=0, max_value=8, value=2, step=1, key="bathrooms")

    condition = st.selectbox("Condition", ["New", "Excellent", "Good", "Fair", "Needs Renovation"], key="condition")

    st.markdown("#### Features")
    has_pool = st.checkbox("Has Pool", key="has_pool")
    has_garden = st.checkbox("Has Garden", key="has_garden")
    has_parking = st.checkbox("Has Parking", key="has_parking")
    sea_view = st.checkbox("Sea View", key="sea_view")
    elevator = st.checkbox("Elevator", value=False, key="elevator", disabled=property_type != "Appartement")

    images = st.file_uploader(
        "Upload Property Images (Max 5)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )

    if images:
        if len(images) > 5:
            st.error("Please upload up to 5 images only.")
        else:
            thumbs = st.columns(min(len(images), 3))
            for idx, img_file in enumerate(images[:5]):
                img = Image.open(BytesIO(img_file.read()))
                thumbs[idx % len(thumbs)].image(img, use_container_width=True)

    description = st.text_area(
        "Property Description (Arabic, French, or English)",
        placeholder="Describe your property...",
        height=120,
        key="description",
    )

    estimate_clicked = st.button("🔍 Estimate Price", type="primary", use_container_width=True)

    if estimate_clicked:
        errors: List[str] = []
        if not city.strip():
            errors.append("City is required.")
        if len(images or []) > 5:
            errors.append("Maximum 5 images allowed.")
        if not description.strip():
            errors.append("Property description is required.")

        if errors:
            for err in errors:
                st.warning(err)
        else:
            payload_state = {
                "property_type": property_type,
                "governorate": governorate,
                "city": city,
                "neighborhood": neighborhood,
                "size_m2": size_m2,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "condition": condition,
                "has_pool": has_pool,
                "has_garden": has_garden,
                "has_parking": has_parking,
                "sea_view": sea_view,
                "elevator": elevator,
                "description": description,
            }
            payload = build_payload(payload_state, len(images or []))

            prog = st.progress(10, text="Input validated")
            with st.spinner("Running AI valuation analysis..."):
                prog.progress(45, text="Analyzing market + property features")
                try:
                    response = requests.post(f"{API_URL}/estimate", json=payload, timeout=20)
                    response.raise_for_status()
                    st.session_state["results"] = response.json()
                    st.success("✅ Valuation Complete!")
                    prog.progress(100, text="Results ready")
                except requests.RequestException as exc:
                    st.error(f"Could not reach valuation API at {API_URL}. Details: {exc}")

with center_col:
    st.markdown("### Valuation Results")
    results = st.session_state.get("results")
    if not results:
        st.info("Submit property details to generate valuation results.")
    else:
        st.markdown(
            f"""
            <div class="card">
                <div class="metric-main">Estimated Price: {format_tnd(results['estimated_price'])}</div>
                <div class="metric-sub">Price per m2: {format_tnd(results['price_per_m2'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        range_df = pd.DataFrame(
            {
                "Label": ["Lower", "Prediction", "Upper"],
                "Value": [results["lower_bound"], results["estimated_price"], results["upper_bound"]],
            }
        )
        range_fig = px.bar(
            range_df,
            x="Label",
            y="Value",
            color="Value",
            color_continuous_scale=["#86efac", "#22c55e", "#15803d"],
            title="Price Range",
        )
        range_fig.update_layout(coloraxis_showscale=False, height=280, margin=dict(l=12, r=12, t=44, b=8))
        st.plotly_chart(range_fig, use_container_width=True)

        badge_color = confidence_color(results["confidence_level"])
        st.markdown(
            f"""
            <div class="conf-wrap" style="background: {badge_color}">
                <div>{results['confidence_level']} Confidence</div>
                <div style="font-size: 1.9rem;">{results['confidence']}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        impact_df = pd.DataFrame(results["features_impact"])
        impact_df["Label"] = impact_df.apply(
            lambda r: f"{r['feature']}: {r['pct']}% ({'↑' if r['amount'] >= 0 else '↓'} {abs(int(r['amount'])):,} TND)", axis=1
        )
        impact_fig = px.bar(
            impact_df,
            x="amount",
            y="Label",
            orientation="h",
            color="amount",
            color_continuous_scale=[DANGER, ACCENT, SUCCESS],
            title="Key Features Impact",
        )
        impact_fig.update_layout(coloraxis_showscale=False, height=320, margin=dict(l=10, r=10, t=44, b=10), yaxis_title="")
        st.plotly_chart(impact_fig, use_container_width=True)

        st.markdown("#### Comparable Properties")
        comps = results["comparables"]
        prev_col, next_col = st.columns([1, 1])
        with prev_col:
            if st.button("◀ Prev", use_container_width=True):
                st.session_state["carousel_start"] = max(0, st.session_state["carousel_start"] - 1)
        with next_col:
            if st.button("Next ▶", use_container_width=True):
                st.session_state["carousel_start"] = min(max(0, len(comps) - 3), st.session_state["carousel_start"] + 1)

        start = st.session_state["carousel_start"]
        visible = comps[start : start + 3]
        card_cols = st.columns(3)
        for idx, comp in enumerate(visible):
            with card_cols[idx]:
                st.markdown(
                    f"""
                    <div class="card">
                        <div><b>📍 {comp['address']}</b></div>
                        <div>Price: {format_tnd(comp['price'])}</div>
                        <div>Size: {comp['size']} m2</div>
                        <div>Similarity: {comp['similarity']}%</div>
                        <div>Difference: {comp['difference']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

with right_col:
    st.markdown("### Explanations & Insights")
    results = st.session_state.get("results")
    if results:
        with st.expander("🤖 AI Explanation", expanded=True):
            st.write(results["ai_explanation"])

        with st.expander("📷 Image Analysis", expanded=True):
            for item in results["image_analysis"]:
                marker = "✅" if "No " not in item and "Limited" not in item else "❌"
                st.write(f"{marker} {item}")

        with st.expander("📝 Text Analysis", expanded=True):
            ta = results["text_analysis"]
            st.write(f"Description quality: {ta['description_quality']}")
            st.write(f"Sentiment: {ta['sentiment']}")
            st.write(f"Marketing effectiveness: {ta['marketing_effectiveness']}")
            st.write(f"Key phrases: {', '.join(ta['key_phrases'])}")

        with st.expander("📊 Market Context", expanded=True):
            mc = results["market_context"]
            st.write(f"{mc['city']} Average: {mc['avg_m2']:,} TND/m2")
            st.write(f"Your property: {mc['property_m2']:,} TND/m2 ({mc['delta_pct']}%)")
            st.write(f"Market trend: {mc['trend']}")
            st.write(f"Demand: {mc['demand']}")

        st.markdown("#### SHAP Explanation")
        shap_df = pd.DataFrame(results["shap"])
        shap_fig = go.Figure(
            go.Waterfall(
                name="SHAP",
                orientation="v",
                measure=["absolute", "relative", "relative", "relative", "relative", "total"],
                x=shap_df["feature"],
                y=shap_df["value"],
                connector={"line": {"color": "#94a3b8"}},
                increasing={"marker": {"color": SUCCESS}},
                decreasing={"marker": {"color": DANGER}},
                totals={"marker": {"color": PRIMARY}},
                hovertemplate="%{x}: %{y:,} TND<extra></extra>",
            )
        )
        shap_fig.update_layout(
            title="Baseline -> Feature Impacts -> Final Prediction",
            height=330,
            margin=dict(l=10, r=10, t=42, b=8),
        )
        st.plotly_chart(shap_fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.download_button("📄 Download Full Report (PDF)", data=b"EstateMind demo report", file_name="estatemind_report.pdf")
        c2.button("📧 Email Results", use_container_width=True, help="Demo action")
        c3.button("🔗 Share Valuation Link", use_container_width=True, help="Demo action")

st.markdown(
    """
    <div class="footer">
        Powered by EstateMind AI | #AarefBledek - Know Your Country
        <div style="margin-top: 6px;">About | Privacy | Terms | Contact</div>
        <div class="socials" style="margin-top: 6px;">
            <a href="#">LinkedIn</a><a href="#">X</a><a href="#">Facebook</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("Tooltips: SHAP explains feature contribution. MAPE is the average prediction error percentage.")
