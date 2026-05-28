"""
Fasal AI — Streamlit UI
Run from fasal-ai/ directory:  streamlit run src/app.py
"""

import sys
import os
import base64
import json
from pathlib import Path
from datetime import date

# ── Path setup ──────────────────────────────────────────────────────────────
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")

import streamlit as st
from fpdf import FPDF

# ── Agent imports ────────────────────────────────────────────────────────────
from agents.vision_agent import analyze_crop
from agents.weather_agent import get_weather_advice
from agents.agronomy_agent import get_agronomy_advice
from agents.export_compliance_agent import check_export_compliance
from agents.language_agent import format_final_output

# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════
def generate_pdf_report(result: dict) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    pdf.set_font("helvetica", style="B", size=20)
    pdf.cell(0, 15, "Comprehensive Crop Analysis Report", ln=True, align='C')
    pdf.set_font("helvetica", style="I", size=12)
    pdf.cell(0, 10, "Fasal AI - Farm Insights", ln=True, align='C')
    pdf.ln(10)
    
    final = result["final"]
    vision = result["vision"]
    weather = result["weather"]
    export = result["export"]
    
    def add_section(title, text):
        if not text: return
        pdf.set_font("helvetica", style="B", size=16)
        pdf.set_text_color(16, 185, 129)
        pdf.cell(0, 10, title, ln=True)
        pdf.set_font("helvetica", size=12)
        pdf.set_text_color(0, 0, 0)
        safe_text = str(text).encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 8, safe_text)
        pdf.ln(5)

    if final.get("greeting"):
        add_section("Overview", final["greeting"])

    if result.get("had_image") and vision:
        dis = vision.get("disease_detected", "N/A")
        sev = vision.get("severity", "N/A")
        action = vision.get("immediate_action", "N/A")
        txt = f"Disease Detected: {dis}\nSeverity: {sev}\nImmediate Action: {action}\n"
        if final.get("disease_summary"):
            txt += f"\nSummary: {final['disease_summary']}"
        add_section("Disease Diagnosis", txt)
        
    weather_txt = final.get("weather_summary", "")
    if weather_txt:
        weather_txt += f"\nIrrigation Recommendation: {final.get('irrigation_today', '')}"
        add_section("Weather & Irrigation", weather_txt)
        
    top_actions = final.get("top_actions", [])
    if top_actions:
        actions_txt = "\n".join(top_actions)
        add_section("Top Actions", actions_txt)
        
    if export:
        add_section("Export Compliance", final.get("export_status", export.get("overall_status", "Action Required")))
        
    return bytes(pdf.output())

# ── Constants ────────────────────────────────────────────────────────────────
PAKISTANI_PESTICIDES = [
    "Chlorpyrifos", "Imidacloprid", "Lambda-cyhalothrin", "Cypermethrin",
    "Acetamiprid", "Emamectin benzoate", "Abamectin", "Profenofos",
    "Dimethoate", "Thiamethoxam", "Carbendazim", "Mancozeb",
    "Metalaxyl", "Trifloxystrobin", "Tebuconazole", "Glyphosate",
]

STATUS_CONFIG = {
    "READY_TO_EXPORT": ("🟢", "green", "#d4edda"),
    "ACTION_REQUIRED": ("🟡", "#856404", "#fff3cd"),
    "EXPORT_BLOCKED":  ("🔴", "#721c24", "#f8d7da"),
}

CHECK_ICONS = {"PASS": "✅", "WARNING": "⚠️", "FAIL": "❌"}

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fasal AI — Smart Farming Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* ── Brand palette (Dark Mode) ── */
:root {
    --primary: #10b981;
    --primary-light: rgba(16, 185, 129, 0.15);
    --primary-dark: #059669;
    --amber: #f59e0b;
    --red: #ef4444;
    --card-bg: rgba(30, 41, 59, 0.6);
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
}

/* ── Logo area ── */
.fasal-logo {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #34d399 0%, #10b981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
    line-height: 1.1;
    animation: fadeInDown 0.8s ease;
}
.fasal-tagline {
    font-size: 1.05rem;
    color: var(--text-muted);
    margin-top: 4px;
    margin-bottom: 28px;
    font-style: italic;
    font-weight: 400;
}

/* ── Custom Labels (White) ── */
.stTextInput label, .stSelectbox label, .stNumberInput label, .stTextArea label, .stRadio label, .stMultiSelect label, .stDateInput label, .stFileUploader label {
    font-weight: 600 !important;
    color: #ffffff !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.2px;
}

/* ── White card (Glassmorphism & Shadows) ── */
.card {
    background: var(--card-bg);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    color: var(--text-main);
}
.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
}
.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.06);
}

/* ── Section headers ── */
.section-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text-main);
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* ── Action item (Animated hover) ── */
.action-item {
    border-left: 4px solid var(--primary);
    background: var(--primary-light);
    border-radius: 0 12px 12px 0;
    padding: 14px 20px;
    margin-bottom: 10px;
    font-size: 1rem;
    color: var(--text-main);
    font-weight: 500;
    transition: all 0.3s ease;
}
.action-item:hover {
    background: rgba(16, 185, 129, 0.25);
    transform: translateX(4px);
    box-shadow: 0 2px 10px rgba(16, 185, 129, 0.1);
}

/* ── Status badge ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 24px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 1.05rem;
    margin-bottom: 20px;
    animation: fadeIn 0.5s ease;
    box-shadow: 0 2px 10px rgba(0,0,0,0.2);
}

/* ── Metric card ── */
.metric-card {
    background: var(--card-bg);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 18px;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    transition: transform 0.3s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
}
.metric-label {
    font-size: 0.8rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
    font-weight: 600;
}
.metric-value {
    font-size: 1.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #34d399 0%, #10b981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ── Compliance check row ── */
.check-row {
    display: flex;
    gap: 14px;
    align-items: flex-start;
    padding: 14px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}
.check-row:last-child { border-bottom: none; }
.check-icon { font-size: 1.4rem; min-width: 30px; }
.check-body { flex: 1; }
.check-category { font-weight: 700; color: var(--text-main); font-size: 0.95rem; }
.check-details { color: var(--text-muted); font-size: 0.9rem; margin-top: 4px; line-height: 1.4; }
.check-action { color: var(--amber); font-size: 0.85rem; margin-top: 6px; font-weight: 600; }

/* ── Button Animations ── */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    width: 100%;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.25) !important;
}
div.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.35) !important;
}
div.stButton > button[kind="primary"]:active {
    transform: translateY(1px) scale(0.99) !important;
}

div.stButton > button[kind="secondary"] {
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    font-weight: 600 !important;
}
div.stButton > button[kind="secondary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-15px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ── Mobile adjustments ── */
@media (max-width: 768px) {
    .fasal-logo { font-size: 2.2rem; }
    .metric-value { font-size: 1.1rem; }
    .card { padding: 16px; }
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# State Initialization
# ════════════════════════════════════════════════════════════════════════════
if "form_key" not in st.session_state:
    st.session_state.form_key = 0

# ════════════════════════════════════════════════════════════════════════════
# Layout
# ════════════════════════════════════════════════════════════════════════════
# Single column layout (full screen width inputs, results below)


# ════════════════════════════════════════════════════════════════════════════
# TOP SECTION — Input form
# ════════════════════════════════════════════════════════════════════════════
with st.container():
    # Logo
    st.markdown(
        '<div class="fasal-logo">🌾 Fasal AI</div>'
        '<div class="fasal-tagline">آپ کے کھیت کا ذہین ساتھی | Your Smart Farm Assistant</div>',
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown('<div style="padding-top: 10px;"></div>', unsafe_allow_html=True)
        # ── Image upload ──────────────────────────────────────────────────────
        uploaded_file = st.file_uploader(
            "📷 Upload crop image (optional)",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear photo of the affected plant or crop",
            key=f"input_image_{st.session_state.form_key}",
        )

        # ── Basic farm info ───────────────────────────────────────────────────
        col1, col2 = st.columns(2)
        with col1:
            crop_type = st.text_input(
                "🌱 Crop type *",
                placeholder="e.g. wheat, mango...",
                key=f"input_crop_type_{st.session_state.form_key}",
            )
        with col2:
            location = st.text_input(
                "📍 Your location *",
                placeholder="e.g. Lahore, Multan...",
                key=f"input_location_{st.session_state.form_key}",
            )

        col3, col4 = st.columns(2)
        with col3:
            crop_stage = st.selectbox(
                "📈 Crop stage *",
                ["", "Germination", "Tillering", "Flowering", "Grain-filling", "Harvest-ready"],
                index=0,
                key=f"input_crop_stage_{st.session_state.form_key}",
            )
        with col4:
            soil_type = st.selectbox(
                "🌍 Soil type",
                ["Loamy", "Clay", "Sandy", "Silty"],
                key=f"input_soil_type_{st.session_state.form_key}",
            )

        col5, col6 = st.columns(2)
        with col5:
            area_acres = st.number_input(
                "📐 Farm size (acres)",
                min_value=1, max_value=5000, value=5, step=1,
                key=f"input_area_acres_{st.session_state.form_key}",
            )
        with col6:
            last_irrigation = st.date_input(
                "💧 Last irrigation date",
                value=date.today(),
                max_value=date.today(),
                key=f"input_last_irrigation_{st.session_state.form_key}",
            )

        farmer_description = st.text_area(
            "📝 Describe the problem",
            placeholder="Describe what you see on your crop... (e.g. yellow spots)",
            height=100,
            key=f"input_farmer_description_{st.session_state.form_key}",
        )

        output_language_raw = st.radio(
            "🗣️ Output language",
            ["English", "اردو"],
            horizontal=True,
            key=f"input_output_language_{st.session_state.form_key}",
        )
        output_language = "english" if output_language_raw == "English" else "urdu"

        # ── Export compliance expander ────────────────────────────────────────
        with st.expander("📦 Export compliance check (optional)"):
            target_country = st.selectbox(
                "Target country",
                ["", "UAE", "Saudi Arabia", "EU", "UK", "China", "USA"],
                index=0,
                key=f"input_target_country_{st.session_state.form_key}",
            ) or None
            pesticides_used = st.multiselect(
                "Pesticides used",
                PAKISTANI_PESTICIDES,
                default=[],
                key=f"input_pesticides_used_{st.session_state.form_key}",
            )
            export_harvest_date = st.date_input(
                "Harvest date",
                value=date.today(),
                key=f"export_harvest_date_{st.session_state.form_key}",
            )
            quantity_kg = st.number_input(
                "Quantity (kg)",
                min_value=0, value=1000, step=100,
                key=f"export_qty_{st.session_state.form_key}",
            )

        st.markdown('<div style="padding-top: 15px;"></div>', unsafe_allow_html=True)
        # ── Submit button ─────────────────────────────────────────────────────
        analyse_clicked = st.button("✨ Analyze My Crop", type="primary", use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PIPELINE — runs when button is clicked
# ════════════════════════════════════════════════════════════════════════════
if analyse_clicked:
    # Validation
    errors = []
    if not crop_type.strip():
        errors.append("Crop type is required.")
    if not location.strip():
        errors.append("Location is required.")
    if not crop_stage:
        errors.append("Crop stage is required.")

    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    loader_placeholder = st.empty()

    def show_loader(msg):
        html = f"""
        <style>
        .fullscreen-overlay {{
            position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
            background: rgba(15, 23, 42, 0.85);
            z-index: 999999;
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            backdrop-filter: blur(8px);
        }}
        .fs-spinner {{
            width: 70px; height: 70px;
            border: 6px solid rgba(16, 185, 129, 0.2);
            border-top-color: #10b981;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}
        .fs-text {{ margin-top: 25px; color: #f8fafc; font-size: 1.6rem; font-weight: 600; font-family: 'Outfit', sans-serif; }}
        @keyframes spin {{ 100% {{ transform: rotate(360deg); }} }}
        </style>
        <div class="fullscreen-overlay">
            <div class="fs-spinner"></div>
            <div class="fs-text">{msg}</div>
        </div>
        """
        loader_placeholder.markdown(html, unsafe_allow_html=True)

    # Convert image to base64
    image_b64 = None
    if uploaded_file:
        image_b64 = base64.b64encode(uploaded_file.read()).decode("utf-8")

    vision_result = None
    weather_result = None
    agronomy_result = None
    export_result = None
    final_output = None
    pipeline_error = None

    try:
        # Step 1 — Vision
        if image_b64:
            show_loader("🔬 Analyzing crop image...")
            vision_result = analyze_crop({
                "image_base64": image_b64,
                "crop_type": crop_type,
                "location": location,
                "farmer_description": farmer_description or "No description provided",
            })

        # Step 2 — Weather
        show_loader(f"🌤️ Checking weather for {location}...")
        weather_result = get_weather_advice({
            "location": location,
            "crop_type": crop_type,
            "crop_stage": crop_stage.lower(),
            "last_irrigation_date": str(last_irrigation),
        })

        # Step 3 — Agronomy
        show_loader("🌱 Getting expert farming advice...")
        agronomy_result = get_agronomy_advice({
            "crop_type": crop_type,
            "crop_stage": crop_stage.lower(),
            "disease_result": vision_result,
            "weather_result": weather_result,
            "farmer_question": farmer_description or "General advice needed",
            "soil_type": soil_type.lower(),
            "area_acres": int(area_acres),
        })

        # Step 4 — Export compliance
        if target_country:
            show_loader("📋 Checking export compliance...")
            export_result = check_export_compliance({
                "crop_type": crop_type,
                "variety": "Standard",
                "target_country": target_country,
                "pesticides_used": pesticides_used,
                "fertilizers_used": [],
                "harvest_date": str(export_harvest_date) or str(last_irrigation),
                "quantity_kg": int(quantity_kg),
            })

        # Step 5 — Language & format
        show_loader("📝 Preparing your personalized report...")
        final_output = format_final_output({
            "vision_result": vision_result,
            "weather_result": weather_result,
            "agronomy_result": agronomy_result,
            "export_result": export_result,
            "output_language": output_language,
            "farmer_phone": None, # Removed
        })

        # Store in session
        st.session_state["fasal_result"] = {
            "vision": vision_result,
            "weather": weather_result,
            "agronomy": agronomy_result,
            "export": export_result,
            "final": final_output,
            "had_image": image_b64 is not None,
            "had_export": target_country is not None,
        }

    except Exception as exc:
        pipeline_error = str(exc)
    finally:
        loader_placeholder.empty()

    if pipeline_error:
        st.error(f"❌ Pipeline error: {pipeline_error}")


# ════════════════════════════════════════════════════════════════════════════
# BOTTOM SECTION — Results
# ════════════════════════════════════════════════════════════════════════════
with st.container():
    result = st.session_state.get("fasal_result")

    if not result:
        # Placeholder state
        st.markdown("""
        <div style="padding-top: 40px; display: flex; align-items: center; justify-content: center;">
            <div class="card" style="text-align:center; padding: 60px 40px; max-width: 600px; margin: 0 auto; animation: fadeIn 0.8s ease;">
                <div style="font-size:4rem; margin-bottom:20px; filter: drop-shadow(0 4px 6px rgba(16, 185, 129, 0.2));">🌾</div>
                <div style="font-size:1.4rem; font-weight:700; color:var(--text-main); margin-bottom: 10px;">
                    Ready for Analysis
                </div>
                <div style="font-size:1rem; color:var(--text-muted); line-height: 1.5;">
                    Fill in the details above and click <strong>Analyze My Crop</strong> to receive personalized, AI-driven farming insights right here.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown('<div id="report-marker"></div>', unsafe_allow_html=True)
        st.markdown("""
        <style>
        div[data-testid="stVerticalBlock"]:has(#report-marker) {
            background: linear-gradient(145deg, #1e293b, #0f172a);
            padding: 40px 30px;
            border-radius: 24px;
            border: 1px solid rgba(16, 185, 129, 0.2);
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
            margin-top: 40px;
            animation: fadeIn 0.8s ease;
        }
        @media (max-width: 768px) {
            div[data-testid="stVerticalBlock"]:has(#report-marker) {
                padding: 20px 15px;
            }
        }
        </style>
        """, unsafe_allow_html=True)

        final = result["final"]
        vision = result["vision"]
        weather = result["weather"]
        agronomy = result["agronomy"]
        export = result["export"]

        # ── Report Title ──────────────────────────────────────────────────
        st.markdown(
            '<div style="text-align: center; margin-bottom: 30px;">'
            '<h1 style="font-size: 2.2rem; font-weight: 800; margin: 0; color: #f8fafc;">Comprehensive Crop Analysis Report</h1>'
            '<p style="color: var(--primary); font-size: 1.1rem; margin-top: 5px;">AI-Generated Insights for your Farm</p>'
            '</div>', 
            unsafe_allow_html=True
        )

        # ── Greeting ──────────────────────────────────────────────────────
        greeting = final.get("greeting", "")
        if greeting:
            st.success(greeting)

        # ── Section 1: Disease Diagnosis ──────────────────────────────────
        if result["had_image"] and vision:
            st.markdown('<div class="section-title">🔬 Disease Diagnosis</div>',
                        unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            def metric_html(label, value):
                return (f'<div class="metric-card"><div class="metric-label">{label}</div>'
                        f'<div class="metric-value">{value}</div></div>')

            m1.markdown(metric_html("Disease", vision.get("disease_detected", "—")),
                        unsafe_allow_html=True)
            m2.markdown(metric_html("Severity", vision.get("severity", "—")),
                        unsafe_allow_html=True)
            m3.markdown(metric_html("Confidence", vision.get("confidence", "—")),
                        unsafe_allow_html=True)
            m4.markdown(metric_html("Area Affected",
                                    f"{vision.get('affected_area_percent', 0)}%"),
                        unsafe_allow_html=True)

            st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

            if final.get("disease_summary"):
                st.info(f"📋 {final['disease_summary']}")

            if vision.get("immediate_action"):
                st.warning(f"⚡ **Immediate Action:** {vision['immediate_action']}")

            if vision.get("plant_id_confirmation"):
                pid = vision["plant_id_confirmation"]
                name = pid.get("name", "unknown")
                prob = pid.get("probability", 0)
                if name not in ("unavailable", "unknown"):
                    st.caption(
                        f"🌿 Plant.id confirmation: **{name}** "
                        f"({prob:.0%} probability)"
                    )

            st.markdown('<div style="margin: 24px 0;"></div>', unsafe_allow_html=True)

        # ── Section 2: Weather & Irrigation ──────────────────────────────
        st.markdown('<div class="section-title">🌤️ Weather & Irrigation</div>',
                    unsafe_allow_html=True)

        conds = final.get("weather_summary") or weather.get("current_conditions", "")
        if conds:
            st.info(f"☀️ {conds}")

        for alert in weather.get("proactive_alerts", []):
            st.warning(f"⚠️ {alert}")

        irr_today = final.get("irrigation_today", "")
        if irr_today:
            st.markdown(
                f'<div class="action-item">💧 {irr_today}</div>',
                unsafe_allow_html=True,
            )

        irr_schedule = agronomy.get("irrigation_schedule", [])
        if irr_schedule:
            with st.expander("📅 7-day irrigation schedule"):
                rows = [
                    {
                        "Day": d.get("day", ""),
                        "Action": d.get("action", "").capitalize(),
                        "Duration (hrs)": d.get("duration_hours", 0),
                        "Reason": d.get("reason", ""),
                    }
                    for d in irr_schedule
                ]
                st.table(rows)

        st.markdown('<div style="margin: 24px 0;"></div>', unsafe_allow_html=True)

        # ── Section 3: Action Plan ────────────────────────────────────────
        st.markdown('<div class="section-title">✅ Your Action Plan This Week</div>',
                    unsafe_allow_html=True)

        top_actions = final.get("top_actions", [])
        if top_actions:
            for action in top_actions:
                st.markdown(
                    f'<div class="action-item">{action}</div>',
                    unsafe_allow_html=True,
                )
        else:
            weekly = agronomy.get("weekly_action_plan", [])
            for i, action in enumerate(weekly[:5], 1):
                st.markdown(
                    f'<div class="action-item">{i}. {action}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown('<div style="margin: 24px 0;"></div>', unsafe_allow_html=True)

        # ── Section 4: Fertilizer & Pest Control ─────────────────────────
        st.markdown('<div class="section-title">🧪 Fertilizer & Pest Control</div>',
                    unsafe_allow_html=True)

        fert_col, pest_col = st.columns(2, gap="medium")

        fert = agronomy.get("fertilizer_recommendation", {})
        with fert_col:
            st.markdown("""
            <div class="card">
                <div class="section-title" style="font-size:1rem;">🌿 Fertilizer</div>
            """, unsafe_allow_html=True)
            if fert:
                st.markdown(f"**Type:** {fert.get('type', '—')}")
                st.markdown(f"**Quantity/acre:** {fert.get('quantity_per_acre', '—')}")
                st.markdown(f"**Timing:** {fert.get('timing', '—')}")
                if fert.get("reason"):
                    st.caption(fert["reason"])
            st.markdown("</div>", unsafe_allow_html=True)

        pest = agronomy.get("pest_control", {})
        with pest_col:
            st.markdown("""
            <div class="card">
                <div class="section-title" style="font-size:1rem;">🐛 Pest Control</div>
            """, unsafe_allow_html=True)
            if pest:
                identified = pest.get("pest_identified", "none")
                if identified and identified.lower() != "none":
                    st.markdown(f"**Pest:** {identified}")
                st.markdown(f"**Treatment:** {pest.get('treatment', '—')}")
                st.markdown(f"**Dosage:** {pest.get('dosage', '—')}")
                if pest.get("safety_note"):
                    st.warning(f"⚠️ {pest['safety_note']}", icon=None)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div style="margin: 24px 0;"></div>', unsafe_allow_html=True)

        # ── Section 5: Export Compliance ──────────────────────────────────
        if result["had_export"] and export:
            st.markdown('<div class="section-title">📦 Export Compliance Report</div>',
                        unsafe_allow_html=True)

            overall = export.get("overall_status", "ACTION_REQUIRED")
            icon, color, bg = STATUS_CONFIG.get(overall, ("ℹ️", "#333", "#eee"))
            st.markdown(
                f'<div class="status-badge" style="background:{bg}; color:{color};">'
                f'{icon} {overall.replace("_", " ")}</div>',
                unsafe_allow_html=True,
            )

            checks = export.get("compliance_checks", [])
            if checks:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                for chk in checks:
                    status_icon = CHECK_ICONS.get(chk.get("status", ""), "ℹ️")
                    action_html = ""
                    if chk.get("action_required"):
                        action_html = (f'<div class="check-action">'
                                       f'→ {chk["action_required"]}</div>')
                    st.markdown(f"""
                    <div class="check-row">
                        <div class="check-icon">{status_icon}</div>
                        <div class="check-body">
                            <div class="check-category">{chk.get('category', '')}</div>
                            <div class="check-details">{chk.get('details', '')}</div>
                            {action_html}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Phytosanitary certificate
            phyto = export.get("phytosanitary_certificate", {})
            if phyto and phyto.get("required"):
                st.info(
                    f"📄 **Phytosanitary Certificate Required** — "
                    f"Issued by: {phyto.get('issuing_authority', '—')} | "
                    f"Contact: {phyto.get('contact', '—')} | "
                    f"Apply at least **{phyto.get('days_before_shipment', 0)} days** before shipment"
                )

            # Packaging requirements table
            pkg = export.get("packaging_requirements", {})
            if pkg:
                with st.expander("📦 Packaging Requirements"):
                    pkg_rows = {
                        "Label Languages": ", ".join(pkg.get("label_language", [])),
                        "Country of Origin Required": "Yes" if pkg.get("country_of_origin_required") else "No",
                        "Weight Marking": pkg.get("weight_marking", "—"),
                        "Cold Chain Temp (°C)": str(pkg.get("cold_chain_temp_celsius") or "Not required"),
                        "Value-Add Certifications": ", ".join(pkg.get("certifications_that_add_value", [])) or "—",
                    }
                    for k, v in pkg_rows.items():
                        st.markdown(f"**{k}:** {v}")

            # Banned substances
            banned = export.get("banned_substances", [])
            if banned:
                st.error(f"🚫 **Banned in {result.get('target_country', 'target country')}:** "
                         f"{', '.join(banned)}")

            if export.get("summary"):
                st.caption(export["summary"])

        # ── Download PDF button ───────────────────────────────────────────
        pdf_bytes = generate_pdf_report(result)
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name="Fasal_AI_Crop_Report.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="secondary"
            )

        # ── Reset button ──────────────────────────────────────────────────
        with col2:
            if st.button("🔄 Analyze Another Crop", use_container_width=True):
                st.session_state.form_key += 1
                st.session_state.pop("fasal_result", None)
                st.rerun()
