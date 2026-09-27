"""
app.py — ESG Authenticity Scorer | Streamlit Frontend
Connects directly to inference.py
Run: streamlit run app.py
"""

import sys
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from pypdf import PdfReader
import io

# ── Make sure inference.py is importable ──
sys.path.append(str(Path(__file__).resolve().parent))

from inference import (
    load_model_and_tokenizer,
    score_text,
    interpret_score,
    report_to_dict,
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ESG Authenticity Scorer",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS  — Professional / Classy Redesign
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    /* ── Global Reset ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: #1A1A2E;
    }

    /* ── App background ── */
    .stApp {
        background: #F7F5F0;
    }

    /* Sidebar background */
    section[data-testid="stSidebar"] {
        background: #1A1A2E !important;
    }
    section[data-testid="stSidebar"] * {
        color: #E8E4DC !important;
    }
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stCheckbox label {
        color: #B8B4AC !important;
        font-size: 13px;
    }
    section[data-testid="stSidebar"] h3 {
        color: #C9A96E !important;
        font-family: 'Cormorant Garamond', serif;
        font-weight: 600;
        font-size: 18px;
        letter-spacing: 0.5px;
        margin-top: 20px;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #2E2E4A !important;
    }

    /* ── HEADER ── */
    .esg-header {
        background: #1A1A2E;
        border-bottom: 1px solid #C9A96E;
        border-radius: 0;
        padding: 36px 48px 32px;
        margin: -1rem -1rem 2rem -1rem;
        position: relative;
        overflow: hidden;
    }
    .esg-header::before {
        content: '';
        position: absolute;
        top: 0; right: 0; bottom: 0;
        width: 320px;
        background: linear-gradient(135deg, transparent 40%, rgba(201,169,110,0.07) 100%);
        pointer-events: none;
    }
    .esg-header-eyebrow {
        font-family: 'DM Sans', sans-serif;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 3.5px;
        text-transform: uppercase;
        color: #C9A96E;
        margin-bottom: 10px;
    }
    .esg-header-title {
        font-family: 'Cormorant Garamond', serif;
        font-size: 42px;
        font-weight: 300;
        color: #F7F5F0;
        letter-spacing: -0.5px;
        line-height: 1.1;
        margin: 0 0 8px 0;
    }
    .esg-header-title span {
        color: #C9A96E;
        font-weight: 600;
    }
    .esg-header-sub {
        font-size: 13px;
        color: #8A8A9A;
        font-weight: 300;
        letter-spacing: 0.3px;
        margin: 0;
    }
    .esg-header-badge {
        display: inline-block;
        background: rgba(201,169,110,0.12);
        border: 1px solid rgba(201,169,110,0.3);
        color: #C9A96E;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        padding: 4px 10px;
        border-radius: 2px;
        margin-bottom: 16px;
    }

    /* ── Section headings ── */
    .section-heading {
        font-family: 'Cormorant Garamond', serif;
        font-size: 26px;
        font-weight: 600;
        color: #1A1A2E;
        letter-spacing: -0.3px;
        border-bottom: 1px solid #DDD9D0;
        padding-bottom: 10px;
        margin: 32px 0 18px 0;
    }
    .section-subheading {
        font-family: 'Cormorant Garamond', serif;
        font-size: 22px;
        font-weight: 600;
        color: #1A1A2E;
        margin: 24px 0 12px 0;
    }

    /* ── MAIN SCORE CARD ── */
    .score-card-outer {
        background: #1A1A2E;
        border-radius: 4px;
        padding: 36px 28px 32px;
        text-align: center;
        position: relative;
        overflow: hidden;
        border: 1px solid #2E2E4A;
    }
    .score-card-outer::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #C9A96E, transparent);
    }
    .score-card-label {
        font-size: 9px;
        font-weight: 600;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #C9A96E;
        margin-bottom: 14px;
    }
    .score-card-number {
        font-family: 'Cormorant Garamond', serif;
        font-size: 88px;
        font-weight: 300;
        color: #F7F5F0;
        line-height: 1;
        letter-spacing: -4px;
    }
    .score-card-denom {
        font-family: 'Cormorant Garamond', serif;
        font-size: 24px;
        color: #5A5A6E;
        font-weight: 300;
        margin-top: -4px;
    }
    .score-card-band {
        margin-top: 16px;
        font-size: 12px;
        font-weight: 500;
        letter-spacing: 1.5px;
        color: #C9A96E;
        text-transform: uppercase;
    }
    .score-card-company {
        margin-top: 6px;
        font-size: 12px;
        color: #5A5A6E;
        font-style: italic;
    }

    /* ── PILLAR CARDS ── */
    .pillar-card-outer {
        border-radius: 4px;
        padding: 22px 20px;
        text-align: center;
        background: #FFFFFF;
        border: 1px solid #E8E4DC;
        border-top: 2px solid #C9A96E;
        height: 100%;
    }
    .pillar-card-score {
        font-family: 'Cormorant Garamond', serif;
        font-size: 52px;
        font-weight: 600;
        color: #1A1A2E;
        line-height: 1;
    }
    .pillar-card-denom {
        font-size: 14px;
        color: #9A9A9A;
        font-weight: 300;
    }
    .pillar-card-name {
        font-size: 9px;
        font-weight: 600;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        color: #C9A96E;
        margin: 10px 0 4px;
    }
    .pillar-card-meta {
        font-size: 11px;
        color: #8A8A9A;
        margin-top: 6px;
        line-height: 1.6;
    }

    /* ── METRICS ── */
    div[data-testid="metric-container"] {
        background: #FFFFFF;
        border: 1px solid #E8E4DC;
        border-radius: 4px;
        padding: 16px 20px !important;
    }
    div[data-testid="metric-container"] label {
        font-size: 10px !important;
        font-weight: 600 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        color: #8A8A9A !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-family: 'Cormorant Garamond', serif !important;
        font-size: 30px !important;
        font-weight: 600 !important;
        color: #1A1A2E !important;
    }

    /* ── FLAGGED SENTENCE CARDS ── */
    .flagged-card {
        background: #FFFFFF;
        border: 1px solid #E8E4DC;
        border-left: 3px solid #C9562A;
        border-radius: 4px;
        padding: 18px 20px;
        margin: 8px 0;
    }
    .flagged-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .flagged-card-index {
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #8A8A9A;
    }
    .flagged-card-badge {
        background: #FDF0EB;
        border: 1px solid #E8B4A0;
        color: #C9562A;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1px;
        padding: 3px 10px;
        border-radius: 2px;
    }
    .flagged-card-pillar {
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #C9A96E;
        margin-bottom: 8px;
    }
    .flagged-card-text {
        font-size: 14px;
        color: #2A2A3A;
        line-height: 1.65;
        font-style: italic;
    }
    .flagged-card-footer {
        margin-top: 12px;
        padding-top: 10px;
        border-top: 1px solid #F0ECE4;
        display: flex;
        gap: 20px;
        font-size: 11px;
        color: #9A9A9A;
        letter-spacing: 0.5px;
    }
    .flagged-card-footer span b {
        color: #4A4A5A;
        font-weight: 600;
    }

    /* ── SPECIFIC SENTENCE CARDS ── */
    .specific-card {
        background: #FFFFFF;
        border: 1px solid #E8E4DC;
        border-left: 3px solid #2A7A4F;
        border-radius: 4px;
        padding: 16px 20px;
        margin: 6px 0;
    }
    .specific-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .specific-card-pillar {
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #2A7A4F;
    }
    .specific-card-badge {
        background: #EBF5EF;
        border: 1px solid #A0D4B4;
        color: #2A7A4F;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1px;
        padding: 3px 10px;
        border-radius: 2px;
    }
    .specific-card-text {
        font-size: 13px;
        color: #2A2A3A;
        line-height: 1.6;
        font-style: italic;
    }

    /* ── SCORE BAND PILLS (sidebar) ── */
    .band-pill {
        border-radius: 3px;
        padding: 7px 14px;
        margin: 4px 0;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.5px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* ── EMPTY STATE CARDS ── */
    .empty-card {
        background: #FFFFFF;
        border: 1px solid #E8E4DC;
        border-top: 2px solid #C9A96E;
        border-radius: 4px;
        padding: 22px 16px;
        text-align: center;
    }
    .empty-card-icon {
        font-size: 26px;
        margin-bottom: 8px;
    }
    .empty-card-title {
        font-family: 'Cormorant Garamond', serif;
        font-size: 16px;
        font-weight: 600;
        color: #1A1A2E;
        margin-bottom: 4px;
    }
    .empty-card-desc {
        font-size: 11px;
        color: #8A8A9A;
        line-height: 1.5;
    }

    /* ── UPLOAD AREA ── */
    div[data-testid="stFileUploader"] {
        background: #FFFFFF;
        border: 1px dashed #C9A96E !important;
        border-radius: 4px;
        padding: 12px;
    }

    /* ── TABS ── */
    div[data-testid="stTabs"] button {
        font-size: 11px !important;
        font-weight: 600 !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        color: #8A8A9A !important;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #1A1A2E !important;
        border-bottom-color: #C9A96E !important;
    }

    /* ── DATAFRAME ── */
    div[data-testid="stDataFrame"] {
        border: 1px solid #E8E4DC !important;
        border-radius: 4px !important;
    }

    /* ── DOWNLOAD BUTTONS ── */
    div.stDownloadButton > button {
        background: #1A1A2E !important;
        color: #F7F5F0 !important;
        border: 1px solid #2E2E4A !important;
        border-radius: 3px !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        padding: 10px 20px !important;
        transition: all 0.2s ease;
    }
    div.stDownloadButton > button:hover {
        background: #C9A96E !important;
        border-color: #C9A96E !important;
        color: #1A1A2E !important;
    }

    /* ── SUCCESS/INFO MESSAGES ── */
    div[data-testid="stAlert"] {
        border-radius: 3px;
        font-size: 13px;
    }

    /* ── DIVIDERS ── */
    hr {
        border: none !important;
        border-top: 1px solid #E8E4DC !important;
        margin: 28px 0 !important;
    }

    /* ── SPINNER ── */
    div[data-testid="stSpinner"] p {
        font-size: 13px;
        color: #5A5A6E;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL — cached so it loads only once
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ClimateBERT model...")
def get_model():
    return load_model_and_tokenizer()


# ─────────────────────────────────────────────
# PDF EXTRACTION
# ─────────────────────────────────────────────
def extract_pdf_text(uploaded_file) -> str:
    reader = PdfReader(io.BytesIO(uploaded_file.read()))
    text   = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text.strip()


# ─────────────────────────────────────────────
# RISK BAND HELPER
# ─────────────────────────────────────────────
def get_band(score: float):
    if score >= 80: return "Highly Specific",  "#1E5A3E"
    if score >= 60: return "Mostly Specific",  "#2A7A4F"
    if score >= 40: return "Mixed Signals",    "#B07D2A"
    if score >= 20: return "Mostly Vague",     "#C9562A"
    return            "Highly Vague",          "#8B1A1A"


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="esg-header">
    <div class="esg-header-eyebrow">Sustainability Intelligence Platform</div>
    <h1 class="esg-header-title">ESG Report <span>Authenticity</span> Scorer</h1>
    <p class="esg-header-sub">
        Greenwashing detection in corporate sustainability disclosures &mdash; powered by ClimateBERT NLP
    </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Analysis Parameters")
    threshold = st.slider(
        "P(VAGUE) Flag Threshold",
        min_value=0.50, max_value=0.95,
        value=0.65, step=0.05,
        help="Sentences above this probability are flagged as vague/greenwashed"
    )
    max_sents = st.slider(
        "Max Sentences to Analyse",
        min_value=100, max_value=3000,
        value=1000, step=100,
        help="Increase for full reports, decrease for faster preview"
    )
    show_specific = st.checkbox("Show well-evidenced sentences", value=False)

    st.markdown("---")
    st.markdown("### Score Bands")

    for rng, label, color, text_col in [
        ("80–100", "Highly Specific",  "#1E5A3E", "#A8D5B8"),
        ("60–79",  "Mostly Specific",  "#2A7A4F", "#A8D5B8"),
        ("40–59",  "Mixed Signals",    "#B07D2A", "#F5DFA8"),
        ("20–39",  "Mostly Vague",     "#C9562A", "#F5C4B0"),
        ("0–19",   "Highly Vague",     "#8B1A1A", "#F5A8A8"),
    ]:
        st.markdown(
            f'<div class="band-pill" style="background:{color}20; border:1px solid {color}40;">'
            f'<span style="color:{color};font-size:11px;font-weight:700;letter-spacing:1px;">{rng}</span>'
            f'<span style="color:{color};font-size:11px;">{label}</span></div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("""
    **ClimateBERT** fine-tuned on ESG text  
    **pypdf** for PDF extraction  
    **Plotly** visualisations

    ---
    """)


# ─────────────────────────────────────────────
# UPLOAD SECTION
# ─────────────────────────────────────────────
st.markdown('<div class="section-heading">Upload Report</div>', unsafe_allow_html=True)

col_up, col_name = st.columns([2, 1])
with col_up:
    uploaded = st.file_uploader(
        "Upload a corporate ESG or sustainability report (PDF)",
        type=["pdf"]
    )
with col_name:
    company = st.text_input(
        "Company Name (optional)",
        placeholder="e.g. Amazon, Unilever..."
    )


# ─────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────
if uploaded:
    display = company.strip() if company.strip() else uploaded.name.replace(".pdf", "")

    with st.spinner("Extracting text from PDF..."):
        raw_text = extract_pdf_text(uploaded)

    if not raw_text:
        st.error("Could not extract text. Ensure the PDF is not scanned / image-only.")
        st.stop()

    st.success(f"Text extracted from **{display}** — ready for analysis.")

    tokenizer, model, device = get_model()

    with st.spinner("Running ClimateBERT greenwashing analysis..."):
        report = score_text(raw_text, tokenizer, model, device)

    # ─────────────────────────────────────────
    # RESULTS
    # ─────────────────────────────────────────
    st.markdown('<div class="section-heading">Results</div>', unsafe_allow_html=True)

    final        = report.final_score
    band, bcolor = get_band(final)

    c1, c2, c3, c4 = st.columns([1.8, 1, 1, 1])

    with c1:
        st.markdown(f"""
        <div class="score-card-outer">
            <div class="score-card-label">Authenticity Score</div>
            <div class="score-card-number">{final:.1f}</div>
            <div class="score-card-denom">/ 100</div>
            <div class="score-card-band">{band}</div>
            <div class="score-card-company">{display}</div>
        </div>
        """, unsafe_allow_html=True)

    pillar_info = [
        (report.e_score, "Environmental", "E"),
        (report.s_score, "Social",        "S"),
        (report.g_score, "Governance",    "G"),
    ]

    for col, (ps, name, letter) in zip([c2, c3, c4], pillar_info):
        with col:
            st.markdown(f"""
            <div class="pillar-card-outer">
                <div class="pillar-card-score">{ps.score:.1f}</div>
                <div class="pillar-card-denom">/ 100</div>
                <div class="pillar-card-name">{letter} &mdash; {name}</div>
                <div class="pillar-card-meta">
                    {ps.n_sentences} sentences<br>
                    {ps.specific_count} specific &nbsp;&middot;&nbsp; {ps.vague_count} vague
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Quick stats ──
    st.markdown("")
    m1, m2, m3, m4 = st.columns(4)
    total   = len(report.sentence_results)
    n_vague = sum(1 for r in report.sentence_results if r.label == "VAGUE")
    n_spec  = total - n_vague
    avg_pv  = sum(r.p_vague for r in report.sentence_results) / max(total, 1)

    m1.metric("Total Sentences",  total)
    m2.metric("Flagged as Vague", f"{n_vague}  ({n_vague/max(total,1)*100:.1f}%)")
    m3.metric("Specific Claims",  f"{n_spec}  ({n_spec/max(total,1)*100:.1f}%)")
    m4.metric("Avg P(VAGUE)",     f"{avg_pv:.3f}")


    # ─────────────────────────────────────────
    # CHARTS
    # ─────────────────────────────────────────
    st.markdown('<div class="section-heading">Analysis Charts</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Score Gauge", "P(VAGUE) Distribution", "Pillar Breakdown"])

    PLOTLY_LAYOUT = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FAFAF8",
        font=dict(family="DM Sans, sans-serif", color="#1A1A2E"),
        margin=dict(t=48, b=24, l=24, r=24),
        height=360,
    )

    with tab1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=final,
            number={"font": {"size": 64, "family": "Cormorant Garamond, serif", "color": "#1A1A2E"},
                    "suffix": "/100"},
            title={"text": f"<b>{display}</b><br><span style='font-size:12px;color:#8A8A9A'>"
                            f"Authenticity Score</span>",
                   "font": {"size": 15, "family": "DM Sans, sans-serif", "color": "#1A1A2E"}},
            gauge={
                "axis": {"range": [0, 100], "tickfont": {"size": 11},
                         "tickcolor": "#AAAAAA", "tickwidth": 1},
                "bar":  {"color": bcolor, "thickness": 0.22},
                "bgcolor": "#F0ECE4",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  20],  "color": "#F5E8E8"},
                    {"range": [20, 40],  "color": "#FAF0E0"},
                    {"range": [40, 60],  "color": "#FAFAE0"},
                    {"range": [60, 80],  "color": "#E8F5EE"},
                    {"range": [80, 100], "color": "#D4EEE0"},
                ],
                "threshold": {"line": {"color": bcolor, "width": 3},
                              "thickness": 0.8, "value": final}
            }
        ))
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        pvague_vals = [r.p_vague for r in report.sentence_results]
        fig2 = px.histogram(
            pvague_vals, nbins=40,
            title="P(VAGUE) Distribution Across All Sentences",
            labels={"value": "P(VAGUE)", "count": "Sentence Count"},
            color_discrete_sequence=["#1A1A2E"]
        )
        fig2.add_vline(
            x=threshold, line_dash="dot", line_color="#C9A96E",
            line_width=2,
            annotation_text=f"  Threshold {threshold}",
            annotation_font=dict(color="#C9A96E", size=11)
        )
        fig2.update_layout(**PLOTLY_LAYOUT, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        pillar_scores = {
            "Environmental": report.e_score.score,
            "Social":        report.s_score.score,
            "Governance":    report.g_score.score,
        }
        colors = ["#C9562A" if v < 40 else "#B07D2A" if v < 60 else "#2A7A4F"
                  for v in pillar_scores.values()]
        fig3 = go.Figure(go.Bar(
            x=list(pillar_scores.keys()),
            y=list(pillar_scores.values()),
            marker_color=colors,
            text=[f"{v:.1f}" for v in pillar_scores.values()],
            textposition="outside",
            textfont=dict(family="Cormorant Garamond, serif", size=18, color="#1A1A2E"),
            width=0.45,
        ))
        fig3.add_hline(y=50, line_dash="dot", line_color="#AAAAAA", line_width=1,
                       annotation_text="  50 — Midpoint",
                       annotation_font=dict(size=10, color="#8A8A9A"))
        fig3.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Score by ESG Pillar (0–100)", font=dict(size=14)),
            yaxis=dict(range=[0, 110], showgrid=True,
                       gridcolor="#E8E4DC", gridwidth=1),
            xaxis=dict(showgrid=False),
            showlegend=False,
        )
        st.plotly_chart(fig3, use_container_width=True)


    # ─────────────────────────────────────────
    # FLAGGED SENTENCES
    # ─────────────────────────────────────────
    st.markdown('<div class="section-heading">Greenwashing Signals Detected</div>',
                unsafe_allow_html=True)
    st.caption(f"Top flagged sentences ranked by P(VAGUE). Threshold: {threshold}")

    flagged = sorted(
        [r for r in report.sentence_results if r.p_vague >= threshold],
        key=lambda r: r.p_vague, reverse=True
    )[:10]

    pillar_labels = {"E": "Environmental", "S": "Social", "G": "Governance"}
    pillar_icons  = {"E": "🌿", "S": "👥", "G": "🏛️"}

    if flagged:
        for i, r in enumerate(flagged, 1):
            icon  = pillar_icons.get(r.pillar, "📄")
            pname = pillar_labels.get(r.pillar, r.pillar)
            snippet = r.text[:300] + ("…" if len(r.text) > 300 else "")
            st.markdown(f"""
            <div class="flagged-card">
                <div class="flagged-card-header">
                    <span class="flagged-card-index">Signal #{i:02d}</span>
                    <span class="flagged-card-badge">P(VAGUE) = {r.p_vague:.3f}</span>
                </div>
                <div class="flagged-card-pillar">{icon} &nbsp; {pname} Pillar</div>
                <div class="flagged-card-text">"{snippet}"</div>
                <div class="flagged-card-footer">
                    <span><b>Quant Density</b> &nbsp; {r.quant_score:.2f}</span>
                    <span><b>Citation Score</b> &nbsp; {r.cite_score:.2f}</span>
                    <span><b>Specificity</b> &nbsp; {r.specificity:.3f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("No sentences flagged above threshold — report appears credible.")

    if show_specific:
        st.markdown('<div class="section-subheading">Well-Evidenced Claims</div>',
                    unsafe_allow_html=True)
        specific = sorted(
            [r for r in report.sentence_results if r.label == "SPECIFIC"],
            key=lambda r: r.p_specific, reverse=True
        )[:5]
        for r in specific:
            icon  = pillar_icons.get(r.pillar, "📄")
            pname = pillar_labels.get(r.pillar, r.pillar)
            st.markdown(f"""
            <div class="specific-card">
                <div class="specific-card-header">
                    <span class="specific-card-pillar">{icon} &nbsp; {pname} Pillar</span>
                    <span class="specific-card-badge">P(SPECIFIC) = {r.p_specific:.3f}</span>
                </div>
                <div class="specific-card-text">"{r.text[:260]}"</div>
            </div>
            """, unsafe_allow_html=True)


    # ─────────────────────────────────────────
    # FULL RESULTS TABLE
    # ─────────────────────────────────────────
    st.markdown('<div class="section-heading">Full Sentence-Level Results</div>',
                unsafe_allow_html=True)

    df = pd.DataFrame([{
        "Sentence"    : r.text,
        "Pillar"      : r.pillar,
        "Label"       : r.label,
        "P(VAGUE)"    : round(r.p_vague, 4),
        "P(SPECIFIC)" : round(r.p_specific, 4),
        "Quant Score" : round(r.quant_score, 2),
        "Cite Score"  : round(r.cite_score, 2),
        "Specificity" : round(r.specificity, 3),
    } for r in report.sentence_results])

    f1, f2, f3 = st.columns(3)
    with f1: flag_only = st.checkbox("Flagged sentences only")
    with f2: pillar_f  = st.selectbox("Filter by pillar", ["All", "E", "S", "G"])
    with f3: label_f   = st.selectbox("Filter by label",  ["All", "VAGUE", "SPECIFIC"])

    if flag_only:      df = df[df["P(VAGUE)"] >= threshold]
    if pillar_f != "All": df = df[df["Pillar"] == pillar_f]
    if label_f  != "All": df = df[df["Label"]  == label_f]

    st.dataframe(
        df.style
          .background_gradient(subset=["P(VAGUE)"],    cmap="RdYlGn_r")
          .background_gradient(subset=["P(SPECIFIC)"], cmap="RdYlGn"),
        use_container_width=True,
        height=420
    )


    # ─────────────────────────────────────────
    # DOWNLOADS
    # ─────────────────────────────────────────
    st.markdown("---")
    dl1, dl2 = st.columns(2)

    with dl1:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV — Sentence Results",
            data=csv,
            file_name=f"{display}_esg_analysis.csv",
            mime="text/csv"
        )

    with dl2:
        import json
        json_data = json.dumps(report_to_dict(report), indent=2).encode("utf-8")
        st.download_button(
            "Download JSON — Full Report",
            data=json_data,
            file_name=f"{display}_esg_report.json",
            mime="application/json"
        )


# ─────────────────────────────────────────────
# EMPTY STATE
# ─────────────────────────────────────────────
else:
    st.info("Upload an ESG / sustainability PDF above to begin analysis.")

    st.markdown('<div class="section-heading">What This Tool Detects</div>',
                unsafe_allow_html=True)

    cols = st.columns(5)
    for col, (emoji, title, desc) in zip(cols, [
        ("💬", "Vague Language",       "Claims with no metrics, targets, or timelines"),
        ("🪄", "Weasel Words",         "Phrases like aim, strive, plan, hope"),
        ("📊", "Selective Disclosure", "Cherry-picked data obscuring poor performance"),
        ("❌", "Unverified Claims",    "Assertions lacking audit or certification"),
        ("🔍", "Sentiment Gap",        "Positive tone with no substantive evidence"),
    ]):
        with col:
            st.markdown(f"""
            <div class="empty-card">
                <div class="empty-card-icon">{emoji}</div>
                <div class="empty-card-title">{title}</div>
                <div class="empty-card-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
