import json
from pathlib import Path
from typing import Dict

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Sentiment Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# MODERN PROFESSIONAL CSS THEME
# =========================================================

st.markdown(
    """
<style>
:root {
    --bg-main: #0b1220;
    --bg-card: rgba(20, 30, 48, 0.6);
    --bg-card-strong: rgba(15, 23, 42, 0.85);
    --border-card: rgba(148, 163, 184, 0.18);
    --text-main: #e2e8f0;
    --text-muted: #94a3b8;
    --accent: #38bdf8;
}

.stApp {
    background: radial-gradient(1200px 400px at 15% -10%, rgba(56, 189, 248, 0.12), transparent),
                linear-gradient(180deg, #0b1220 0%, #020617 100%);
}

.header {
    padding: 10px 0 20px 0;
}

.title {
    font-size: 40px;
    font-weight: 700;
    color: var(--text-main);
    margin-bottom: 6px;
}

.subtitle {
    font-size: 16px;
    color: var(--text-muted);
    margin-bottom: 22px;
}

.section-title {
    font-size: 22px;
    font-weight: 600;
    color: var(--text-main);
    margin: 24px 0 14px 0;
}

.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border-card);
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    backdrop-filter: blur(12px);
}

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 14px;
}

.kpi-card {
    background: var(--bg-card-strong);
    border: 1px solid var(--border-card);
    border-radius: 14px;
    padding: 16px;
}

.kpi-label {
    color: var(--text-muted);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
}

.kpi-value {
    color: var(--text-main);
    font-size: 22px;
    font-weight: 700;
}

.topic-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid var(--border-card);
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 12px;
}

.topic-title {
    color: var(--text-main);
    font-weight: 600;
    margin-bottom: 8px;
}

.topic-words {
    color: var(--text-muted);
    font-size: 14px;
}

.footer {
    text-align: center;
    color: var(--text-muted);
    margin-top: 30px;
    font-size: 12px;
}

@media (max-width: 1200px) {
    .kpi-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 720px) {
    .kpi-grid {
        grid-template-columns: 1fr;
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# PATH SETUP
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"


# =========================================================
# LOAD FUNCTIONS
# =========================================================

def load_json(name: str):

    path = OUTPUT_DIR / name

    if not path.exists():
        st.error(f"Missing file: {path}")
        st.stop()

    return json.loads(path.read_text())


def load_csv(name: str):

    path = OUTPUT_DIR / name

    if not path.exists():
        st.error(f"Missing file: {path}")
        st.stop()

    return pd.read_csv(path)


def load_html(name: str):

    path = OUTPUT_DIR / name

    if not path.exists():
        st.error(f"Missing file: {path}")
        st.stop()

    return path.read_text()


# =========================================================
# LOAD DATA
# =========================================================

metrics = load_json("sentiment_metrics.json")
predictions = load_csv("sentiment_predictions.csv")
topics = load_json("topics.json")
cleaned = load_csv("preprocessed_data.csv")
lda_html = load_html("lda_visualization.html")


# =========================================================
# HEADER
# =========================================================

st.markdown(
        """
<div class="header">
    <div class="title">Sentiment Intelligence Dashboard</div>
    <div class="subtitle">Executive analytics for sentiment performance and topic intelligence</div>
</div>
""",
        unsafe_allow_html=True,
)


# =========================================================
# KPI SECTION
# =========================================================

st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)

kpi_html = f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">Accuracy</div>
        <div class="kpi-value">{metrics['accuracy']:.3f}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Precision (Macro)</div>
        <div class="kpi-value">{metrics['precision_macro']:.3f}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Recall (Macro)</div>
        <div class="kpi-value">{metrics['recall_macro']:.3f}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">F1 Score (Macro)</div>
        <div class="kpi-value">{metrics['f1_score_macro']:.3f}</div>
    </div>
</div>
"""

st.markdown(kpi_html, unsafe_allow_html=True)


# =========================================================
# DATA OVERVIEW
# =========================================================

st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)

overview_html = f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">Total Cleaned Records</div>
        <div class="kpi-value">{len(cleaned):,}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Test Dataset Size</div>
        <div class="kpi-value">{len(predictions):,}</div>
    </div>
</div>
"""

st.markdown(overview_html, unsafe_allow_html=True)


# =========================================================
# SENTIMENT DISTRIBUTION (FIXED LAYOUT)
# =========================================================

st.markdown('<div class="section-title">Sentiment Distribution</div>', unsafe_allow_html=True)

chart_col, table_col = st.columns([3,2], gap="large")


# Chart (NO OVERLAP FIX)
with chart_col:

    counts = predictions["predicted_sentiment"].value_counts()

    fig = px.bar(
        x=counts.index,
        y=counts.values,
        color=counts.index,
        title="Sentiment Distribution (Test Set)",
        text=counts.values,
        height=420,
        labels={"x": "Sentiment", "y": "Count"},
        color_discrete_sequence=["#38bdf8", "#f97316", "#a78bfa"],
    )

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
        title_font=dict(size=16),
    )

    st.plotly_chart(fig, use_container_width=True)


# Table
with table_col:

    df = pd.DataFrame({

        "Sentiment": counts.index,

        "Count": counts.values,

        "Percentage": (counts.values / len(predictions) * 100).round(2)

    })

    st.dataframe(df, use_container_width=True)


# =========================================================
# TOPIC SECTION
# =========================================================

st.markdown('<div class="section-title">Topic Intelligence</div>', unsafe_allow_html=True)

for topic, words in topics.items():

    topic_id = topic.split("_")[1]

    st.markdown(
        f"""
    <div class="topic-card">
        <div class="topic-title">Topic {topic_id}</div>
        <div class="topic-words">{", ".join(words)}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )


# =========================================================
# LDA VISUALIZATION
# =========================================================

st.markdown('<div class="section-title">Interactive Topic Visualization</div>', unsafe_allow_html=True)

components.html(

    lda_html,

    height=800,

    scrolling=True
)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
<div class="footer">
Production ML Pipeline • Logistic Regression • LDA Topic Modeling • Streamlit Dashboard
</div>
""",
    unsafe_allow_html=True,
)
