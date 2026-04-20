import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import pandas as pd
import io
from detector import VehicleDetector

st.set_page_config(
    page_title="Vehicle Classifier — Sourav Burman",
    page_icon="🚗",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700;800&family=Poppins:wght@400;500&display=swap');
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    background-color: #0C0C0C;
    color: #FAFAFA;
}
h1, h2, h3 {
    font-family: 'Montserrat', sans-serif !important;
    color: #D4AF37 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
section[data-testid="stSidebar"] {
    background-color: #0A0A0A !important;
    border-right: 1px solid #D4AF3755;
}
div[data-testid="metric-container"] {
    background-color: #141414 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 10px !important;
    padding: 1.2rem 1rem !important;
}
[data-testid="metric-container"] label {
    color: #888 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #D4AF37 !important;
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 800 !important;
    font-size: 2rem !important;
}
.stButton > button {
    background-color: #D4AF37 !important;
    color: #0C0C0C !important;
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 6px !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.stButton > button:hover { background-color: #EAD7A1 !important; }
.stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid #1f1f1f !important; }
.stTabs [data-baseweb="tab"] {
    font-family: 'Montserrat', sans-serif !important;
    color: #555 !important;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.8rem !important;
    padding: 0.6rem 1.5rem !important;
}
.stTabs [aria-selected="true"] {
    color: #D4AF37 !important;
    border-bottom: 2px solid #D4AF37 !important;
    background: transparent !important;
}
hr { border-color: #1f1f1f !important; }
.card {
    background: #111;
    border: 1px solid #1f1f1f;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin: 0.4rem 0;
}
.card-accent {
    background: #111;
    border: 1px solid #D4AF3744;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin: 0.4rem 0;
}
.section-label {
    font-family: 'Montserrat', sans-serif;
    font-weight: 800;
    color: #D4AF37;
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding-bottom: 10px;
    border-bottom: 1px solid #D4AF3733;
    margin-bottom: 1rem;
}
.img-label {
    font-family: 'Montserrat', sans-serif;
    font-weight: 700;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.badge-class {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'Montserrat', sans-serif;
    font-weight: 700;
    margin: 3px 2px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.samsung-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 5px 14px;
    border-radius: 6px;
    font-family: 'Montserrat', sans-serif;
    font-weight: 800;
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    background: #1428A0;
    color: #FAFAFA;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;
     padding:18px 0 22px 0;border-bottom:1px solid #D4AF3733;margin-bottom:32px;">
    <div style="display:flex;align-items:center;gap:16px;">
        <div>
            <div style="font-family:'Montserrat',sans-serif;font-weight:800;
                 font-size:1.3rem;color:#D4AF37;letter-spacing:0.1em;">SOURAV BURMAN</div>
            <div style="font-family:'Poppins',sans-serif;font-size:0.68rem;
                 color:#666;letter-spacing:0.18em;margin-top:2px;">CS ENGINEER · BUILDER</div>
        </div>
    </div>
    <div style="display:flex;align-items:center;gap:12px;">
        <span class="samsung-badge">▣ SAMSUNG &nbsp; R&D PROJECT</span>
        <div style="font-family:'Montserrat',sans-serif;font-size:0.62rem;
             color:#D4AF37;opacity:0.35;letter-spacing:0.12em;">⬡ VEHICLE CLASSIFIER</div>
    </div>
</div>
""", unsafe_allow_html=True)

col_title, col_stats = st.columns([2, 1])
with col_title:
    st.markdown("""
    <div style="margin-bottom:2rem;">
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:0.5rem;">
            <span style="font-size:2.2rem;">🚗</span>
            <h1 style="font-size:2rem;margin:0;line-height:1.1;">
                VEHICLE DETECTION<br>& CLASSIFICATION
            </h1>
        </div>
        <p style="color:#666;font-size:0.85rem;margin:0;letter-spacing:0.02em;padding-left:3.5rem;">
            YOLOv8s · Real-time dash-cam analysis · Samsung R&D Innovation Campus, Kolkata
        </p>
    </div>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:1.2rem;">
        <span class="samsung-badge" style="font-size:0.68rem;">▣ SAMSUNG &nbsp; R&D PROJECT</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="section-label">Settings</div>', unsafe_allow_html=True)
    confidence = st.slider("Detection Confidence", 0.1, 0.9, 0.25, 0.05)
    st.markdown('<div class="section-label">Vehicle Classes</div>', unsafe_allow_html=True)
    for icon, name, color in [("🚗", "Car", "#1D6FF0"), ("🏍️", "Motorcycle", "#00C896"), ("🚌", "Bus", "#FF4757"), ("🚛", "Truck / SUV", "#FFA502"), ("🚲", "Bicycle", "#A855F7")]:
        st.markdown(f'<span class="badge-class" style="background:{color}18;color:{color};border:1px solid {color}55;">{icon} {name}</span>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-label" style="margin-top:1.5rem;">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.82rem;color:#AAA;line-height:2.1;">
        <span style="color:#D4AF37;font-weight:700;font-family:Montserrat,sans-serif;">Sourav Burman</span><br/>
        Samsung R&D Innovation Campus<br/>
        Brainware University · CSE '27
    </div>
    <div style="margin-top:0.9rem;display:flex;flex-direction:column;gap:8px;">
        <a href="mailto:thesouravburman@gmail.com" style="color:#D4AF37;font-size:0.78rem;font-family:Montserrat,sans-serif;font-weight:700;text-decoration:none;">✉ thesouravburman@gmail.com</a>
        <div style="display:flex;gap:12px;">
            <a href="https://github.com/thesouravburman" style="color:#1D6FF0;font-size:0.78rem;font-family:Montserrat,sans-serif;font-weight:700;text-decoration:none;">GITHUB</a>
            <span style="color:#333;">·</span>
            <a href="https://linkedin.com/in/sourav-burman" style="color:#00C896;font-size:0.78rem;font-family:Montserrat,sans-serif;font-weight:700;text-decoration:none;">LINKEDIN</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["  🔍 DETECTION  ", "  ⚙️ HOW IT WORKS  ", "  📋 PROJECT INFO  "])

with tab1:
    uploaded = st.file_uploader("Upload a traffic or dash-cam image", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        detector = VehicleDetector(confidence=confidence)
        annotated, detections = detector.detect(image)
        st.image(annotated, use_column_width=True)

with tab2:
    st.markdown("## HOW IT WORKS")

with tab3:
    st.markdown("<h2 style='margin-bottom:1.5rem;'>PROJECT INFO</h2>", unsafe_allow_html=True)
    st.markdown("<br><h3 style='margin-bottom:1rem;'>CREATOR</h3>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="card" style="border-top:2px solid #D4AF37;padding:1.4rem 1.6rem;">
        <div style="display:flex;align-items:center;gap:1.4rem;">
            <div style="width:56px;height:56px;border-radius:50%;background:#D4AF3722;border:2px solid #D4AF37;display:flex;align-items:center;justify-content:center;font-family:Montserrat,sans-serif;font-weight:800;font-size:1.4rem;color:#D4AF37;flex-shrink:0;">S</div>
            <div>
                <div style="font-family:'Montserrat',sans-serif;font-weight:800;color:#D4AF37;font-size:1rem;letter-spacing:0.05em;">Sourav Burman</div>
                <div style="color:#888;font-size:0.82rem;margin-top:4px;">B.Tech CSE · Brainware University · Samsung R&D Innovation Campus</div>
                <div style="display:flex;gap:14px;margin-top:10px;">
                    <a href="mailto:thesouravburman@gmail.com" style="color:#D4AF37;font-size:0.78rem;font-family:Montserrat,sans-serif;font-weight:700;text-decoration:none;">✉ thesouravburman@gmail.com</a>
                    <a href="https://github.com/thesouravburman" style="color:#1D6FF0;font-size:0.78rem;font-family:Montserrat,sans-serif;font-weight:700;text-decoration:none;">GITHUB</a>
                    <a href="https://linkedin.com/in/sourav-burman" style="color:#00C896;font-size:0.78rem;font-family:Montserrat,sans-serif;font-weight:700;text-decoration:none;">LINKEDIN</a>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
