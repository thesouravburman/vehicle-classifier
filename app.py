import streamlit as st
import cv2
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
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700;800&family=Poppins:wght@400;500&display=swap');

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
    background-color: #0C0C0C !important;
    border-right: 1px solid #D4AF37;
}
div[data-testid="metric-container"] {
    background-color: #1F1F1F !important;
    border: 1px solid #D4AF37 !important;
    border-radius: 8px !important;
    padding: 1rem !important;
}
[data-testid="metric-container"] label {
    color: #EAD7A1 !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 0.8rem !important;
}
[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #D4AF37 !important;
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 800 !important;
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
.stButton > button:hover {
    background-color: #EAD7A1 !important;
    color: #0C0C0C !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Montserrat', sans-serif !important;
    color: #EAD7A1 !important;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.stTabs [aria-selected="true"] {
    color: #D4AF37 !important;
    border-bottom: 2px solid #D4AF37 !important;
}
.stSlider label {
    color: #EAD7A1 !important;
    font-family: 'Poppins', sans-serif !important;
}
.dataframe thead tr th {
    background-color: #D4AF37 !important;
    color: #0C0C0C !important;
    font-family: 'Montserrat', sans-serif !important;
}
.dataframe tbody tr {
    background-color: #1F1F1F !important;
    color: #FAFAFA !important;
}
hr { border-color: #D4AF37 !important; opacity: 0.3; }
.upload-box {
    border: 2px dashed #D4AF37;
    border-radius: 10px;
    padding: 2.5rem;
    text-align: center;
    background: #1F1F1F;
    margin: 1rem 0;
}
.result-card {
    background: #1F1F1F;
    border: 1px solid #D4AF37;
    border-radius: 8px;
    padding: 1.2rem;
    margin: 0.5rem 0;
}
.badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-family: 'Montserrat', sans-serif;
    font-weight: 700;
    margin: 2px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)

# ── Brand Header ───────────────────────────────────
st.markdown("""
<div style="display:flex; align-items:center; justify-content:space-between;
     padding:16px 0 24px 0; border-bottom:1px solid #D4AF37; margin-bottom:28px;">
    <div>
        <div style="font-family:'Montserrat',sans-serif; font-weight:800;
             font-size:1.2rem; color:#D4AF37; letter-spacing:0.08em;">
            SOURAV BURMAN
        </div>
        <div style="font-family:'Poppins',sans-serif; font-size:0.72rem;
             color:#EAD7A1; letter-spacing:0.15em; margin-top:2px;">
            CS ENGINEER · BUILDER
        </div>
    </div>
    <div style="font-family:'Montserrat',sans-serif; font-size:0.7rem;
         color:#D4AF37; opacity:0.5; letter-spacing:0.1em;">
        ⬡ VEHICLE CLASSIFIER
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style="font-size:2rem; margin-bottom:0.2rem;">VEHICLE DETECTION & CLASSIFICATION</h1>
<p style="color:#EAD7A1; font-size:0.95rem; margin-bottom:2rem;">
    Real-time vehicle detection from dash-cam footage · YOLOv8 ·
    Samsung R&D Innovation Campus, Kolkata
</p>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Montserrat',sans-serif; font-weight:800;
         color:#D4AF37; font-size:0.85rem; letter-spacing:0.1em;
         margin-bottom:1rem; padding-bottom:8px; border-bottom:1px solid #D4AF37;">
        SETTINGS
    </div>
    """, unsafe_allow_html=True)

    confidence = st.slider("Detection Confidence", 0.1, 0.9, 0.4, 0.05)

    st.markdown("""
    <div style="font-family:'Montserrat',sans-serif; font-weight:800;
         color:#D4AF37; font-size:0.85rem; letter-spacing:0.1em;
         margin:1.5rem 0 0.8rem; padding-bottom:8px; border-bottom:1px solid #D4AF37;">
        VEHICLE CLASSES
    </div>
    """, unsafe_allow_html=True)

    classes = [
        ("Car",        "#1D4ED8"),
        ("Motorcycle", "#D4AF37"),
        ("Bus",        "#DC143C"),
        ("Truck",      "#EAD7A1"),
        ("Bicycle",    "#FAFAFA"),
    ]
    for name, color in classes:
        st.markdown(
            f'''<span class="badge"
            style="background:{color}22;color:{color};border:1px solid {color}">
            {name}</span>''',
            unsafe_allow_html=True
        )

    st.markdown("""
    <div style="font-family:'Montserrat',sans-serif; font-weight:800;
         color:#D4AF37; font-size:0.85rem; letter-spacing:0.1em;
         margin:1.5rem 0 0.8rem; padding-bottom:8px; border-bottom:1px solid #D4AF37;">
        ABOUT
    </div>
    <div style="font-family:'Poppins',sans-serif; font-size:0.8rem; color:#EAD7A1; line-height:1.8;">
        Built by <span style="color:#D4AF37; font-weight:600;">Sourav Burman</span><br/>
        Samsung R&D Innovation Campus<br/>
        Brainware University · CSE '27<br/><br/>
        <a href="https://github.com/thesouravburman" style="color:#D4AF37;">GitHub</a>
        &nbsp;·&nbsp;
        <a href="https://linkedin.com/in/sourav-burman" style="color:#D4AF37;">LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["DETECTION", "HOW IT WORKS", "PROJECT INFO"])

# ── Tab 1: Detection ───────────────────────────────
with tab1:
    uploaded = st.file_uploader(
        "Upload a traffic image (JPG, PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        image_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        with st.spinner("Running detection..."):
            detector = VehicleDetector(confidence=confidence)
            annotated_bgr, detections = detector.detect(image_bgr)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p style="color:#EAD7A1;font-family:Montserrat,sans-serif;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;font-size:0.85rem;">Original Image</p>', unsafe_allow_html=True)
            st.image(image_rgb, use_column_width=True)
        with col2:
            st.markdown('<p style="color:#EAD7A1;font-family:Montserrat,sans-serif;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;font-size:0.85rem;">Detected Vehicles</p>', unsafe_allow_html=True)
            st.image(annotated_rgb, use_column_width=True)

        st.markdown("---")

        if detections:
            counts = {}
            for d in detections:
                counts[d["label"]] = counts.get(d["label"], 0) + 1
            avg_conf = sum(d["confidence"] for d in detections) / len(detections)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total Vehicles", len(detections))
            with c2:
                st.metric("Avg Confidence", f"{avg_conf:.0%}")
            with c3:
                st.metric("Vehicle Types", len(counts))

            st.markdown("<br>", unsafe_allow_html=True)
            col_chart, col_table = st.columns([1, 1])

            color_map = {
                "Car": "#1D4ED8", "Motorcycle": "#D4AF37",
                "Bus": "#DC143C", "Truck": "#EAD7A1", "Bicycle": "#FAFAFA"
            }

            with col_chart:
                fig = go.Figure(go.Bar(
                    x=list(counts.keys()),
                    y=list(counts.values()),
                    marker_color=[color_map.get(k, "#D4AF37") for k in counts.keys()],
                    text=list(counts.values()),
                    textposition="outside",
                    textfont=dict(color="#FAFAFA", family="Montserrat")
                ))
                fig.update_layout(
                    title=dict(text="VEHICLES BY TYPE", font=dict(
                        family="Montserrat", color="#D4AF37", size=13)),
                    paper_bgcolor="#0C0C0C",
                    plot_bgcolor="#1F1F1F",
                    font=dict(color="#FAFAFA", family="Poppins"),
                    showlegend=False,
                    margin=dict(t=40, b=20),
                    xaxis=dict(gridcolor="#2a2a2a"),
                    yaxis=dict(gridcolor="#2a2a2a"),
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_table:
                st.markdown('<p style="color:#EAD7A1;font-family:Montserrat,sans-serif;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;font-size:0.85rem;">Detection Details</p>', unsafe_allow_html=True)
                df = pd.DataFrame([{
                    "Vehicle": d["label"],
                    "Confidence": f"{d['confidence']:.1%}",
                    "Box": f"({d['bbox'][0]},{d['bbox'][1]})→({d['bbox'][2]},{d['bbox'][3]})"
                } for d in detections])
                st.dataframe(df, use_container_width=True, hide_index=True)

            buf = io.BytesIO()
            Image.fromarray(annotated_rgb).save(buf, format="PNG")
            st.download_button(
                "Download Annotated Image",
                buf.getvalue(),
                "detected_vehicles.png",
                "image/png"
            )
        else:
            st.info("No vehicles detected. Lower the confidence threshold in the sidebar.")
    else:
        st.markdown("""
        <div class="upload-box">
            <div style="font-family:'Montserrat',sans-serif; font-weight:800;
                 color:#D4AF37; font-size:1rem; letter-spacing:0.08em;">
                UPLOAD A TRAFFIC IMAGE
            </div>
            <div style="color:#EAD7A1; font-size:0.85rem; margin-top:0.5rem;">
                Supports JPG and PNG · Max 200MB
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Tab 2: How It Works ────────────────────────────
with tab2:
    st.markdown("<h2>HOW IT WORKS</h2>", unsafe_allow_html=True)
    steps = [
        ("01", "INPUT",          "Upload a traffic or dash-cam image"),
        ("02", "PREPROCESSING",  "Image is resized and normalised for the YOLOv8 model"),
        ("03", "DETECTION",      "YOLOv8 scans every region of the image simultaneously"),
        ("04", "CLASSIFICATION", "Each object is assigned a vehicle class and confidence score"),
        ("05", "ANNOTATION",     "Bounding boxes and labels are drawn on the original image"),
        ("06", "ANALYTICS",      "Full breakdown of vehicle counts and statistics is generated"),
    ]
    for num, title, desc in steps:
        st.markdown(f"""
        <div class="result-card" style="display:flex; align-items:center; gap:1.2rem;">
            <div style="font-family:'Montserrat',sans-serif; font-weight:800;
                 font-size:1.4rem; color:#D4AF37; min-width:36px;">{num}</div>
            <div>
                <div style="font-family:'Montserrat',sans-serif; font-weight:700;
                     color:#D4AF37; font-size:0.85rem; letter-spacing:0.08em;">{title}</div>
                <div style="color:#EAD7A1; font-size:0.85rem; margin-top:2px;">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><h3>YOLOV8 — THE MODEL</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class="result-card">
        <p style="color:#EAD7A1; line-height:1.9; font-size:0.9rem;">
        YOLO stands for <span style="color:#D4AF37; font-weight:600;">You Only Look Once</span>.
        It processes the entire image in a single pass — outputting bounding boxes,
        class labels, and confidence scores simultaneously. The YOLOv8n model achieves
        <span style="color:#D4AF37; font-weight:600;">90%+ mAP</span> across vehicle classes
        at <span style="color:#D4AF37; font-weight:600;">30+ FPS</span> on standard hardware.
        This was the core model used in our Samsung R&D Innovation Campus project.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ── Tab 3: Project Info ────────────────────────────
with tab3:
    st.markdown("<h2>PROJECT INFO</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="result-card">
            <div style="font-family:'Montserrat',sans-serif; font-weight:800;
                 color:#D4AF37; font-size:0.8rem; letter-spacing:0.1em; margin-bottom:12px;">
                BACKGROUND
            </div>
            <div style="color:#EAD7A1; font-size:0.85rem; line-height:2;">
                Venue: Samsung R&D Innovation Campus, Kolkata<br/>
                Programme: AI & API Integration<br/>
                Duration: Sept – Nov 2025<br/>
                Team: 4 members
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="result-card">
            <div style="font-family:'Montserrat',sans-serif; font-weight:800;
                 color:#D4AF37; font-size:0.8rem; letter-spacing:0.1em; margin-bottom:12px;">
                PERFORMANCE
            </div>
            <div style="color:#EAD7A1; font-size:0.85rem; line-height:2;">
                Model: YOLOv8n (Ultralytics)<br/>
                Accuracy: 90%+ mAP<br/>
                Speed: 30+ FPS on GPU<br/>
                Classes: 5 vehicle types
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><h3>TEAM</h3>", unsafe_allow_html=True)
    team = [
        ("Sourav Kumar Burman", "AI Pipeline & API Integration"),
        ("Sathi Santra",        "Model Training & Evaluation"),
        ("Shovan Mondal",       "Video Processing"),
        ("Tulika Adak",         "Visualisation & Reporting"),
    ]
    cols = st.columns(4)
    for col, (name, role) in zip(cols, team):
        with col:
            st.markdown(f"""
            <div class="result-card" style="text-align:center;">
                <div style="font-family:'Montserrat',sans-serif; font-weight:700;
                     color:#D4AF37; font-size:0.8rem;">{name}</div>
                <div style="color:#EAD7A1; font-size:0.75rem; margin-top:4px;">{role}</div>
            </div>
            """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:12px 0; font-family:'Poppins',sans-serif;
     font-size:0.75rem; color:#EAD7A1; letter-spacing:0.08em; opacity:0.6;">
    Built by <span style="color:#D4AF37; font-weight:600;">Sourav Burman</span>
    · Samsung R&D Innovation Campus ·
    <a href="https://github.com/thesouravburman/vehicle-classifier"
       style="color:#D4AF37;">GitHub Repo</a>
</div>
""", unsafe_allow_html=True)
