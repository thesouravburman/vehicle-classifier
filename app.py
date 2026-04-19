import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from detector import VehicleDetector

st.set_page_config(
    page_title="Vehicle Classifier — Samsung R&D",
    page_icon="🚗",
    layout="wide"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #0066CC, #00AAFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #1E2130;
        border: 1px solid #2E3250;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-number {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00AAFF;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #888;
        margin-top: 0.2rem;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px;
    }
    .footer {
        text-align: center;
        color: #555;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #2E3250;
    }
    div[data-testid="stSidebar"] {
        background: #1E2130;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/Samsung-R%26D%20Project-1428A0?style=for-the-badge&logo=samsung&logoColor=white")
    st.markdown("### ⚙️ Settings")
    confidence = st.slider("Detection Confidence", 0.1, 0.9, 0.4, 0.05,
                           help="Higher = fewer but more certain detections")
    st.markdown("---")
    st.markdown("### 📋 Vehicle Classes")
    classes = {"🚗 Car": "#1E90FF", "🏍️ Motorcycle": "#00FF7F",
               "🚌 Bus": "#DC143C", "🚛 Truck": "#9400D3", "🚲 Bicycle": "#FFA500"}
    for name, color in classes.items():
        st.markdown(f'<span class="badge" style="background:{color}22;color:{color};border:1px solid {color}">{name}</span>',
                    unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 👨‍💻 About")
    st.markdown("""
    Built by **Sourav Burman**  
    Samsung R&D Innovation Campus  
    Brainware University · CSE '27  
    [GitHub](https://github.com/thesouravburman) · [LinkedIn](https://linkedin.com/in/sourav-burman)
    """)

# ── Header ───────────────────────────────────────────
st.markdown('<div class="main-title">🚗 Vehicle Detection & Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-time vehicle detection from dash-cam footage · Powered by YOLOv8 · Samsung R&D Innovation Campus</div>', unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📸 Image Detection", "ℹ️ How It Works", "📊 Project Info"])

# ── Tab 1: Detection ─────────────────────────────────
with tab1:
    uploaded = st.file_uploader(
        "Upload a traffic image (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload any road or traffic photo"
    )

    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        image_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        with st.spinner("🔍 Detecting vehicles..."):
            detector = VehicleDetector(confidence=confidence)
            annotated_bgr, detections = detector.detect(image_bgr)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original Image**")
            st.image(image_rgb, use_column_width=True)
        with col2:
            st.markdown("**Detected Vehicles**")
            st.image(annotated_rgb, use_column_width=True)

        st.markdown("---")

        if detections:
            counts = {}
            for d in detections:
                counts[d["label"]] = counts.get(d["label"], 0) + 1

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'''<div class="metric-card">
                    <div class="metric-number">{len(detections)}</div>
                    <div class="metric-label">Total Vehicles</div></div>''', unsafe_allow_html=True)
            with c2:
                avg_conf = sum(d["confidence"] for d in detections) / len(detections)
                st.markdown(f'''<div class="metric-card">
                    <div class="metric-number">{avg_conf:.0%}</div>
                    <div class="metric-label">Avg Confidence</div></div>''', unsafe_allow_html=True)
            with c3:
                st.markdown(f'''<div class="metric-card">
                    <div class="metric-number">{len(counts)}</div>
                    <div class="metric-label">Vehicle Types</div></div>''', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            col_chart, col_table = st.columns([1, 1])

            with col_chart:
                color_map = {"Car": "#1E90FF", "Motorcycle": "#00FF7F",
                             "Bus": "#DC143C", "Truck": "#9400D3", "Bicycle": "#FFA500"}
                fig = go.Figure(go.Bar(
                    x=list(counts.keys()),
                    y=list(counts.values()),
                    marker_color=[color_map.get(k, "#888") for k in counts.keys()],
                    text=list(counts.values()),
                    textposition="outside"
                ))
                fig.update_layout(
                    title="Vehicles Detected by Type",
                    paper_bgcolor="#1E2130",
                    plot_bgcolor="#1E2130",
                    font_color="#FFF",
                    showlegend=False,
                    margin=dict(t=40, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_table:
                st.markdown("**Detection Details**")
                df = pd.DataFrame([{
                    "Vehicle": d["label"],
                    "Confidence": f"{d['confidence']:.1%}",
                    "Bounding Box": f"({d['bbox'][0]}, {d['bbox'][1]}) → ({d['bbox'][2]}, {d['bbox'][3]})"
                } for d in detections])
                st.dataframe(df, use_container_width=True, hide_index=True)

            result_img = Image.fromarray(annotated_rgb)
            import io
            buf = io.BytesIO()
            result_img.save(buf, format="PNG")
            st.download_button("⬇️ Download Annotated Image", buf.getvalue(),
                               "detected_vehicles.png", "image/png")
        else:
            st.info("No vehicles detected. Try lowering the confidence threshold in the sidebar.")

    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem;border:2px dashed #2E3250;border-radius:16px;margin:1rem 0">
            <div style="font-size:3rem">📸</div>
            <div style="font-size:1.1rem;color:#888;margin-top:0.5rem">
                Upload a traffic or road image to get started
            </div>
            <div style="font-size:0.85rem;color:#555;margin-top:0.5rem">
                Supports JPG and PNG · Max 200MB
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Tab 2: How It Works ───────────────────────────────
with tab2:
    st.markdown("## How It Works")
    steps = [
        ("1️⃣", "Input", "You upload a traffic or dash-cam image"),
        ("2️⃣", "Preprocessing", "Image is resized and normalised for YOLOv8"),
        ("3️⃣", "Detection", "YOLOv8 scans every region of the image for vehicles"),
        ("4️⃣", "Classification", "Each detected object is assigned a vehicle class and confidence score"),
        ("5️⃣", "Annotation", "Bounding boxes and labels are drawn on the original image"),
        ("6️⃣", "Analytics", "A full breakdown of vehicle counts and statistics is generated"),
    ]
    for icon, title, desc in steps:
        st.markdown(f"""
        <div style="background:#1E2130;border-left:4px solid #0066CC;
                    border-radius:8px;padding:1rem 1.2rem;margin:0.5rem 0">
            <span style="font-size:1.2rem">{icon}</span>
            <strong style="color:#00AAFF;margin-left:0.5rem">{title}</strong>
            <span style="color:#CCC;margin-left:0.5rem">— {desc}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## YOLOv8 — The AI Model")
    st.markdown("""
    **YOLO** stands for *You Only Look Once*. It's one of the fastest and most accurate
    object detection models in the world. Here's what makes it special:

    - It looks at the **entire image in one pass** — unlike older methods that scan regions one by one
    - It outputs **bounding boxes + class labels + confidence scores** all at once
    - It runs at **30+ frames per second** on standard hardware
    - The `yolov8n` (nano) version we use is optimised for speed while maintaining high accuracy

    This model was the core of our Samsung R&D Innovation Campus project.
    """)

# ── Tab 3: Project Info ───────────────────────────────
with tab3:
    st.markdown("## Project Background")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Developed at:** Samsung R&D Innovation Campus, Kolkata  
        **Programme:** AI & API Integration  
        **Duration:** Sept – Nov 2025  
        **Team:** Sourav Burman, Sathi Santra, Shovan Mondal, Tulika Adak
        """)
    with col2:
        st.markdown("""
        **Model:** YOLOv8n (Ultralytics)  
        **Accuracy:** 90%+ mAP across vehicle classes  
        **Speed:** 30+ FPS on GPU-accelerated hardware  
        **Classes:** Car, Motorcycle, Bus, Truck, Bicycle
        """)

    st.markdown("---")
    st.markdown("## Performance Highlights")
    metrics = {"Detection Accuracy": "90%+", "Processing Speed": "30+ FPS",
                "Vehicle Classes": "5", "Confidence Threshold": "Adjustable"}
    cols = st.columns(4)
    for col, (label, value) in zip(cols, metrics.items()):
        with col:
            st.markdown(f'''<div class="metric-card">
                <div class="metric-number">{value}</div>
                <div class="metric-label">{label}</div></div>''', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    ## Real-World Applications
    - 🚦 **Traffic Monitoring** — Vehicle counting and flow analysis
    - 🚔 **Law Enforcement** — Automated violation detection
    - 🚚 **Fleet Management** — Commercial vehicle tracking
    - 🚗 **ADAS** — Collision warnings and driver assistance
    """)

# ── Footer ────────────────────────────────────────────
st.markdown('''<div class="footer">
    Built by Sourav Burman · Samsung R&D Innovation Campus ·
    <a href="https://github.com/thesouravburman/vehicle-classifier" style="color:#0066CC">
    GitHub Repo</a>
</div>''', unsafe_allow_html=True)
