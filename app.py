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
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 1px solid #1f1f1f !important;
}
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

# ── Brand Header ───────────────────────────────────
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
        <span class="samsung-badge">
            ▣ SAMSUNG &nbsp; R&D PROJECT
        </span>
        <div style="font-family:'Montserrat',sans-serif;font-size:0.62rem;
             color:#D4AF37;opacity:0.35;letter-spacing:0.12em;">⬡ VEHICLE CLASSIFIER</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Page Title ─────────────────────────────────────
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

# ── Sidebar ────────────────────────────────────────
with st.sidebar:
    # Samsung badge in sidebar too
    st.markdown("""
    <div style="margin-bottom:1.2rem;">
        <span class="samsung-badge" style="font-size:0.68rem;">▣ SAMSUNG &nbsp; R&D PROJECT</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Settings</div>', unsafe_allow_html=True)
    confidence = st.slider("Detection Confidence", 0.1, 0.9, 0.25, 0.05)
    sensitivity = "High sensitivity" if confidence < 0.35 else "Balanced" if confidence < 0.6 else "High precision"
    st.markdown(f"""
    <div style="font-size:0.73rem;color:#666;margin-top:-0.5rem;margin-bottom:1.2rem;">
        {confidence:.0%} — {sensitivity}
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Vehicle Classes</div>', unsafe_allow_html=True)

    classes = [
        ("🚗", "Car",          "#1D6FF0"),
        ("🏍️", "Motorcycle",  "#00C896"),
        ("🚌", "Bus",          "#FF4757"),
        ("🚛", "Truck / SUV",  "#FFA502"),
        ("🚲", "Bicycle",      "#A855F7"),
    ]
    for icon, name, color in classes:
        st.markdown(f'''
        <span class="badge-class"
            style="background:{color}18;color:{color};border:1px solid {color}55;">
            {icon} {name}
        </span>''', unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:1.5rem;">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.82rem;color:#AAA;line-height:2.1;">
        <span style="color:#D4AF37;font-weight:700;font-family:Montserrat,sans-serif;">
        Sourav Burman</span><br/>
        Samsung R&D Innovation Campus<br/>
        Brainware University · CSE '27
    </div>
    <div style="margin-top:0.9rem;display:flex;gap:12px;">
        <a href="https://github.com/thesouravburman"
           style="color:#1D6FF0;font-size:0.78rem;font-family:Montserrat,sans-serif;
           font-weight:700;letter-spacing:0.05em;text-decoration:none;">GITHUB</a>
        <span style="color:#333;">·</span>
        <a href="https://linkedin.com/in/sourav-burman"
           style="color:#00C896;font-size:0.78rem;font-family:Montserrat,sans-serif;
           font-weight:700;letter-spacing:0.05em;text-decoration:none;">LINKEDIN</a>
    </div>
    """, unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["  🔍 DETECTION  ", "  ⚙️ HOW IT WORKS  ", "  📋 PROJECT INFO  "])

with tab1:
    uploaded = st.file_uploader(
        "Upload a traffic or dash-cam image",
        type=["jpg", "jpeg", "png"],
        help="Supports JPG and PNG up to 200MB"
    )

    if not uploaded:
        st.markdown("""
        <div style="border:2px dashed #D4AF3755;border-radius:14px;padding:3.5rem 2rem;
             text-align:center;background:#0D0D0D;margin:1rem 0;">
            <div style="font-size:3rem;margin-bottom:1rem;">🚗 🚌 🏍️</div>
            <div style="font-family:'Montserrat',sans-serif;font-weight:700;
                 color:#D4AF37;font-size:1rem;letter-spacing:0.1em;">
                DROP A TRAFFIC IMAGE HERE</div>
            <div style="color:#444;font-size:0.82rem;margin-top:0.5rem;">
                JPG or PNG · Any road or traffic scene · Up to 200MB</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        image = Image.open(uploaded).convert("RGB")
        with st.spinner("Running YOLOv8s detection..."):
            detector = VehicleDetector(confidence=confidence)
            annotated, detections = detector.detect(image)

        col1, col2 = st.columns(2, gap="medium")
        with col1:
            st.markdown('<div class="img-label" style="color:#888;">Original Image</div>',
                        unsafe_allow_html=True)
            st.image(image, use_column_width=True)
        with col2:
            st.markdown('<div class="img-label" style="color:#D4AF37;">Detected Vehicles</div>',
                        unsafe_allow_html=True)
            st.image(annotated, use_column_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if detections:
            counts = {}
            for d in detections:
                counts[d["label"]] = counts.get(d["label"], 0) + 1
            avg_conf = sum(d["confidence"] for d in detections) / len(detections)
            top_conf = detections[0]["confidence"] if detections else 0

            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Total Detected", len(detections))
            with c2: st.metric("Avg Confidence", f"{avg_conf:.0%}")
            with c3: st.metric("Vehicle Types", len(counts))
            with c4: st.metric("Top Confidence", f"{top_conf:.0%}")

            st.markdown("<br>", unsafe_allow_html=True)
            col_chart, col_table = st.columns([1.1, 0.9], gap="medium")

            color_map = {
                "Car":        "#1D6FF0",
                "Motorcycle": "#00C896",
                "Bus":        "#FF4757",
                "Truck / SUV":"#FFA502",
                "Bicycle":    "#A855F7",
            }

            with col_chart:
                fig = go.Figure(go.Bar(
                    x=list(counts.keys()),
                    y=list(counts.values()),
                    marker_color=[color_map.get(k,"#D4AF37") for k in counts.keys()],
                    marker_line_color="#0C0C0C",
                    marker_line_width=2,
                    text=list(counts.values()),
                    textposition="outside",
                    textfont=dict(color="#FAFAFA", family="Montserrat", size=13)
                ))
                fig.update_layout(
                    title=dict(text="VEHICLES BY TYPE",
                        font=dict(family="Montserrat", color="#D4AF37", size=12), x=0),
                    paper_bgcolor="#0C0C0C",
                    plot_bgcolor="#111",
                    font=dict(color="#FAFAFA", family="Poppins"),
                    showlegend=False,
                    margin=dict(t=40, b=10, l=10, r=10),
                    xaxis=dict(gridcolor="#1a1a1a",
                        tickfont=dict(family="Montserrat", color="#AAA", size=11)),
                    yaxis=dict(gridcolor="#1a1a1a", tickfont=dict(color="#444")),
                    bargap=0.35,
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_table:
                st.markdown('<div class="img-label" style="color:#888;margin-bottom:8px;">Detection Details</div>',
                            unsafe_allow_html=True)
                df = pd.DataFrame([{
                    "Vehicle": d["label"],
                    "Confidence": f"{d['confidence']:.1%}",
                    "Location": f"({d['bbox'][0]},{d['bbox'][1]})→({d['bbox'][2]},{d['bbox'][3]})"
                } for d in detections])
                st.dataframe(df, use_container_width=True, hide_index=True, height=320)

            st.markdown("<br>", unsafe_allow_html=True)
            buf = io.BytesIO()
            annotated.save(buf, format="PNG")
            st.download_button("Download Annotated Image", buf.getvalue(),
                               "detected_vehicles.png", "image/png")
        else:
            st.markdown("""
            <div class="card-accent" style="text-align:center;padding:2rem;">
                <div style="color:#D4AF37;font-family:Montserrat,sans-serif;
                     font-weight:700;letter-spacing:0.08em;">NO VEHICLES DETECTED</div>
                <div style="color:#555;font-size:0.85rem;margin-top:0.5rem;">
                    Lower the confidence threshold in the sidebar and try again.</div>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("<h2 style='margin-bottom:1.5rem;'>HOW IT WORKS</h2>", unsafe_allow_html=True)
    steps = [
        ("01", "INPUT",          "🖼️",  "#1D6FF0", "Upload any traffic or dash-cam image in JPG or PNG format"),
        ("02", "PREPROCESSING",  "⚙️",  "#00C896", "Image is resized and optimised for the YOLOv8 model"),
        ("03", "DETECTION",      "🔍",  "#D4AF37", "YOLOv8s scans every region of the image in a single pass"),
        ("04", "CLASSIFICATION", "🏷️",  "#FFA502", "Each object is assigned a vehicle class and confidence score"),
        ("05", "ANNOTATION",     "✏️",  "#FF4757", "Coloured bounding boxes and labels drawn on original image"),
        ("06", "ANALYTICS",      "📊",  "#A855F7", "Full vehicle count breakdown and statistics generated"),
    ]
    for num, title, icon, color, desc in steps:
        st.markdown(f"""
        <div class="card" style="display:flex;align-items:center;gap:1.4rem;
             margin:0.3rem 0;border-left:3px solid {color};">
            <div style="font-family:'Montserrat',sans-serif;font-weight:800;
                 font-size:1.3rem;color:{color};min-width:36px;opacity:0.6;">{num}</div>
            <div style="font-size:1.3rem;min-width:28px;">{icon}</div>
            <div>
                <div style="font-family:'Montserrat',sans-serif;font-weight:700;
                     color:{color};font-size:0.8rem;letter-spacing:0.1em;">{title}</div>
                <div style="color:#888;font-size:0.84rem;margin-top:3px;">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="card-accent">
        <div style="font-family:'Montserrat',sans-serif;font-weight:800;color:#D4AF37;
             font-size:0.78rem;letter-spacing:0.12em;margin-bottom:12px;">
             🤖 YOLOV8S — THE MODEL</div>
        <p style="color:#888;line-height:1.9;font-size:0.87rem;margin:0;">
        YOLO stands for <span style="color:#1D6FF0;font-weight:600;">You Only Look Once</span>.
        YOLOv8 processes the entire frame in a single pass — outputting bounding boxes,
        class labels, and confidence scores simultaneously. The
        <span style="color:#00C896;font-weight:600;">YOLOv8s (Small)</span> model achieves
        <span style="color:#D4AF37;font-weight:600;">90%+ mAP</span> across all vehicle classes
        at <span style="color:#FFA502;font-weight:600;">30+ FPS</span> on GPU hardware.
        <br/><br/>
        <span style="color:#FF4757;font-size:0.8rem;">
        Note: SUVs are classified as Truck/SUV — this is standard COCO dataset behaviour
        where large passenger vehicles share the truck category.</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.markdown("<h2 style='margin-bottom:1.5rem;'>PROJECT INFO</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown("""
        <div class="card" style="border-left:3px solid #1D6FF0;">
            <div style="font-family:'Montserrat',sans-serif;font-weight:800;color:#1D6FF0;
                 font-size:0.75rem;letter-spacing:0.12em;margin-bottom:14px;">BACKGROUND</div>
            <div style="color:#888;font-size:0.85rem;line-height:2.3;">
                <span style="color:#EAD7A1;">Venue</span>
                &nbsp;—&nbsp; Samsung R&D Innovation Campus, Kolkata<br/>
                <span style="color:#EAD7A1;">Programme</span>
                &nbsp;—&nbsp; AI & API Integration<br/>
                <span style="color:#EAD7A1;">Duration</span>
                &nbsp;—&nbsp; Sept – Nov 2025<br/>
                <span style="color:#EAD7A1;">Team</span>
                &nbsp;—&nbsp; 4 members
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card" style="border-left:3px solid #00C896;">
            <div style="font-family:'Montserrat',sans-serif;font-weight:800;color:#00C896;
                 font-size:0.75rem;letter-spacing:0.12em;margin-bottom:14px;">PERFORMANCE</div>
            <div style="color:#888;font-size:0.85rem;line-height:2.3;">
                <span style="color:#EAD7A1;">Model</span>
                &nbsp;—&nbsp; YOLOv8s (Ultralytics)<br/>
                <span style="color:#EAD7A1;">Accuracy</span>
                &nbsp;—&nbsp; 90%+ mAP across all classes<br/>
                <span style="color:#EAD7A1;">Speed</span>
                &nbsp;—&nbsp; 30+ FPS on GPU hardware<br/>
                <span style="color:#EAD7A1;">Classes</span>
                &nbsp;—&nbsp; 5 vehicle types
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><h3 style='margin-bottom:1rem;'>TEAM</h3>", unsafe_allow_html=True)
    team = [
        ("Sourav Kumar Burman", "AI Pipeline & API Integration", "#D4AF37"),
        ("Sathi Santra",        "Model Training & Evaluation",   "#1D6FF0"),
        ("Shovan Mondal",       "Video Processing",              "#00C896"),
        ("Tulika Adak",         "Visualisation & Reporting",     "#FF4757"),
    ]
    cols = st.columns(4, gap="small")
    for col, (name, role, color) in zip(cols, team):
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center;padding:1rem 0.8rem;border-top:2px solid {color};">
                <div style="width:42px;height:42px;border-radius:50%;
                     background:{color}22;border:2px solid {color}66;
                     display:flex;align-items:center;justify-content:center;
                     margin:0 auto 10px;font-family:Montserrat,sans-serif;
                     font-weight:800;font-size:1rem;color:{color};">
                     {name[0]}</div>
                <div style="font-family:'Montserrat',sans-serif;font-weight:700;
                     color:#EAD7A1;font-size:0.77rem;line-height:1.4;">{name}</div>
                <div style="color:#555;font-size:0.71rem;margin-top:5px;">{role}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""<br>
    <div class="card">
        <div style="font-family:'Montserrat',sans-serif;font-weight:800;color:#D4AF37;
             font-size:0.72rem;letter-spacing:0.15em;margin-bottom:14px;">
             REAL-WORLD APPLICATIONS</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.8rem;">
            <div style="color:#888;font-size:0.83rem;">
                <span style="color:#1D6FF0;">🚦</span>&nbsp; Traffic monitoring & flow analysis</div>
            <div style="color:#888;font-size:0.83rem;">
                <span style="color:#FF4757;">🚔</span>&nbsp; Automated violation detection</div>
            <div style="color:#888;font-size:0.83rem;">
                <span style="color:#FFA502;">🚚</span>&nbsp; Fleet & logistics management</div>
            <div style="color:#888;font-size:0.83rem;">
                <span style="color:#00C896;">🤖</span>&nbsp; Autonomous vehicle systems</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:10px 0;font-family:'Poppins',sans-serif;
     font-size:0.73rem;color:#333;letter-spacing:0.06em;">
    Built by <span style="color:#D4AF37;font-weight:600;">Sourav Burman</span>
    &nbsp;·&nbsp; Samsung R&D Innovation Campus
    &nbsp;·&nbsp;
    <a href="https://github.com/thesouravburman/vehicle-classifier"
       style="color:#1D6FF0;text-decoration:none;">GitHub Repo</a>
</div>
""", unsafe_allow_html=True)
