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
    background-color: #0C0C0C !important;
    border-right: 1px solid #D4AF37;
}
div[data-testid="metric-container"] {
    background-color: #1A1A1A !important;
    border: 1px solid #D4AF37 !important;
    border-radius: 10px !important;
    padding: 1.2rem 1rem !important;
}
[data-testid="metric-container"] label {
    color: #EAD7A1 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
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
    padding: 0.5rem 1.5rem !important;
}
.stButton > button:hover { background-color: #EAD7A1 !important; }
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 1px solid #2a2a2a !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Montserrat', sans-serif !important;
    color: #888 !important;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.82rem !important;
    padding: 0.6rem 1.5rem !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #D4AF37 !important;
    border-bottom: 2px solid #D4AF37 !important;
    background: transparent !important;
}
.stSlider label {
    color: #EAD7A1 !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 0.85rem !important;
}
hr { border-color: #D4AF37 !important; opacity: 0.2; }
.card {
    background: #141414;
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin: 0.4rem 0;
}
.card-gold {
    background: #141414;
    border: 1px solid #D4AF37;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin: 0.4rem 0;
}
.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'Montserrat', sans-serif;
    font-weight: 700;
    margin: 3px 2px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.section-label {
    font-family: 'Montserrat', sans-serif;
    font-weight: 800;
    color: #D4AF37;
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding-bottom: 8px;
    border-bottom: 1px solid #D4AF37;
    margin-bottom: 1rem;
}
.img-label {
    font-family: 'Montserrat', sans-serif;
    font-weight: 700;
    color: #EAD7A1;
    font-size: 0.78rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.upload-zone {
    border: 2px dashed #D4AF37;
    border-radius: 12px;
    padding: 3rem 2rem;
    text-align: center;
    background: #0F0F0F;
    margin: 1rem 0;
    transition: all 0.2s;
}
</style>
""", unsafe_allow_html=True)

# ── Brand Header ───────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;
     padding:18px 0 22px 0;border-bottom:1px solid #D4AF37;margin-bottom:32px;">
    <div>
        <div style="font-family:'Montserrat',sans-serif;font-weight:800;
             font-size:1.3rem;color:#D4AF37;letter-spacing:0.1em;">SOURAV BURMAN</div>
        <div style="font-family:'Poppins',sans-serif;font-size:0.7rem;
             color:#EAD7A1;letter-spacing:0.18em;margin-top:3px;opacity:0.8;">
             CS ENGINEER · BUILDER</div>
    </div>
    <div style="display:flex;align-items:center;gap:12px;">
        <div style="font-family:'Montserrat',sans-serif;font-size:0.65rem;
             color:#D4AF37;opacity:0.4;letter-spacing:0.12em;">⬡ VEHICLE CLASSIFIER</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Page Title ─────────────────────────────────────
st.markdown("""
<div style="margin-bottom:2rem;">
    <h1 style="font-size:2.2rem;margin-bottom:0.3rem;line-height:1.1;">
        VEHICLE DETECTION<br>& CLASSIFICATION
    </h1>
    <p style="color:#888;font-size:0.9rem;margin:0;letter-spacing:0.02em;">
        YOLOv8 · Samsung R&D Innovation Campus, Kolkata · Real-time dash-cam analysis
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-label">Settings</div>', unsafe_allow_html=True)
    confidence = st.slider("Detection Confidence", 0.1, 0.9, 0.25, 0.05,
        help="Lower = detect more vehicles. Higher = only very certain detections.")
    st.markdown(f"""
    <div style="font-size:0.75rem;color:#888;margin-top:-0.5rem;margin-bottom:1rem;">
        Current: {confidence:.0%} — {"High sensitivity" if confidence < 0.35 else "Balanced" if confidence < 0.6 else "High precision"}
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:1.5rem;">Vehicle Classes</div>', unsafe_allow_html=True)
    classes = [("Car","#1D4ED8"),("Motorcycle","#D4AF37"),
               ("Bus","#DC143C"),("Truck","#EAD7A1"),("Bicycle","#FAFAFA")]
    for name, color in classes:
        st.markdown(f'''<span class="badge"
            style="background:{color}18;color:{color};border:1px solid {color}44;">
            {name}</span>''', unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:1.5rem;">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.82rem;color:#EAD7A1;line-height:2.1;">
        <span style="color:#D4AF37;font-weight:700;">Sourav Burman</span><br/>
        Samsung R&D Innovation Campus<br/>
        Brainware University · CSE '27
    </div>
    <div style="margin-top:0.8rem;">
        <a href="https://github.com/thesouravburman"
           style="color:#D4AF37;font-size:0.8rem;font-family:Montserrat,sans-serif;
           font-weight:700;letter-spacing:0.05em;text-decoration:none;">
           GITHUB</a>
        <span style="color:#444;margin:0 8px;">·</span>
        <a href="https://linkedin.com/in/sourav-burman"
           style="color:#D4AF37;font-size:0.8rem;font-family:Montserrat,sans-serif;
           font-weight:700;letter-spacing:0.05em;text-decoration:none;">
           LINKEDIN</a>
    </div>
    """, unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["  DETECTION  ", "  HOW IT WORKS  ", "  PROJECT INFO  "])

with tab1:
    uploaded = st.file_uploader(
        "Upload a traffic or dash-cam image",
        type=["jpg", "jpeg", "png"],
        help="Supports JPG and PNG up to 200MB"
    )

    if not uploaded:
        st.markdown("""
        <div class="upload-zone">
            <div style="font-size:2.5rem;margin-bottom:0.8rem;">🚗</div>
            <div style="font-family:'Montserrat',sans-serif;font-weight:700;
                 color:#D4AF37;font-size:0.95rem;letter-spacing:0.08em;">
                DROP A TRAFFIC IMAGE HERE</div>
            <div style="color:#555;font-size:0.82rem;margin-top:0.4rem;">
                JPG or PNG · Up to 200MB · Any traffic or road scene</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        image = Image.open(uploaded).convert("RGB")

        with st.spinner("Running YOLOv8 detection..."):
            detector = VehicleDetector(confidence=confidence)
            annotated, detections = detector.detect(image)

        # Images side by side
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            st.markdown('<div class="img-label">Original Image</div>', unsafe_allow_html=True)
            st.image(image, use_column_width=True)
        with col2:
            st.markdown('<div class="img-label">Detected Vehicles</div>', unsafe_allow_html=True)
            st.image(annotated, use_column_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if detections:
            counts = {}
            for d in detections:
                counts[d["label"]] = counts.get(d["label"], 0) + 1
            avg_conf = sum(d["confidence"] for d in detections) / len(detections)
            top_conf = detections[0]["confidence"] if detections else 0

            # Metrics row
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Total Detected", len(detections))
            with c2: st.metric("Avg Confidence", f"{avg_conf:.0%}")
            with c3: st.metric("Vehicle Types", len(counts))
            with c4: st.metric("Top Confidence", f"{top_conf:.0%}")

            st.markdown("<br>", unsafe_allow_html=True)

            col_chart, col_table = st.columns([1.1, 0.9], gap="medium")
            color_map = {"Car":"#1D4ED8","Motorcycle":"#D4AF37",
                         "Bus":"#DC143C","Truck":"#EAD7A1","Bicycle":"#FAFAFA"}

            with col_chart:
                fig = go.Figure(go.Bar(
                    x=list(counts.keys()),
                    y=list(counts.values()),
                    marker_color=[color_map.get(k,"#D4AF37") for k in counts.keys()],
                    marker_line_color="#0C0C0C",
                    marker_line_width=1.5,
                    text=list(counts.values()),
                    textposition="outside",
                    textfont=dict(color="#EAD7A1", family="Montserrat", size=13)
                ))
                fig.update_layout(
                    title=dict(text="VEHICLES BY TYPE",
                               font=dict(family="Montserrat",color="#D4AF37",size=12),
                               x=0),
                    paper_bgcolor="#0C0C0C",
                    plot_bgcolor="#141414",
                    font=dict(color="#FAFAFA", family="Poppins"),
                    showlegend=False,
                    margin=dict(t=40, b=10, l=10, r=10),
                    xaxis=dict(gridcolor="#1a1a1a", tickfont=dict(
                        family="Montserrat", color="#EAD7A1", size=11)),
                    yaxis=dict(gridcolor="#1a1a1a", tickfont=dict(color="#555")),
                    bargap=0.35,
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_table:
                st.markdown('<div class="img-label" style="margin-bottom:8px;">Detection Details</div>', unsafe_allow_html=True)
                df = pd.DataFrame([{
                    "Vehicle": d["label"],
                    "Confidence": f"{d['confidence']:.1%}",
                    "Location": f"({d['bbox'][0]},{d['bbox'][1]})→({d['bbox'][2]},{d['bbox'][3]})"
                } for d in detections])
                st.dataframe(df, use_container_width=True, hide_index=True, height=320)

            st.markdown("<br>", unsafe_allow_html=True)
            buf = io.BytesIO()
            annotated.save(buf, format="PNG")
            st.download_button(
                "Download Annotated Image",
                buf.getvalue(),
                "detected_vehicles.png",
                "image/png",
                use_container_width=False
            )
        else:
            st.markdown("""
            <div class="card-gold" style="text-align:center;padding:2rem;">
                <div style="color:#D4AF37;font-family:Montserrat,sans-serif;
                     font-weight:700;letter-spacing:0.08em;">NO VEHICLES DETECTED</div>
                <div style="color:#888;font-size:0.85rem;margin-top:0.5rem;">
                    Lower the confidence threshold in the sidebar and try again.</div>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("<h2 style='margin-bottom:1.5rem;'>HOW IT WORKS</h2>", unsafe_allow_html=True)
    steps = [
        ("01", "INPUT",          "🖼", "Upload any traffic or dash-cam image in JPG or PNG format"),
        ("02", "PREPROCESSING",  "⚙️", "Image is resized and converted to a numpy array for the model"),
        ("03", "DETECTION",      "🔍", "YOLOv8 scans every region of the image in a single forward pass"),
        ("04", "CLASSIFICATION", "🏷", "Each detected object is assigned a vehicle class and confidence score"),
        ("05", "ANNOTATION",     "✏️", "Coloured bounding boxes and labels are drawn on the original image"),
        ("06", "ANALYTICS",      "📊", "Full breakdown of vehicle counts, types, and statistics is generated"),
    ]
    for num, title, icon, desc in steps:
        st.markdown(f"""
        <div class="card" style="display:flex;align-items:center;gap:1.4rem;margin:0.4rem 0;">
            <div style="font-family:'Montserrat',sans-serif;font-weight:800;
                 font-size:1.5rem;color:#D4AF37;min-width:40px;opacity:0.7;">{num}</div>
            <div style="font-size:1.4rem;min-width:28px;">{icon}</div>
            <div>
                <div style="font-family:'Montserrat',sans-serif;font-weight:700;
                     color:#D4AF37;font-size:0.82rem;letter-spacing:0.1em;">{title}</div>
                <div style="color:#AAA;font-size:0.85rem;margin-top:3px;">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="card-gold">
        <div style="font-family:'Montserrat',sans-serif;font-weight:800;color:#D4AF37;
             font-size:0.82rem;letter-spacing:0.1em;margin-bottom:12px;">YOLOV8 — THE MODEL</div>
        <p style="color:#AAA;line-height:1.9;font-size:0.88rem;margin:0;">
        YOLO stands for <span style="color:#D4AF37;font-weight:600;">You Only Look Once</span>.
        Unlike traditional methods that scan an image multiple times,
        YOLOv8 processes the entire frame in a single pass — outputting bounding boxes,
        class labels, and confidence scores simultaneously. The <span style="color:#D4AF37;">
        YOLOv8s (Small)</span> model used here achieves
        <span style="color:#D4AF37;font-weight:600;">90%+ mAP</span> across all vehicle classes
        while running at <span style="color:#D4AF37;font-weight:600;">30+ FPS</span> on GPU hardware.
        </p>
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.markdown("<h2 style='margin-bottom:1.5rem;'>PROJECT INFO</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown("""
        <div class="card-gold">
            <div class="section-label" style="margin-bottom:12px;border-bottom:none;padding-bottom:0;">
                BACKGROUND</div>
            <div style="color:#AAA;font-size:0.85rem;line-height:2.2;">
                <span style="color:#EAD7A1;">Venue</span> — Samsung R&D Innovation Campus, Kolkata<br/>
                <span style="color:#EAD7A1;">Programme</span> — AI & API Integration<br/>
                <span style="color:#EAD7A1;">Duration</span> — Sept – Nov 2025<br/>
                <span style="color:#EAD7A1;">Team</span> — 4 members
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card-gold">
            <div class="section-label" style="margin-bottom:12px;border-bottom:none;padding-bottom:0;">
                PERFORMANCE</div>
            <div style="color:#AAA;font-size:0.85rem;line-height:2.2;">
                <span style="color:#EAD7A1;">Model</span> — YOLOv8s (Ultralytics Small)<br/>
                <span style="color:#EAD7A1;">Accuracy</span> — 90%+ mAP across classes<br/>
                <span style="color:#EAD7A1;">Speed</span> — 30+ FPS on GPU<br/>
                <span style="color:#EAD7A1;">Classes</span> — 5 vehicle types
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><h3 style='margin-bottom:1rem;'>TEAM</h3>", unsafe_allow_html=True)
    team = [
        ("Sourav Kumar Burman", "AI Pipeline & API Integration"),
        ("Sathi Santra",        "Model Training & Evaluation"),
        ("Shovan Mondal",       "Video Processing"),
        ("Tulika Adak",         "Visualisation & Reporting"),
    ]
    cols = st.columns(4, gap="small")
    for col, (name, role) in zip(cols, team):
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center;padding:1rem 0.8rem;">
                <div style="width:40px;height:40px;border-radius:50%;
                     background:#D4AF3722;border:1px solid #D4AF37;
                     display:flex;align-items:center;justify-content:center;
                     margin:0 auto 10px;font-family:Montserrat,sans-serif;
                     font-weight:800;font-size:0.85rem;color:#D4AF37;">
                     {name[0]}</div>
                <div style="font-family:'Montserrat',sans-serif;font-weight:700;
                     color:#EAD7A1;font-size:0.78rem;">{name}</div>
                <div style="color:#666;font-size:0.72rem;margin-top:4px;">{role}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""<br>
    <div class="card" style="padding:1.2rem 1.5rem;">
        <div class="section-label" style="border-bottom:none;margin-bottom:10px;">
            REAL-WORLD APPLICATIONS</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.8rem;margin-top:0.5rem;">
            <div style="color:#AAA;font-size:0.83rem;">
                <span style="color:#D4AF37;">🚦</span> Traffic monitoring & flow analysis</div>
            <div style="color:#AAA;font-size:0.83rem;">
                <span style="color:#D4AF37;">🚔</span> Automated violation detection</div>
            <div style="color:#AAA;font-size:0.83rem;">
                <span style="color:#D4AF37;">🚚</span> Fleet & logistics management</div>
            <div style="color:#AAA;font-size:0.83rem;">
                <span style="color:#D4AF37;">🤖</span> Autonomous vehicle systems</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:10px 0;font-family:'Poppins',sans-serif;
     font-size:0.75rem;color:#444;letter-spacing:0.06em;">
    Built by <span style="color:#D4AF37;font-weight:600;">Sourav Burman</span>
    · Samsung R&D Innovation Campus ·
    <a href="https://github.com/thesouravburman/vehicle-classifier"
       style="color:#D4AF37;text-decoration:none;">GitHub Repo</a>
</div>
""", unsafe_allow_html=True)
