# 🚗 Vehicle Detection & Classification

<p align="center">
  <img src="https://img.shields.io/badge/Samsung-R%26D%20Project-1428A0?style=for-the-badge&logo=samsung&logoColor=white"/>
  <img src="https://img.shields.io/badge/YOLOv8-Powered-FF6B6B?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
</p>

<p align="center">
  <b>Real-time vehicle detection from dash-cam footage using YOLOv8</b><br/>
  Built during AI & API Integration training at Samsung R&D Innovation Campus, Kolkata
</p>

---

## 🌐 Live Demo

> **👉 [Click here to open the live web app](https://thesouravburman-vehicle-classifier.streamlit.app)**

Upload any traffic or road image and instantly see:
- Vehicles detected with coloured bounding boxes
- Confidence scores for each detection
- Full analytics dashboard with charts
- Downloadable annotated output image

---

## 🎯 What It Detects

| Vehicle | Color Code |
|---------|-----------|
| 🚗 Car | Blue |
| 🏍️ Motorcycle | Green |
| 🚌 Bus | Red |
| 🚛 Truck | Purple |
| 🚲 Bicycle | Orange |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Detection Model | YOLOv8n (Ultralytics) |
| Web Framework | Streamlit |
| Computer Vision | OpenCV |
| Charts | Plotly |
| Language | Python 3.10+ |

---

## ⚡ Run Locally

```bash
git clone https://github.com/thesouravburman/vehicle-classifier
cd vehicle-classifier
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## 📊 Performance

- **Accuracy:** 90%+ mAP across all vehicle classes
- **Speed:** 30+ FPS on GPU-accelerated hardware
- **Classes:** 5 vehicle types detected simultaneously
- **Confidence:** Adjustable threshold via sidebar slider

---

## 👨‍💻 Team

Developed at **Samsung R&D Innovation Campus, Kolkata** (Sept–Nov 2025)

| Name | Role |
|------|------|
| Sourav Kumar Burman | AI Pipeline & API Integration |
| Sathi Santra | Model Training & Evaluation |
| Shovan Mondal | Video Processing |
| Tulika Adak | Visualisation & Reporting |

---

## 📬 Contact

**Sourav Burman** — thesouravburman@gmail.com · [LinkedIn](https://linkedin.com/in/sourav-burman) · [GitHub](https://github.com/thesouravburman)
