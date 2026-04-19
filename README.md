# 🚗 Vehicle Classifier — Real-Time Traffic Detection

> A computer vision pipeline that processes live traffic footage and classifies vehicles into 5 categories.
> Built during AI & API Integration training at **Samsung R&D Institute, Kolkata**.

---

## 🎯 What It Does

Detects vehicles in images/video frames and classifies them into:
- 🏍️ Two-Wheeler (motorbike/scooter)
- 🚗 Sedan
- 🚙 SUV
- 🚌 Bus
- 🚛 Truck

Results are sent via REST API to a reporting dashboard in real time.

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Detection Model | YOLOv8 / Pre-trained CNN |
| API Layer | Python Requests / REST |
| Data Processing | NumPy, OpenCV |
| Language | Python 3.10+ |

## 📁 Project Structure

## ⚡ Quick Start

```bash
git clone https://github.com/thesouravburman/vehicle-classifier
cd vehicle-classifier
pip install -r requirements.txt
python main.py --input sample_traffic.jpg
```

## 🔗 Context

Developed as the final deliverable of the AI & API Integration programme at Samsung R&D Institute, Kolkata (Sept–Nov 2025).

## 📬 Contact

**Sourav Burman** — thesouravburman@gmail.com · [LinkedIn](https://linkedin.com/in/sourav-burman)
