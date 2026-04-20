from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont

VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
    1: "Bicycle"
}

COLORS = {
    "Car":        "#1D4ED8",
    "Motorcycle": "#D4AF37",
    "Bus":        "#DC143C",
    "Truck":      "#EAD7A1",
    "Bicycle":    "#FAFAFA",
}

class VehicleDetector:
    def __init__(self, confidence: float = 0.4):
        self.model = YOLO("yolov8n.pt")
        self.model.conf = confidence

    def detect(self, image: Image.Image):
        img_array = np.array(image)
        results = self.model(img_array, verbose=False)[0]
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue
            label = VEHICLE_CLASSES[cls_id]
            conf  = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = COLORS[label]

            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            text = f"{label} {conf:.0%}"
            draw.rectangle([x1, y1 - 24, x1 + len(text) * 9, y1], fill=color)
            draw.text((x1 + 4, y1 - 20), text, fill="#0C0C0C")

            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2)
            })

        return annotated, detections
