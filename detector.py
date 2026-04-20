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

HEX_TO_RGB = {
    "#1D4ED8": (29, 78, 216),
    "#D4AF37": (212, 175, 55),
    "#DC143C": (220, 20, 60),
    "#EAD7A1": (234, 215, 161),
    "#FAFAFA": (250, 250, 250),
}

class VehicleDetector:
    def __init__(self, confidence: float = 0.25):
        # yolov8s = small model, much more accurate than nano
        self.model = YOLO("yolov8s.pt")
        self.confidence = confidence

    def detect(self, image: Image.Image):
        # Resize large images for faster processing while keeping quality
        max_size = 1280
        w, h = image.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        img_array = np.array(image)
        results = self.model(img_array, conf=self.confidence, verbose=False)[0]

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
            color_hex = COLORS[label]
            color_rgb = HEX_TO_RGB[color_hex]

            # Thick bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=4)

            # Label background
            text = f"{label}  {conf:.0%}"
            text_w = len(text) * 10
            text_h = 28
            draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill=color_rgb)

            # Label text — black on colored background
            draw.text((x1 + 5, y1 - text_h + 5), text, fill=(10, 10, 10))

            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2)
            })

        # Sort by confidence descending
        detections.sort(key=lambda x: x["confidence"], reverse=True)
        return annotated, detections
