from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw

VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck / SUV",
    1: "Bicycle"
}

COLORS = {
    "Car":         "#1D6FF0",
    "Motorcycle":  "#00C896",
    "Bus":         "#FF4757",
    "Truck / SUV": "#FFA502",
    "Bicycle":     "#A855F7",
}

HEX_TO_RGB = {
    "#1D6FF0": (29,  111, 240),
    "#00C896": (0,   200, 150),
    "#FF4757": (255,  71,  87),
    "#FFA502": (255, 165,   2),
    "#A855F7": (168,  85, 247),
}

class VehicleDetector:
    def __init__(self, confidence: float = 0.25):
        self.model = YOLO("yolov8s.pt")
        self.confidence = confidence

    def detect(self, image: Image.Image):
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

            # Thick bounding box with inner glow effect
            draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=3)
            draw.rectangle([x1+1, y1+1, x2-1, y2-1],
                           outline=(color_rgb[0]//3, color_rgb[1]//3, color_rgb[2]//3),
                           width=1)

            # Label
            text = f" {label}  {conf:.0%} "
            text_w = len(text) * 9
            text_h = 24
            draw.rectangle([x1, y1 - text_h, x1 + text_w, y1],
                           fill=color_rgb)
            draw.text((x1 + 4, y1 - text_h + 4), text, fill=(10, 10, 10))

            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2)
            })

        detections.sort(key=lambda x: x["confidence"], reverse=True)
        return annotated, detections
