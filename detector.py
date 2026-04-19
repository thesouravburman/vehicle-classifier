from ultralytics import YOLO
import cv2
import numpy as np

VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
    1: "Bicycle"
}

COLORS = {
    "Car":        (30,  144, 255),
    "Motorcycle": (0,   255, 127),
    "Bus":        (220,  20,  60),
    "Truck":      (148,   0, 211),
    "Bicycle":    (255, 165,   0),
}

class VehicleDetector:
    def __init__(self, confidence: float = 0.4):
        self.model = YOLO("yolov8n.pt")
        self.model.conf = confidence

    def detect(self, image: np.ndarray):
        results = self.model(image, verbose=False)[0]
        detections = []
        annotated = image.copy()

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue
            label = VEHICLE_CLASSES[cls_id]
            conf  = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = COLORS[label]

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {conf:.0%}"
            tw = len(text) * 11
            cv2.rectangle(annotated, (x1, y1 - 26), (x1 + tw, y1), color, -1)
            cv2.putText(annotated, text, (x1 + 4, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2)
            })

        return annotated, detections
