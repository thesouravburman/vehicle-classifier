"""
Vehicle Classifier — Main Entry Point
Samsung R&D Institute, Kolkata — AI & API Integration Programme
"""

import argparse
from classifier.detector import VehicleDetector
from api.client import DashboardClient

def parse_args():
    parser = argparse.ArgumentParser(description="Vehicle Classifier")
    parser.add_argument("--input", type=str, required=True, help="Path to image or video file")
    parser.add_argument("--confidence", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--output", type=str, default=None, help="Optional: path to save output")
    return parser.parse_args()

def main():
    args = parse_args()
    detector = VehicleDetector(confidence_threshold=args.confidence)
    client = DashboardClient()
    print(f"[INFO] Processing input: {args.input}")
    results = detector.process(args.input)
    print(f"[INFO] Detected {len(results)} vehicles:")
    for r in results:
        print(f"  → {r['category']:15s}  confidence: {r['confidence']:.2f}")
    response = client.send_results(results)
    if response.get("status") == "success":
        print("[INFO] Results successfully sent to dashboard.")
    else:
        print("[WARN] Dashboard API did not confirm receipt.")

if __name__ == "__main__":
    main()
