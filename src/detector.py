"""
detector.py
Model A: Conventional Object Detection Baseline Feature Extractor.
Uses YOLOv8n to extract frame-level detection features (counts, areas, coordinates, proximities).
Strictly excludes tracking-derived velocity, displacements, and trajectory features.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

# COCO Target Classes
# 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
TARGET_CLASSES = {
    0: "pedestrian",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}
PEDESTRIAN_CLASS_IDS = {0}

class ObjectDetector:
    def __init__(self, model_path="models/yolov8n.pt", conf_thresh=0.25, device="cpu"):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.device = device
        self.target_ids = set(TARGET_CLASSES.keys())

    def detect_frame(self, frame):
        """
        Run inference on a single frame. Returns list of detection dicts.
        """
        h_img, w_img = frame.shape[:2]
        img_area = float(w_img * h_img)

        results = self.model(
            frame,
            conf=self.conf_thresh,
            device=self.device,
            verbose=False
        )[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id not in self.target_ids:
                continue

            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            area = w * h
            norm_area = area / img_area

            cx = (x1 + x2) / 2.0 / w_img
            cy = (y1 + y2) / 2.0 / h_img
            bottom_y = y2 / h_img

            detections.append({
                "class_id": cls_id,
                "class_name": TARGET_CLASSES[cls_id],
                "conf": conf,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "norm_area": norm_area,
                "cx": cx,
                "cy": cy,
                "bottom_y": bottom_y,
                "is_vehicle": 1 if cls_id in VEHICLE_CLASS_IDS else 0,
                "is_pedestrian": 1 if cls_id in PEDESTRIAN_CLASS_IDS else 0,
                "in_center_corridor": 1 if 0.35 <= cx <= 0.65 else 0
            })

        return detections

    def extract_frame_features(self, frame):
        """
        Extract frame-level detection feature vector for Model A.
        No tracking or velocity features.
        """
        dets = self.detect_frame(frame)
        if len(dets) == 0:
            return {
                "num_vehicles": 0,
                "num_pedestrians": 0,
                "total_objects": 0,
                "max_bbox_area": 0.0,
                "mean_bbox_area": 0.0,
                "sum_bbox_area": 0.0,
                "max_conf": 0.0,
                "mean_conf": 0.0,
                "center_corridor_count": 0,
                "closest_bottom_y": 0.0,
                "closest_bbox_area": 0.0,
                "closest_is_pedestrian": 0,
                "closest_dist_to_center": 1.0,
                "has_large_object": 0
            }

        num_veh = sum(d["is_vehicle"] for d in dets)
        num_ped = sum(d["is_pedestrian"] for d in dets)
        areas = [d["norm_area"] for d in dets]
        confs = [d["conf"] for d in dets]
        center_cnt = sum(d["in_center_corridor"] for d in dets)

        # Closest object by bottom_y
        sorted_by_y = sorted(dets, key=lambda d: d["bottom_y"], reverse=True)
        closest = sorted_by_y[0]

        max_area = max(areas)
        mean_area = float(np.mean(areas))
        sum_area = float(np.sum(areas))

        dist_to_center = abs(closest["cx"] - 0.5)

        return {
            "num_vehicles": num_veh,
            "num_pedestrians": num_ped,
            "total_objects": len(dets),
            "max_bbox_area": max_area,
            "mean_bbox_area": mean_area,
            "sum_bbox_area": sum_area,
            "max_conf": max(confs),
            "mean_conf": float(np.mean(confs)),
            "center_corridor_count": center_cnt,
            "closest_bottom_y": closest["bottom_y"],
            "closest_bbox_area": closest["norm_area"],
            "closest_is_pedestrian": closest["is_pedestrian"],
            "closest_dist_to_center": dist_to_center,
            "has_large_object": 1 if max_area > 0.12 else 0
        }

    def extract_window_features(self, video_path, start_frame, end_frame, step_frames=6):
        """
        Processes sampled frames across a window and aggregates frame-level features.
        Computes max, mean, and end-of-window statistics.
        """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_feats = []
        curr = start_frame
        while curr <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            if (curr - start_frame) % step_frames == 0:
                f_feat = self.extract_frame_features(frame)
                frame_feats.append(f_feat)
            curr += 1
        cap.release()

        if len(frame_feats) == 0:
            frame_feats = [self.extract_frame_features(np.zeros((720, 1280, 3), dtype=np.uint8))]

        df_f = pd.DataFrame(frame_feats)

        # Aggregate across window
        aggregated = {}
        for col in df_f.columns:
            aggregated[f"{col}_mean"] = float(df_f[col].mean())
            aggregated[f"{col}_max"] = float(df_f[col].max())
            aggregated[f"{col}_last"] = float(df_f[col].iloc[-1])

        return aggregated

if __name__ == "__main__":
    print("Testing ObjectDetector...")
    detector = ObjectDetector()
    sample_frame = cv2.imread("test_street.jpg")
    features = detector.extract_frame_features(sample_frame)
    print("\nSample Frame Detection Features (Model A):")
    for k, v in features.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
