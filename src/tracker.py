"""
tracker.py
Model B: Object Detection + Multi-Object Tracking (ByteTrack).
Maintains track IDs, computes frame-to-frame displacements, image-space velocities,
bounding box dimension expansion rates, and relative pairwise motion between road users.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ultralytics import YOLO
from src.detector import TARGET_CLASSES, VEHICLE_CLASS_IDS, PEDESTRIAN_CLASS_IDS

class MultiObjectTracker:
    def __init__(self, model_path="models/yolov8n.pt", conf_thresh=0.25, device="cpu"):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.device = device
        self.target_ids = set(TARGET_CLASSES.keys())

    def process_window(self, video_path, start_frame, end_frame, step_frames=6):
        """
        Track objects sequentially across the temporal window.
        Returns detailed track histories and engineered Model B tracking features.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        effective_dt = step_frames / fps

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Track history dictionary: track_id -> list of frame states
        # state: {frame_idx, time_sec, cx, cy, w, h, area, bottom_y, cls_id, conf}
        track_history = defaultdict(list)
        frame_tracks = [] # per-frame snapshot for pair analysis

        curr_frame = start_frame
        frame_count = 0

        while curr_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if (curr_frame - start_frame) % step_frames == 0:
                h_img, w_img = frame.shape[:2]
                img_area = float(w_img * h_img)

                # Run tracker with persistence
                results = self.model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=self.conf_thresh,
                    device=self.device,
                    verbose=False
                )[0]

                current_frame_objs = []
                if results.boxes is not None and results.boxes.id is not None:
                    boxes = results.boxes.xyxy.cpu().numpy()
                    track_ids = results.boxes.id.int().cpu().numpy()
                    cls_ids = results.boxes.cls.int().cpu().numpy()
                    confs = results.boxes.conf.cpu().numpy()

                    for box, tid, cid, conf in zip(boxes, track_ids, cls_ids, confs):
                        if cid not in self.target_ids:
                            continue

                        x1, y1, x2, y2 = box
                        w = max(1.0, x2 - x1)
                        h = max(1.0, y2 - y1)
                        norm_area = (w * h) / img_area
                        cx = (x1 + x2) / 2.0 / w_img
                        cy = (y1 + y2) / 2.0 / h_img
                        bottom_y = y2 / h_img

                        state = {
                            "frame_idx": curr_frame,
                            "time_sec": curr_frame / fps,
                            "track_id": int(tid),
                            "cls_id": int(cid),
                            "conf": float(conf),
                            "cx": cx,
                            "cy": cy,
                            "w": w / w_img,
                            "h": h / h_img,
                            "area": norm_area,
                            "bottom_y": bottom_y
                        }
                        track_history[int(tid)].append(state)
                        current_frame_objs.append(state)

                frame_tracks.append(current_frame_objs)
                frame_count += 1

            curr_frame += 1

        cap.release()

        # Compute Model B tracking features
        features = self._compute_tracking_features(track_history, frame_tracks, effective_dt)
        return features, track_history

    def _compute_tracking_features(self, track_history, frame_tracks, dt):
        """
        Calculates image-space kinematic features, expansion rates, and relative motion.
        """
        if len(track_history) == 0:
            return {
                "active_track_count": 0,
                "mean_track_duration": 0.0,
                "max_track_duration": 0.0,
                "max_object_speed": 0.0,
                "mean_object_speed": 0.0,
                "max_approach_velocity_y": 0.0,
                "max_lateral_velocity_x": 0.0,
                "max_area_growth_rate": 0.0,
                "mean_area_growth_rate": 0.0,
                "min_pair_distance": 1.0,
                "max_pair_closing_rate": 0.0,
                "max_tracked_area": 0.0,
                "closest_tracked_bottom_y": 0.0
            }

        speeds = []
        vy_approaches = []
        vx_laterals = []
        area_growths = []
        durations = []
        tracked_areas = []
        bottom_ys = []

        for tid, states in track_history.items():
            durations.append(len(states))
            for s in states:
                tracked_areas.append(s["area"])
                bottom_ys.append(s["bottom_y"])

            if len(states) >= 2:
                for i in range(1, len(states)):
                    s_prev = states[i - 1]
                    s_curr = states[i]

                    dx = (s_curr["cx"] - s_prev["cx"])
                    dy = (s_curr["cy"] - s_prev["cy"])
                    vx = dx / dt
                    vy = dy / dt
                    speed = np.sqrt(vx**2 + vy**2)

                    speeds.append(speed)
                    vx_laterals.append(abs(vx))
                    vy_approaches.append(vy)  # positive means downward/approaching camera in image space

                    # Area growth rate
                    if s_prev["area"] > 1e-5:
                        growth = (s_curr["area"] - s_prev["area"]) / (s_prev["area"] * dt)
                        area_growths.append(growth)

        # Pairwise relative motion between objects in the same frame
        pair_distances = []
        pair_closing_rates = []

        for i in range(1, len(frame_tracks)):
            prev_objs = {o["track_id"]: o for o in frame_tracks[i - 1]}
            curr_objs = {o["track_id"]: o for o in frame_tracks[i]}

            # Find common tracks present in both frames
            common_ids = list(set(prev_objs.keys()) & set(curr_objs.keys()))
            if len(common_ids) >= 2:
                for idx_a in range(len(common_ids)):
                    for idx_b in range(idx_a + 1, len(common_ids)):
                        id_a, id_b = common_ids[idx_a], common_ids[idx_b]
                        # current distance
                        d_curr = np.sqrt(
                            (curr_objs[id_a]["cx"] - curr_objs[id_b]["cx"])**2 +
                            (curr_objs[id_a]["cy"] - curr_objs[id_b]["cy"])**2
                        )
                        # previous distance
                        d_prev = np.sqrt(
                            (prev_objs[id_a]["cx"] - prev_objs[id_b]["cx"])**2 +
                            (prev_objs[id_a]["cy"] - prev_objs[id_b]["cy"])**2
                        )
                        closing_rate = -(d_curr - d_prev) / dt
                        pair_distances.append(d_curr)
                        pair_closing_rates.append(closing_rate)

        return {
            "active_track_count": len(track_history),
            "mean_track_duration": float(np.mean(durations)) if durations else 0.0,
            "max_track_duration": float(np.max(durations)) if durations else 0.0,
            "max_object_speed": float(np.max(speeds)) if speeds else 0.0,
            "mean_object_speed": float(np.mean(speeds)) if speeds else 0.0,
            "max_approach_velocity_y": float(np.max(vy_approaches)) if vy_approaches else 0.0,
            "max_lateral_velocity_x": float(np.max(vx_laterals)) if vx_laterals else 0.0,
            "max_area_growth_rate": float(np.max(area_growths)) if area_growths else 0.0,
            "mean_area_growth_rate": float(np.mean(area_growths)) if area_growths else 0.0,
            "min_pair_distance": float(np.min(pair_distances)) if pair_distances else 1.0,
            "max_pair_closing_rate": float(np.max(pair_closing_rates)) if pair_closing_rates else 0.0,
            "max_tracked_area": float(np.max(tracked_areas)) if tracked_areas else 0.0,
            "closest_tracked_bottom_y": float(np.max(bottom_ys)) if bottom_ys else 0.0
        }

if __name__ == "__main__":
    print("Testing MultiObjectTracker on bench_pos_0001.mp4...")
    tracker = MultiObjectTracker()
    feats, history = tracker.process_window("data/raw/positive/bench_pos_0001.mp4", 0, 90, step_frames=6)
    print("\nModel B Tracking Features:")
    for k, v in feats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"Total unique tracks identified: {len(history)}")
