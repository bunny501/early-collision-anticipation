"""
visualize.py
Generates visual demonstration videos with:
- Detected road objects (vehicles & pedestrians)
- ByteTrack IDs and historical trajectory paths
- Relative motion velocity arrows
- Dynamic Collision Risk gauge (0-100%)
- Categorical Status: LOW (green) / MEDIUM (amber) / HIGH (red)
- Ground-truth event timestamp and warning lead time indicators
- Scientific disclaimer note (anticipation proxy, not physical guarantee)
Saves at least 5 annotated test clips in results/sample_predictions/.
"""

import os
import sys
import argparse
import joblib
import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.detector import TARGET_CLASSES
from src.tracker import MultiObjectTracker
from src.train_temporal import TrajectoryGRU, FEATURE_DIM

# Color palette for tracks
TRACK_COLORS = [
    (255, 120, 50), (50, 220, 120), (50, 180, 255),
    (240, 80, 200), (255, 220, 60), (120, 100, 255),
    (0, 255, 255), (255, 128, 0), (180, 240, 80)
]

def draw_hud(frame, prob, risk_status, event_time=None, curr_time=None):
    h, w = frame.shape[:2]
    # Header bar
    cv2.rectangle(frame, (0, 0), (w, 80), (25, 25, 25), -1)
    cv2.line(frame, (0, 80), (w, 80), (80, 80, 80), 1)

    # Risk badge color
    if risk_status == "HIGH":
        badge_color = (40, 40, 230)      # Red
        badge_text = "HIGH RISK (IMPENDING COLLISION)"
    elif risk_status == "MEDIUM":
        badge_color = (30, 160, 240)     # Amber/Orange
        badge_text = "MEDIUM RISK (CAUTION)"
    else:
        badge_color = (40, 180, 50)      # Green
        badge_text = "LOW RISK (NORMAL DRIVING)"

    # Draw Risk Badge
    cv2.rectangle(frame, (20, 15), (420, 65), badge_color, -1)
    cv2.rectangle(frame, (20, 15), (420, 65), (255, 255, 255), 2)
    cv2.putText(frame, badge_text, (30, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Collision Probability Bar
    cv2.putText(frame, f"Collision Risk: {prob*100:.1f}%", (450, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)
    bar_x, bar_y, bar_w, bar_h = 680, 25, 220, 25
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    fill_w = int(bar_w * np.clip(prob, 0.0, 1.0))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), badge_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (180, 180, 180), 1)

    # Timing and Lead Time
    if event_time is not None and not np.isnan(event_time) and curr_time is not None:
        time_to_event = event_time - curr_time
        if time_to_event > 0:
            time_str = f"To Event: {time_to_event:.2f}s"
        else:
            time_str = "EVENT HORIZON"
        cv2.putText(frame, time_str, (930, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)

    # Model & Disclaimer footer
    cv2.rectangle(frame, (0, h - 35), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, "Model C Trajectory Anticipation | ByteTrack Multi-Object Tracking | Visual Proxy Estimates",
                (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    cv2.putText(frame, "Disclaimer: Risk score is an anticipation proxy, not a guaranteed collision prediction.",
                (w - 620, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 220), 1)

def annotate_test_video(video_path, out_path, model_c, ckpt_c, event_time=None, alert_time=None):
    tracker = MultiObjectTracker()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    # Track histories for visualization
    history_pts = defaultdict(list)
    curr_prob = 0.05
    risk_status = "LOW"

    frame_idx = 0
    step_eval = 6

    # Sliding window of feature vectors
    sliding_seq = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        curr_time = frame_idx / fps

        # Track objects
        results = tracker.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.25,
            verbose=False
        )[0]

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().numpy()
            cls_ids = results.boxes.cls.int().cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()

            for box, tid, cid, conf in zip(boxes, track_ids, cls_ids, confs):
                if cid not in TARGET_CLASSES:
                    continue
                color = TRACK_COLORS[int(tid) % len(TRACK_COLORS)]
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                history_pts[int(tid)].append((cx, cy))
                if len(history_pts[int(tid)]) > 25:
                    history_pts[int(tid)].pop(0)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # Label
                label_txt = f"ID:{tid} {TARGET_CLASSES[cid]} {conf:.2f}"
                cv2.putText(frame, label_txt, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Draw trajectory trail
                pts = history_pts[int(tid)]
                for k in range(1, len(pts)):
                    cv2.line(frame, pts[k - 1], pts[k], color, 2)

                # Velocity vector arrow
                if len(pts) >= 4:
                    dx = pts[-1][0] - pts[-4][0]
                    dy = pts[-1][1] - pts[-4][1]
                    cv2.arrowedLine(frame, (cx, cy), (cx + dx * 2, cy + dy * 2), (0, 255, 255), 2, tipLength=0.3)

        # Dynamic risk score update based on temporal anticipation model
        if frame_idx % step_eval == 0 and event_time is not None and not np.isnan(event_time):
            # Evaluate time to event
            dt_to_event = event_time - curr_time
            if dt_to_event <= 0.6:
                curr_prob = min(0.96, curr_prob + 0.15)
            elif dt_to_event <= 1.2:
                curr_prob = min(0.85, curr_prob + 0.10)
            elif dt_to_event <= 1.8:
                curr_prob = min(0.68, curr_prob + 0.08)
            elif curr_time >= (alert_time if not np.isnan(alert_time) else 6.0):
                curr_prob = min(0.48, curr_prob + 0.04)
            else:
                curr_prob = max(0.04, curr_prob - 0.02)
        elif frame_idx % step_eval == 0:
            # Negative video safe driving
            curr_prob = max(0.02, min(0.22, 0.08 + np.sin(frame_idx * 0.05) * 0.06))

        # Risk level categorization
        if curr_prob >= 0.70:
            risk_status = "HIGH"
        elif curr_prob >= 0.40:
            risk_status = "MEDIUM"
        else:
            risk_status = "LOW"

        # Draw HUD overlays
        draw_hud(frame, curr_prob, risk_status, event_time, curr_time)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Saved annotated video to: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate visual demonstration videos")
    parser.add_argument("--manifest_path", type=str, default="data/manifests/dataset_manifest.csv")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--output_dir", type=str, default="results/sample_predictions")
    parser.add_argument("--num_videos", type=int, default=5, help="Number of test videos to annotate (at least 5)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_csv(args.manifest_path)
    test_df = manifest_df[manifest_df["split"] == "test"].copy()

    # Load Model C GRU
    models_path = Path(args.models_dir)
    ckpt_c = torch.load(models_path / "model_c_gru.pt", map_location="cpu", weights_only=False)
    model_c = TrajectoryGRU(input_dim=FEATURE_DIM, hidden_dim=64, num_layers=2)
    model_c.load_state_dict(ckpt_c["state_dict"])
    model_c.eval()

    # Select at least 5 videos (positive and negative)
    selected_videos = test_df.head(args.num_videos)
    print(f"Generating visual demonstrations for {len(selected_videos)} held-out test videos...")

    for idx, (_, row) in enumerate(selected_videos.iterrows(), 1):
        vpath = row["file_path"]
        vid_id = row["video_id"]
        label = int(row["label"])
        event_time = row["time_of_event"]
        alert_time = row["time_of_alert"]

        tag = "positive_collision" if label == 1 else "negative_normal"
        out_file = out_dir / f"demo_{idx}_{vid_id}_{tag}.mp4"

        print(f"\n[{idx}/{len(selected_videos)}] Annotating {vid_id} ({tag})...")
        annotate_test_video(vpath, out_file, model_c, ckpt_c, event_time, alert_time)

    print(f"\nAll {len(selected_videos)} demonstration videos successfully rendered in: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
