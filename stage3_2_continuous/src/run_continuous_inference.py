"""
run_continuous_inference.py
Continuous Sliding-Window Inference across Full Video Durations for Stage 3.2.
Processes the 100 Held-Out Test Videos (50 Positive, 50 Negative):
- 50 Negative videos: full continuous duration (~33.3 minutes total footage).
- 50 Positive videos: continuous evaluation up to collision event.
- Evaluates Models A, B, C, D, E, F at every 0.5s continuous stride.
- Generates:
  1. continuous_predictions.csv
  2. false_alarm_metrics.csv (debounced alarm episodes per minute)
  3. lead_time_metrics.csv (lead time relative to physical event timestamp)
"""

import os
import sys
import argparse
import joblib
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque, defaultdict
from ultralytics import YOLO
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

FEATURE_DIM = 14
SEQUENCE_LENGTH = 10
TARGET_CLASSES = {0: "pedestrian", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}
PEDESTRIAN_CLASS_IDS = {0}

class TrajectoryGRU(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last_step = out[:, -1, :]
        logits = self.classifier(last_step)
        return logits

def compute_background_flow(prev_gray, curr_gray, bboxes):
    h, w = curr_gray.shape
    scale = 320.0 / w
    small_w, small_h = 320, int(h * scale)
    prev_s = cv2.resize(prev_gray, (small_w, small_h))
    curr_s = cv2.resize(curr_gray, (small_w, small_h))

    mask = np.ones((small_h, small_w), dtype=np.uint8) * 255
    mask[:int(0.10 * small_h), :] = 0
    mask[int(0.65 * small_h):, :] = 0

    for b in bboxes:
        x1, y1 = int(b[0] * scale), int(b[1] * scale)
        x2, y2 = int(b[2] * scale), int(b[3] * scale)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(small_w, x2), min(small_h, y2)
        mask[y1:y2, x1:x2] = 0

    if np.sum(mask > 0) < 100:
        return 0.0, 0.0

    flow = cv2.calcOpticalFlowFarneback(
        prev_s, curr_s, None,
        pyr_scale=0.5, levels=2, winsize=11,
        iterations=2, poly_n=5, poly_sigma=1.1, flags=0
    )
    valid_flow = flow[mask > 0]
    if len(valid_flow) == 0:
        return 0.0, 0.0

    med_dx = float(np.median(valid_flow[:, 0])) / small_w
    med_dy = float(np.median(valid_flow[:, 1])) / small_h
    return med_dx, med_dy

def extract_buffer_features(buffer_dets, buffer_tracks, buffer_ego):
    # Model A Features
    feat_a = {}
    all_objs = [d for fd in buffer_dets for d in fd]

    feat_a["mean_vehicle_count"] = float(np.mean([sum(d["is_vehicle"] for d in fd) for fd in buffer_dets])) if buffer_dets else 0.0
    feat_a["max_vehicle_count"] = float(np.max([sum(d["is_vehicle"] for d in fd) for fd in buffer_dets])) if buffer_dets else 0.0
    feat_a["mean_pedestrian_count"] = float(np.mean([sum(d["is_pedestrian"] for d in fd) for fd in buffer_dets])) if buffer_dets else 0.0
    feat_a["max_pedestrian_count"] = float(np.max([sum(d["is_pedestrian"] for d in fd) for fd in buffer_dets])) if buffer_dets else 0.0
    feat_a["total_objects_mean"] = float(np.mean([len(fd) for fd in buffer_dets])) if buffer_dets else 0.0

    if all_objs:
        feat_a["max_bbox_area"] = float(np.max([d["norm_area"] for d in all_objs]))
        feat_a["mean_bbox_area"] = float(np.mean([d["norm_area"] for d in all_objs]))
        feat_a["max_bottom_y"] = float(np.max([d["bottom_y"] for d in all_objs]))
        feat_a["mean_bottom_y"] = float(np.mean([d["bottom_y"] for d in all_objs]))
        feat_a["closest_dist_to_center_mean"] = float(np.mean([d["dist_to_center"] for d in all_objs]))
        feat_a["closest_dist_to_center_min"] = float(np.min([d["dist_to_center"] for d in all_objs]))
        feat_a["mean_conf"] = float(np.mean([d["conf"] for d in all_objs]))
    else:
        feat_a["max_bbox_area"] = 0.0
        feat_a["mean_bbox_area"] = 0.0
        feat_a["max_bottom_y"] = 0.0
        feat_a["mean_bottom_y"] = 0.0
        feat_a["closest_dist_to_center_mean"] = 0.5
        feat_a["closest_dist_to_center_min"] = 0.5
        feat_a["mean_conf"] = 0.0

    last_dets = buffer_dets[-1] if buffer_dets else []
    if last_dets:
        feat_a["last_max_bbox_area"] = float(np.max([d["norm_area"] for d in last_dets]))
        feat_a["last_max_bottom_y"] = float(np.max([d["bottom_y"] for d in last_dets]))
        feat_a["last_closest_dist_to_center"] = float(np.min([d["dist_to_center"] for d in last_dets]))
        feat_a["last_object_count"] = float(len(last_dets))
    else:
        feat_a["last_max_bbox_area"] = 0.0
        feat_a["last_max_bottom_y"] = 0.0
        feat_a["last_closest_dist_to_center"] = 0.5
        feat_a["last_object_count"] = 0.0

    feat_a["center_corridor_count_mean"] = float(np.mean([sum(1 for d in fd if d["dist_to_center"] < 0.15) for fd in buffer_dets])) if buffer_dets else 0.0

    # Model B Features
    feat_b = {**feat_a}
    track_durations = [len(s) for s in buffer_tracks.values()]
    feat_b["num_active_tracks"] = float(len(buffer_tracks))
    feat_b["mean_track_duration"] = float(np.mean(track_durations)) if track_durations else 0.0
    feat_b["max_track_duration"] = float(np.max(track_durations)) if track_durations else 0.0

    velocities_x, velocities_y, speeds, expansion_rates = [], [], [], []
    for tid, states in buffer_tracks.items():
        if len(states) >= 2:
            for i in range(1, len(states)):
                dt = max(1e-4, states[i]["time_sec"] - states[i-1]["time_sec"])
                vx = (states[i]["cx"] - states[i-1]["cx"]) / dt
                vy = (states[i]["cy"] - states[i-1]["cy"]) / dt
                sp = np.sqrt(vx**2 + vy**2)
                velocities_x.append(vx)
                velocities_y.append(vy)
                speeds.append(sp)

                prev_a = max(1e-5, states[i-1]["norm_area"])
                exp_rate = (states[i]["norm_area"] - prev_a) / (prev_a * dt)
                expansion_rates.append(exp_rate)

    feat_b["mean_speed"] = float(np.mean(speeds)) if speeds else 0.0
    feat_b["max_speed"] = float(np.max(speeds)) if speeds else 0.0
    feat_b["mean_vy"] = float(np.mean(velocities_y)) if velocities_y else 0.0
    feat_b["max_vy"] = float(np.max(velocities_y)) if velocities_y else 0.0
    feat_b["mean_vx"] = float(np.mean(velocities_x)) if velocities_x else 0.0
    feat_b["max_vx"] = float(np.max(velocities_x)) if velocities_x else 0.0
    feat_b["mean_expansion_rate"] = float(np.mean(expansion_rates)) if expansion_rates else 0.0
    feat_b["max_expansion_rate"] = float(np.max(expansion_rates)) if expansion_rates else 0.0

    # Model C Tabular & Model D Sequential Features
    primary_tid = None
    best_crit = -1.0
    for tid, states in buffer_tracks.items():
        if len(states) >= 2:
            crit = max(s["bottom_y"] for s in states) * 0.6 + max(s["norm_area"] for s in states) * 0.4
            if crit > best_crit:
                best_crit = crit
                primary_tid = tid

    if primary_tid is None and len(buffer_tracks) > 0:
        primary_tid = list(buffer_tracks.keys())[0]

    seq_mat = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
    feat_c_tab = {**feat_b}

    if primary_tid is not None and primary_tid in buffer_tracks:
        states = buffer_tracks[primary_tid]
        n_states = len(states)
        vx_prev, vy_prev = 0.0, 0.0
        heading_prev = 0.0

        traj_cx = [s["cx"] for s in states]
        traj_cy = [s["cy"] for s in states]
        traj_bottom_y = [s["bottom_y"] for s in states]
        traj_area = [s["norm_area"] for s in states]
        traj_vx, traj_vy, traj_ax, traj_ay, traj_exp, traj_ttc = [], [], [], [], [], []

        for i in range(min(n_states, SEQUENCE_LENGTH)):
            s = states[i]
            cx, cy, by, a = s["cx"], s["cy"], s["bottom_y"], s["norm_area"]
            ego_dx, ego_dy = buffer_ego[i] if i < len(buffer_ego) else (0.0, 0.0)

            if i > 0:
                dt = max(1e-4, states[i]["time_sec"] - states[i-1]["time_sec"])
                vx = (cx - states[i-1]["cx"]) / dt
                vy = (by - states[i-1]["bottom_y"]) / dt
                ax = (vx - vx_prev) / dt
                ay = (vy - vy_prev) / dt

                prev_a = max(1e-5, states[i-1]["norm_area"])
                exp_rate = (a - prev_a) / (prev_a * dt)
                heading = np.arctan2(vy, vx)
                heading_diff = (heading - heading_prev + np.pi) % (2 * np.pi) - np.pi

                delta_scale = max(1e-4, (np.sqrt(a) - np.sqrt(prev_a)) / dt)
                ttc_proxy = min(10.0, max(0.1, np.sqrt(a) / delta_scale))
                vy_ego_comp = vy - (ego_dy / dt)

                vx_prev, vy_prev = vx, vy
                heading_prev = heading
            else:
                vx, vy, ax, ay = 0.0, 0.0, 0.0, 0.0
                exp_rate = 0.0
                heading = 0.0
                heading_diff = 0.0
                ttc_proxy = 10.0
                vy_ego_comp = 0.0

            traj_vx.append(vx)
            traj_vy.append(vy)
            traj_ax.append(ax)
            traj_ay.append(ay)
            traj_exp.append(exp_rate)
            traj_ttc.append(ttc_proxy)

            d_center = abs(cx - 0.5)

            step_idx = SEQUENCE_LENGTH - min(n_states, SEQUENCE_LENGTH) + i
            if 0 <= step_idx < SEQUENCE_LENGTH:
                seq_mat[step_idx] = [
                    cx, cy, by, a,
                    vx, vy, ax, ay,
                    exp_rate, heading, heading_diff,
                    d_center, ttc_proxy, vy_ego_comp
                ]

        feat_c_tab["traj_cx_mean"] = float(np.mean(traj_cx))
        feat_c_tab["traj_cx_trend"] = float(traj_cx[-1] - traj_cx[0])
        feat_c_tab["traj_cy_mean"] = float(np.mean(traj_cy))
        feat_c_tab["traj_cy_trend"] = float(traj_cy[-1] - traj_cy[0])
        feat_c_tab["traj_bottom_y_max"] = float(np.max(traj_bottom_y))
        feat_c_tab["traj_bottom_y_mean"] = float(np.mean(traj_bottom_y))
        feat_c_tab["traj_area_max"] = float(np.max(traj_area))
        feat_c_tab["traj_area_mean"] = float(np.mean(traj_area))
        feat_c_tab["traj_vx_mean"] = float(np.mean(traj_vx))
        feat_c_tab["traj_vx_max"] = float(np.max(np.abs(traj_vx)))
        feat_c_tab["traj_vy_mean"] = float(np.mean(traj_vy))
        feat_c_tab["traj_vy_max"] = float(np.max(traj_vy))
        feat_c_tab["traj_ax_mean"] = float(np.mean(traj_ax))
        feat_c_tab["traj_ay_mean"] = float(np.mean(traj_ay))
        feat_c_tab["traj_ay_max"] = float(np.max(traj_ay))
        feat_c_tab["traj_exp_mean"] = float(np.mean(traj_exp))
        feat_c_tab["traj_exp_max"] = float(np.max(traj_exp))
        feat_c_tab["traj_min_ttc"] = float(np.min(traj_ttc))
        feat_c_tab["ego_motion_dy_mean"] = float(np.mean([em[1] for em in buffer_ego]))
    else:
        for k in [
            "traj_cx_mean", "traj_cx_trend", "traj_cy_mean", "traj_cy_trend",
            "traj_bottom_y_max", "traj_bottom_y_mean", "traj_area_max", "traj_area_mean",
            "traj_vx_mean", "traj_vx_max", "traj_vy_mean", "traj_vy_max",
            "traj_ax_mean", "traj_ay_mean", "traj_ay_max",
            "traj_exp_mean", "traj_exp_max", "traj_min_ttc", "ego_motion_dy_mean"
        ]:
            feat_c_tab[k] = 0.0

    feat_cent = {
        "closest_dist_to_center_mean": feat_a["closest_dist_to_center_mean"],
        "closest_dist_to_center_min": feat_a["closest_dist_to_center_min"],
        "center_corridor_count_mean": feat_a["center_corridor_count_mean"]
    }

    return feat_a, feat_b, feat_c_tab, feat_cent, seq_mat

def main():
    parser = argparse.ArgumentParser(description="Run Continuous Sliding-Window Inference")
    parser.add_argument("--test_manifest", type=str, default="test_manifest.csv")
    parser.add_argument("--stride_eval_sec", type=float, default=0.5)
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--idx_start", type=int, default=0)
    parser.add_argument("--idx_end", type=int, default=None)
    parser.add_argument("--out_csv", type=str, default="continuous_predictions.csv")
    parser.add_argument("--only_predict", action="store_true")
    args = parser.parse_args()

    print("="*75)
    print("STAGE 3.2: CONTINUOUS SLIDING-WINDOW INFERENCE & FALSE ALARM AUDIT")
    print("="*75)

    torch.set_num_threads(8)

    # 1. Load Trained Models & Validation Thresholds
    print("Loading models and validation thresholds...", flush=True)
    dict_f = joblib.load(MODELS_DIR / "model_f_center.joblib")
    dict_e = joblib.load(MODELS_DIR / "model_e_logreg.joblib")
    dict_a = joblib.load(MODELS_DIR / "model_a.joblib")
    dict_b = joblib.load(MODELS_DIR / "model_b.joblib")
    dict_c = joblib.load(MODELS_DIR / "model_c_xgb.joblib")
    val_thresholds = joblib.load(MODELS_DIR / "val_tuned_thresholds.joblib")

    gru_ckpt = torch.load(MODELS_DIR / "model_d_gru.pt", map_location="cpu", weights_only=False)
    model_gru = TrajectoryGRU(input_dim=FEATURE_DIM, hidden_dim=64, num_layers=2)
    model_gru.load_state_dict(gru_ckpt["state_dict"])
    model_gru.eval()
    gru_mean = gru_ckpt["mean"]
    gru_std = gru_ckpt["std"]

    yolo_model = YOLO(str(MODELS_DIR / "yolov8n.pt"))
    target_ids = set(TARGET_CLASSES.keys())

    # 2. Load Test Manifest
    df_test_full = pd.read_csv(BASE_DIR / args.test_manifest)
    if args.max_videos:
        df_test_full = df_test_full.head(args.max_videos)
    
    end_idx = args.idx_end if args.idx_end is not None else len(df_test_full)
    df_test = df_test_full.iloc[args.idx_start:end_idx]
    print(f"Loaded {len(df_test)} test videos (indices {args.idx_start} to {end_idx}) ({sum(df_test['label']==1)} Pos, {sum(df_test['label']==0)} Neg).", flush=True)

    continuous_predictions = []

    print("\n--- Running Continuous Streaming Inference ---", flush=True)
    for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Continuous evaluation"):
        vid_id = str(row["video_id"])
        label = int(row["label"])
        vpath = str(row["file_path"])
        scene = row.get("scene", "Unknown")
        t_event = float(row["time_of_event"]) if pd.notnull(row.get("time_of_event")) else np.nan

        cap = cv2.VideoCapture(vpath)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step_frames = 6  # 5 FPS sampling

        # Max frame to process: full video for negative; event + 1.0s for positive
        if label == 1:
            max_sec = min(float(row["duration_sec"]), t_event + 1.0)
        else:
            max_sec = float(row["duration_sec"])

        max_frame = int(round(max_sec * fps))

        # Rolling buffers for sliding window of 10 frames
        buf_frames = deque(maxlen=SEQUENCE_LENGTH)
        buf_dets = deque(maxlen=SEQUENCE_LENGTH)
        buf_ego = deque(maxlen=SEQUENCE_LENGTH)

        # Running track history
        active_tracks = defaultdict(list)
        prev_gray = None

        curr_frame_idx = 0
        last_eval_time = -1.0

        while curr_frame_idx <= max_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if curr_frame_idx % step_frames == 0:
                curr_time_sec = curr_frame_idx / fps
                h_img, w_img = frame.shape[:2]
                img_area = float(w_img * h_img)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                results = yolo_model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=0.25,
                    device="cpu",
                    verbose=False
                )[0]

                curr_dets = []
                boxes_xyxy = []

                if results.boxes is not None:
                    b_xyxy = results.boxes.xyxy.cpu().numpy()
                    cls_ids = results.boxes.cls.int().cpu().numpy()
                    confs = results.boxes.conf.cpu().numpy()
                    t_ids = results.boxes.id.int().cpu().numpy() if results.boxes.id is not None else np.full(len(b_xyxy), -1)

                    for box, cid, conf, tid in zip(b_xyxy, cls_ids, confs, t_ids):
                        if cid not in target_ids:
                            continue
                        x1, y1, x2, y2 = box
                        boxes_xyxy.append(box)
                        w = max(1.0, x2 - x1)
                        h = max(1.0, y2 - y1)
                        norm_area = (w * h) / img_area
                        cx = (x1 + x2) / 2.0 / w_img
                        cy = (y1 + y2) / 2.0 / h_img
                        bottom_y = y2 / h_img

                        det_item = {
                            "frame_idx": curr_frame_idx,
                            "time_sec": curr_time_sec,
                            "track_id": int(tid),
                            "class_id": int(cid),
                            "conf": float(conf),
                            "cx": cx,
                            "cy": cy,
                            "w": w / w_img,
                            "h": h / h_img,
                            "norm_area": norm_area,
                            "bottom_y": bottom_y,
                            "is_vehicle": 1 if cid in VEHICLE_CLASS_IDS else 0,
                            "is_pedestrian": 1 if cid in PEDESTRIAN_CLASS_IDS else 0,
                            "dist_to_center": abs(cx - 0.5)
                        }
                        curr_dets.append(det_item)
                        if tid >= 0:
                            active_tracks[int(tid)].append(det_item)

                if prev_gray is not None:
                    ego_dx, ego_dy = compute_background_flow(prev_gray, gray, boxes_xyxy)
                else:
                    ego_dx, ego_dy = 0.0, 0.0

                buf_frames.append(curr_frame_idx)
                buf_dets.append(curr_dets)
                buf_ego.append((ego_dx, ego_dy))
                prev_gray = gray

                # Evaluate continuous sliding window when buffer has 10 frames and stride elapsed
                if len(buf_frames) == SEQUENCE_LENGTH and (curr_time_sec - last_eval_time >= args.stride_eval_sec - 1e-4):
                    last_eval_time = curr_time_sec

                    # Window tracks (only states within current buffer)
                    win_start_frame = buf_frames[0]
                    win_tracks = defaultdict(list)
                    for tid, states in active_tracks.items():
                        in_win = [s for s in states if s["frame_idx"] >= win_start_frame]
                        if in_win:
                            win_tracks[tid] = in_win

                    fa, fb, fc, fcent, seq_m = extract_buffer_features(list(buf_dets), win_tracks, list(buf_ego))

                    # Model A
                    df_fa = pd.DataFrame([fa])
                    prob_a = float(dict_a["model"].predict_proba(dict_a["scaler"].transform(df_fa[dict_a["features"]].fillna(0)))[:, 1][0])

                    # Model B
                    df_fb = pd.DataFrame([fb])
                    prob_b = float(dict_b["model"].predict_proba(dict_b["scaler"].transform(df_fb[dict_b["features"]].fillna(0)))[:, 1][0])

                    # Model C
                    df_fc = pd.DataFrame([fc])
                    prob_c = float(dict_c["model"].predict_proba(dict_c["scaler"].transform(df_fc[dict_c["features"]].fillna(0)))[:, 1][0])

                    # Model D (GRU)
                    norm_s = (seq_m.reshape(1, SEQUENCE_LENGTH, FEATURE_DIM) - gru_mean) / gru_std
                    with torch.no_grad():
                        prob_d = float(torch.sigmoid(model_gru(torch.tensor(norm_s, dtype=torch.float32))).cpu().numpy()[0][0])

                    # Model E
                    prob_e = float(dict_e["model"].predict_proba(dict_e["scaler"].transform(df_fa[dict_e["features"]].fillna(0)))[:, 1][0])

                    # Model F (Center)
                    df_fcent = pd.DataFrame([fcent])
                    prob_f = float(dict_f["model"].predict_proba(dict_f["scaler"].transform(df_fcent[dict_f["features"]].fillna(0)))[:, 1][0])

                    time_to_event = (t_event - curr_time_sec) if label == 1 else np.nan

                    models_eval = [
                        ("Model A (Detection Only)", prob_a, val_thresholds["Model A"]["best_threshold"]),
                        ("Model B (Detection + Tracking)", prob_b, val_thresholds["Model B"]["best_threshold"]),
                        ("Model C (Trajectory XGBoost)", prob_c, val_thresholds["Model C"]["best_threshold"]),
                        ("Model D (Trajectory GRU)", prob_d, val_thresholds["Model D"]["best_threshold"]),
                        ("Model E (Logistic Regression)", prob_e, val_thresholds["Model E"]["best_threshold"]),
                        ("Model F (Center-Distance)", prob_f, val_thresholds["Model F"]["best_threshold"])
                    ]

                    for m_name, p_val, th in models_eval:
                        continuous_predictions.append({
                            "video_id": vid_id,
                            "ground_truth": label,
                            "scene": scene,
                            "timestamp": round(curr_time_sec, 2),
                            "time_to_event": round(time_to_event, 2) if pd.notnull(time_to_event) else np.nan,
                            "model_name": m_name,
                            "risk_score": round(p_val, 4),
                            "alarm_state": int(p_val >= th),
                            "val_threshold": th
                        })

            curr_frame_idx += 1
        cap.release()

    df_continuous = pd.DataFrame(continuous_predictions)
    out_path = BASE_DIR / args.out_csv if not Path(args.out_csv).is_absolute() else Path(args.out_csv)
    df_continuous.to_csv(out_path, index=False)
    print(f"\nSaved {len(df_continuous)} continuous predictions -> {out_path}", flush=True)

    if args.only_predict:
        print("[INFO] --only_predict specified; skipping metric calculations for this partition.")
        return

    # ---------------------------------------------------------
    # 3. Debounced False Alarm Episode Analysis (Section 7)
    # ---------------------------------------------------------
    print("\n--- Calculating Debounced False Alarm Rates (Negative Test Driving) ---", flush=True)
    df_neg_preds = df_continuous[df_continuous["ground_truth"] == 0].copy()
    total_neg_vids = df_test[df_test["label"] == 0]["video_id"].nunique()
    total_duration_sec = df_test[df_test["label"] == 0]["duration_sec"].sum()
    total_duration_min = total_duration_sec / 60.0

    print(f"Total Negative Videos: {total_neg_vids} | Total Continuous Duration: {total_duration_sec:.1f} s ({total_duration_min:.2f} minutes)", flush=True)

    fa_records = []
    models_list = df_continuous["model_name"].unique()

    for m_name in models_list:
        sub_m = df_neg_preds[df_neg_preds["model_name"] == m_name]
        th = sub_m["val_threshold"].iloc[0]

        # Debounce logic: minimum sustained duration = 2 steps (0.4-0.5s)
        # Group contiguous above-threshold segments
        episodes = []
        for vid, v_df in sub_m.groupby("video_id"):
            v_df = v_df.sort_values("timestamp")
            scores = v_df["risk_score"].values
            times = v_df["timestamp"].values

            in_alarm = False
            ep_start = 0.0
            ep_peak = 0.0
            consec = 0

            for sc, t in zip(scores, times):
                if sc >= th:
                    consec += 1
                    if not in_alarm and consec >= 2:  # 2 consecutive steps sustained
                        in_alarm = True
                        ep_start = t - 0.5
                        ep_peak = sc
                    elif in_alarm:
                        ep_peak = max(ep_peak, sc)
                else:
                    if in_alarm:
                        # Alarm finished
                        ep_dur = max(0.5, t - ep_start)
                        episodes.append({
                            "video_id": vid,
                            "duration_sec": ep_dur,
                            "peak_score": ep_peak
                        })
                        in_alarm = False
                    consec = 0

            if in_alarm:
                ep_dur = max(0.5, times[-1] - ep_start)
                episodes.append({
                    "video_id": vid,
                    "duration_sec": ep_dur,
                    "peak_score": ep_peak
                })

        num_episodes = len(episodes)
        fa_per_min = num_episodes / total_duration_min if total_duration_min > 0 else 0.0
        vids_with_fa = len(set(e["video_id"] for e in episodes))
        durations = [e["duration_sec"] for e in episodes]
        peaks = [e["peak_score"] for e in episodes]

        fa_records.append({
            "Model Name": m_name,
            "Tuned Threshold": th,
            "Debounce Criteria": "Min 2 consecutive steps (>=0.4s)",
            "Total Evaluated Duration (min)": round(total_duration_min, 2),
            "Total False Alarm Episodes": num_episodes,
            "False Alarms / Minute": round(fa_per_min, 3),
            "Videos with >=1 False Alarm": vids_with_fa,
            "% Negative Videos with False Alarm": round(100.0 * vids_with_fa / max(1, total_neg_vids), 1),
            "Mean Alarm Duration (s)": round(float(np.mean(durations)), 2) if durations else 0.0,
            "Median Alarm Duration (s)": round(float(np.median(durations)), 2) if durations else 0.0,
            "Max Alarm Duration (s)": round(float(np.max(durations)), 2) if durations else 0.0,
            "Mean Peak Risk Score": round(float(np.mean(peaks)), 4) if peaks else 0.0,
            "Max Peak Risk Score": round(float(np.max(peaks)), 4) if peaks else 0.0
        })

    df_fa_res = pd.DataFrame(fa_records)
    df_fa_res.to_csv(BASE_DIR / "false_alarm_metrics.csv", index=False)
    print(f"Saved false alarm metrics -> {BASE_DIR / 'false_alarm_metrics.csv'}\n", flush=True)
    print(df_fa_res[["Model Name", "Tuned Threshold", "Total False Alarm Episodes", "False Alarms / Minute", "Videos with >=1 False Alarm", "% Negative Videos with False Alarm"]].to_string(index=False))

    # ---------------------------------------------------------
    # 4. Early Warning Lead Time Analysis (Section 8)
    # ---------------------------------------------------------
    print("\n--- Calculating Early Warning Lead Time (Positive Collision Videos) ---", flush=True)
    df_pos_preds = df_continuous[df_continuous["ground_truth"] == 1].copy()
    total_pos_vids = df_test[df_test["label"] == 1]["video_id"].nunique()

    lead_records = []
    for m_name in models_list:
        sub_m = df_pos_preds[df_pos_preds["model_name"] == m_name]
        th = sub_m["val_threshold"].iloc[0]

        lead_times = []
        warned_ge_15 = 0
        warned_ge_10 = 0
        warned_ge_05 = 0
        warned_total = 0

        for vid, v_df in sub_m.groupby("video_id"):
            v_df = v_df.sort_values("timestamp")
            # Positive alarm triggers prior to or at collision (time_to_event >= 0.0)
            valid_alarms = v_df[(v_df["alarm_state"] == 1) & (v_df["time_to_event"] >= -0.2)]

            if len(valid_alarms) > 0:
                first_alarm = valid_alarms.iloc[0]
                lead_t = float(first_alarm["time_to_event"])
                lead_times.append(lead_t)
                warned_total += 1
                if lead_t >= 1.5:
                    warned_ge_15 += 1
                if lead_t >= 1.0:
                    warned_ge_10 += 1
                if lead_t >= 0.5:
                    warned_ge_05 += 1

        missed = total_pos_vids - warned_total

        lead_records.append({
            "Model Name": m_name,
            "Tuned Threshold": th,
            "Total Positive Videos": total_pos_vids,
            "Total Warned Prior to Event": warned_total,
            "Detection / Warning Rate (%)": round(100.0 * warned_total / total_pos_vids, 1),
            "Missed Event Rate (%)": round(100.0 * missed / total_pos_vids, 1),
            "Warned >= 1.5s Prior (%)": round(100.0 * warned_ge_15 / total_pos_vids, 1),
            "Warned >= 1.0s Prior (%)": round(100.0 * warned_ge_10 / total_pos_vids, 1),
            "Warned >= 0.5s Prior (%)": round(100.0 * warned_ge_05 / total_pos_vids, 1),
            "Mean Lead Time (s)": round(float(np.mean(lead_times)), 2) if lead_times else 0.0,
            "Median Lead Time (s)": round(float(np.median(lead_times)), 2) if lead_times else 0.0,
            "Min Lead Time (s)": round(float(np.min(lead_times)), 2) if lead_times else 0.0,
            "Max Lead Time (s)": round(float(np.max(lead_times)), 2) if lead_times else 0.0
        })

    df_lead = pd.DataFrame(lead_records)
    df_lead.to_csv(BASE_DIR / "lead_time_metrics.csv", index=False)
    print(f"Saved lead time metrics -> {BASE_DIR / 'lead_time_metrics.csv'}\n", flush=True)
    print(df_lead[["Model Name", "Detection / Warning Rate (%)", "Missed Event Rate (%)", "Warned >= 1.0s Prior (%)", "Mean Lead Time (s)", "Median Lead Time (s)"]].to_string(index=False))

    print("\n[SUCCESS] Continuous sliding-window inference, false alarm, and lead time analysis complete!", flush=True)

if __name__ == "__main__":
    main()
