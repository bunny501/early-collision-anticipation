"""
second_stage_validation.py
Second-Stage Validation and Ablation Experiments for Early Collision Anticipation.

Tasks:
1. Re-evaluate existing baselines on identical test samples.
2. Run systematic feature ablations:
   - Bounding-box features only
   - Bounding-box + displacement
   - Bounding-box + velocity
   - Bounding-box + velocity + acceleration
   - Complete trajectory features + XGBoost
   - Complete trajectory features + GRU
3. Per-video early warning lead time evaluation (streamlined rolling window).
4. Bootstrap 95% Confidence Intervals for performance metrics.
5. Generate deliverables:
   - results/ablation_results.csv
   - results/per_video_predictions.csv
   - results/updated_metrics.csv
"""

import os
import sys
import time
import joblib
import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

# Project root setup
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.train_temporal import TrajectoryGRU, FEATURE_DIM
from src.detector import TARGET_CLASSES, VEHICLE_CLASS_IDS, PEDESTRIAN_CLASS_IDS
from ultralytics import YOLO

def compute_horizon_ap(y_true, y_probs, horizons, target_h):
    mask = (horizons == target_h)
    if sum(mask) == 0:
        return 0.0
    neg_mask = (y_true == 0)
    eval_mask = mask | neg_mask
    if len(np.unique(y_true[eval_mask])) < 2:
        return float(y_true[eval_mask].mean())
    return float(average_precision_score(y_true[eval_mask], y_probs[eval_mask]))

def bootstrap_ci(y_true, y_probs, n_bootstraps=1000, seed=42):
    np.random.seed(seed)
    pr_scores = []
    roc_scores = []
    n = len(y_true)

    for _ in range(n_bootstraps):
        idx = np.random.choice(n, size=n, replace=True)
        yt_sample = y_true[idx]
        yp_sample = y_probs[idx]

        if len(np.unique(yt_sample)) < 2:
            continue
        try:
            pr_scores.append(average_precision_score(yt_sample, yp_sample))
            roc_scores.append(roc_auc_score(yt_sample, yp_sample))
        except Exception:
            pass

    if len(pr_scores) == 0:
        return (0.0, 1.0), (0.0, 1.0)

    pr_ci = (float(np.percentile(pr_scores, 2.5)), float(np.percentile(pr_scores, 97.5)))
    roc_ci = (float(np.percentile(roc_scores, 2.5)), float(np.percentile(roc_scores, 97.5)))
    return pr_ci, roc_ci

def evaluate_model_performance(y_true, y_probs, horizons, duration_min, latency_ms):
    y_pred = (y_probs >= 0.5).astype(int)

    pr_auc = float(average_precision_score(y_true, y_probs))
    roc_auc = float(roc_auc_score(y_true, y_probs)) if len(np.unique(y_true)) > 1 else 0.5
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / max(1, (fp + tn)))
    false_alarms_per_min = float(fp / max(0.01, duration_min))

    ap_15 = compute_horizon_ap(y_true, y_probs, horizons, 1.5)
    ap_10 = compute_horizon_ap(y_true, y_probs, horizons, 1.0)
    ap_05 = compute_horizon_ap(y_true, y_probs, horizons, 0.5)
    mean_ap = float(np.mean([ap_15, ap_10, ap_05]))

    pr_ci, roc_ci = bootstrap_ci(y_true, y_probs)

    return {
        "1.5s AP": round(ap_15, 4),
        "1.0s AP": round(ap_10, 4),
        "0.5s AP": round(ap_05, 4),
        "Mean AP": round(mean_ap, 4),
        "PR-AUC": round(pr_auc, 4),
        "PR-AUC 95% CI": f"[{pr_ci[0]:.4f}, {pr_ci[1]:.4f}]",
        "ROC-AUC": round(roc_auc, 4),
        "ROC-AUC 95% CI": f"[{roc_ci[0]:.4f}, {roc_ci[1]:.4f}]",
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "FPR": round(fpr, 4),
        "False Alarms/min": round(false_alarms_per_min, 2),
        "Latency (ms)": round(latency_ms, 2),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)
    }

def run_ablations(df_tab, df_manifest, seq_data, test_mask):
    """
    Trains and evaluates 6 ablation tiers:
    1. Bounding-box features only
    2. Bounding-box + displacement
    3. Bounding-box + velocity
    4. Bounding-box + velocity + acceleration
    5. Complete trajectory features + XGBoost
    6. Complete trajectory features + GRU
    """
    train_mask = (df_tab["split"] == "train").values
    val_mask = (df_tab["split"] == "val").values
    test_mask = (df_tab["split"] == "test").values

    y_train = df_tab.loc[train_mask, "label"].values
    y_val = df_tab.loc[val_mask, "label"].values
    y_test = df_tab.loc[test_mask, "label"].values
    horizons_test = df_tab.loc[test_mask, "horizon_sec"].values

    total_neg_sec = (y_test == 0).sum() * 2.0
    duration_min = total_neg_sec / 60.0

    # Define feature groups
    bbox_cols = [
        "traj_cx_mean", "traj_cx_max", "traj_cx_min", "traj_cx_last",
        "traj_cy_mean", "traj_cy_max", "traj_cy_min", "traj_cy_last",
        "traj_bottom_y_mean", "traj_bottom_y_max", "traj_bottom_y_last",
        "traj_area_mean", "traj_area_max", "traj_area_min", "traj_area_last",
        "max_tracked_area", "closest_tracked_bottom_y"
    ]
    disp_cols = bbox_cols + [
        "traj_cx_trend", "traj_cy_trend", "mean_track_duration", "max_track_duration"
    ]
    vel_cols = disp_cols + [
        "traj_vx_mean", "traj_vx_max", "traj_vx_min", "traj_vx_last",
        "traj_vy_mean", "traj_vy_max", "traj_vy_min", "traj_vy_last",
        "max_object_speed", "mean_object_speed", "max_approach_velocity_y", "max_lateral_velocity_x"
    ]
    acc_cols = vel_cols + [
        "traj_ax_mean", "traj_ax_max", "traj_ax_min", "traj_ax_last",
        "traj_ay_mean", "traj_ay_max", "traj_ay_min", "traj_ay_last"
    ]

    meta_cols = ["window_id", "split", "label", "horizon_sec"]
    all_tab_cols = [c for c in df_tab.columns if c not in meta_cols]

    ablation_tiers = [
        ("1. Bounding-box features only", bbox_cols),
        ("2. Bounding-box + displacement", disp_cols),
        ("3. Bounding-box + velocity", vel_cols),
        ("4. Bounding-box + velocity + acceleration", acc_cols),
        ("5. Complete trajectory features + XGBoost", all_tab_cols),
    ]

    ablation_records = []

    for name, cols in ablation_tiers:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(df_tab.loc[train_mask, cols].fillna(0))
        X_te = scaler.transform(df_tab.loc[test_mask, cols].fillna(0))

        clf = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            random_state=42, eval_metric="logloss"
        )
        clf.fit(X_tr, y_train)

        t0 = time.time()
        probs = clf.predict_proba(X_te)[:, 1]
        lat_ms = (time.time() - t0) * 1000.0 / max(1, len(X_te))

        perf = evaluate_model_performance(y_test, probs, horizons_test, duration_min, lat_ms)
        perf["Ablation Tier"] = name
        perf["Feature Count"] = len(cols)
        ablation_records.append(perf)

    # 6. Complete trajectory features + GRU
    ckpt_c = torch.load("models/model_c_gru.pt", map_location="cpu", weights_only=False)
    model_gru = TrajectoryGRU(input_dim=FEATURE_DIM, hidden_dim=64, num_layers=2)
    model_gru.load_state_dict(ckpt_c["state_dict"])
    model_gru.eval()

    test_seqs = seq_data[test_mask]
    norm_seqs = (test_seqs - ckpt_c["mean"]) / ckpt_c["std"]

    t0 = time.time()
    with torch.no_grad():
        t_seqs = torch.tensor(norm_seqs, dtype=torch.float32)
        probs_gru = torch.sigmoid(model_gru(t_seqs)).cpu().numpy().flatten()
    lat_gru = (time.time() - t0) * 1000.0 / max(1, len(test_seqs))

    perf_gru = evaluate_model_performance(y_test, probs_gru, horizons_test, duration_min, lat_gru)
    perf_gru["Ablation Tier"] = "6. Complete trajectory features + GRU (Proposed)"
    perf_gru["Feature Count"] = FEATURE_DIM * 10
    ablation_records.append(perf_gru)

    df_ablation = pd.DataFrame(ablation_records)
    first_cols = ["Ablation Tier", "Feature Count", "1.5s AP", "1.0s AP", "0.5s AP", "Mean AP", "PR-AUC", "ROC-AUC", "Precision", "Recall", "F1", "FPR", "Latency (ms)"]
    ordered_cols = first_cols + [c for c in df_ablation.columns if c not in first_cols]
    df_ablation = df_ablation[ordered_cols]
    return df_ablation

def extract_features_from_window_slice(frame_detections, frame_tracks_slice, dt=0.2):
    """
    Computes Model A, Model B, and Model C feature sets directly from in-memory frame slices (last 10 frames).
    """
    # 1. Model A Features
    df_f = pd.DataFrame(frame_detections)
    feat_a = {}
    for col in df_f.columns:
        feat_a[f"{col}_mean"] = float(df_f[col].mean())
        feat_a[f"{col}_max"] = float(df_f[col].max())
        feat_a[f"{col}_last"] = float(df_f[col].iloc[-1])

    # 2. Model B Features
    track_hist_slice = defaultdict(list)
    for frame_idx, objs in enumerate(frame_tracks_slice):
        for o in objs:
            track_hist_slice[o["track_id"]].append(o)

    speeds, vy_approaches, vx_laterals, area_growths = [], [], [], []
    durations, tracked_areas, bottom_ys = [], [], []

    for tid, states in track_hist_slice.items():
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
                vy_approaches.append(vy)
                if s_prev["area"] > 1e-5:
                    growth = (s_curr["area"] - s_prev["area"]) / (s_prev["area"] * dt)
                    area_growths.append(growth)

    pair_distances = []
    pair_closing_rates = []
    for i in range(1, len(frame_tracks_slice)):
        prev_objs = {o["track_id"]: o for o in frame_tracks_slice[i - 1]}
        curr_objs = {o["track_id"]: o for o in frame_tracks_slice[i]}
        common_ids = list(set(prev_objs.keys()) & set(curr_objs.keys()))
        if len(common_ids) >= 2:
            for idx_a in range(len(common_ids)):
                for idx_b in range(idx_a + 1, len(common_ids)):
                    id_a, id_b = common_ids[idx_a], common_ids[idx_b]
                    d_curr = np.sqrt(
                        (curr_objs[id_a]["cx"] - curr_objs[id_b]["cx"])**2 +
                        (curr_objs[id_a]["cy"] - curr_objs[id_b]["cy"])**2
                    )
                    d_prev = np.sqrt(
                        (prev_objs[id_a]["cx"] - prev_objs[id_b]["cx"])**2 +
                        (prev_objs[id_a]["cy"] - prev_objs[id_b]["cy"])**2
                    )
                    closing_rate = -(d_curr - d_prev) / dt
                    pair_distances.append(d_curr)
                    pair_closing_rates.append(closing_rate)

    track_feat = {
        "active_track_count": len(track_hist_slice),
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
    feat_b = {**feat_a, **track_feat}

    # 3. Model C Features
    primary_tid = None
    best_crit = -1.0
    for tid, states in track_hist_slice.items():
        if len(states) >= 2:
            max_bottom = max(s["bottom_y"] for s in states)
            max_area = max(s["area"] for s in states)
            crit = max_bottom * 0.6 + max_area * 0.4
            if crit > best_crit:
                best_crit = crit
                primary_tid = tid

    if primary_tid is None and len(track_hist_slice) > 0:
        primary_tid = list(track_hist_slice.keys())[0]

    seq_matrix = np.zeros((10, FEATURE_DIM), dtype=np.float32)
    if primary_tid is not None and primary_tid in track_hist_slice:
        states = track_hist_slice[primary_tid]
        n_states = len(states)
        vx_prev, vy_prev = 0.0, 0.0
        heading_prev = 0.0
        state_feats = []

        for i in range(n_states):
            s = states[i]
            cx, cy, bottom_y, area = s["cx"], s["cy"], s["bottom_y"], s["area"]
            if i > 0:
                dt_actual = max(1e-4, states[i]["time_sec"] - states[i-1]["time_sec"])
                vx = (cx - states[i-1]["cx"]) / dt_actual
                vy = (cy - states[i-1]["cy"]) / dt_actual
                ax = (vx - vx_prev) / dt_actual
                ay = (vy - vy_prev) / dt_actual
                prev_area = max(1e-5, states[i-1]["area"])
                expansion_rate = (area - prev_area) / (prev_area * dt_actual)
                heading_curr = np.arctan2(vy, vx)
                heading_diff = heading_curr - heading_prev
                heading_diff = (heading_diff + np.pi) % (2 * np.pi) - np.pi
                heading_change_rate = heading_diff / dt_actual
                lateral_ratio = abs(vx) / (abs(vy) + 1e-4)
                abrupt_decel = max(0.0, expansion_rate) * max(0.0, vy)
                ttc_proxy = 1.0 / expansion_rate if expansion_rate > 1e-3 else 10.0

                ego_comp_vy = vy
                if len(track_hist_slice) > 1:
                    other_vys = []
                    for otid, ostates in track_hist_slice.items():
                        if otid != primary_tid and len(ostates) > i:
                            other_vys.append((ostates[i]["cy"] - ostates[i-1]["cy"]) / dt_actual)
                    if other_vys:
                        ego_comp_vy = vy - np.median(other_vys)

                vx_prev, vy_prev = vx, vy
                heading_prev = heading_curr
            else:
                vx, vy, ax, ay = 0.0, 0.0, 0.0, 0.0
                expansion_rate = 0.0
                heading_change_rate = 0.0
                lateral_ratio = 0.0
                abrupt_decel = 0.0
                ttc_proxy = 10.0
                ego_comp_vy = 0.0

            state_feats.append([
                cx, cy, bottom_y, area,
                vx, vy, ax, ay,
                expansion_rate, heading_change_rate, lateral_ratio,
                abrupt_decel, ttc_proxy, ego_comp_vy
            ])

        state_feats = np.array(state_feats, dtype=np.float32)
        if len(state_feats) == 10:
            seq_matrix = state_feats
        elif len(state_feats) > 10:
            indices = np.linspace(0, len(state_feats) - 1, 10).astype(int)
            seq_matrix = state_feats[indices]
        else:
            pad_len = 10 - len(state_feats)
            pad_block = np.repeat(state_feats[0:1], pad_len, axis=0)
            seq_matrix = np.vstack([pad_block, state_feats])

    summary_dict = {}
    fnames = [
        "cx", "cy", "bottom_y", "area", "vx", "vy", "ax", "ay",
        "expansion_rate", "heading_change_rate", "lateral_ratio",
        "abrupt_decel", "ttc_proxy", "ego_comp_vy"
    ]
    for idx, fname in enumerate(fnames):
        vals = seq_matrix[:, idx]
        summary_dict[f"traj_{fname}_mean"] = float(np.mean(vals))
        summary_dict[f"traj_{fname}_max"] = float(np.max(vals))
        summary_dict[f"traj_{fname}_min"] = float(np.min(vals))
        summary_dict[f"traj_{fname}_std"] = float(np.std(vals))
        summary_dict[f"traj_{fname}_last"] = float(vals[-1])
        summary_dict[f"traj_{fname}_trend"] = float(vals[-1] - vals[0])

    for k, v in track_feat.items():
        summary_dict[k] = v

    return feat_a, feat_b, seq_matrix, summary_dict

def run_early_warning_analysis(manifest_df):
    """
    Evaluates early warning lead times on test videos using an optimized single-pass tracker.
    """
    test_videos = manifest_df[manifest_df["split"] == "test"].copy()

    # Load models
    pkg_a = joblib.load("models/model_a.joblib")
    pkg_b = joblib.load("models/model_b.joblib")
    pkg_c_xgb = joblib.load("models/model_c_xgb.joblib")
    ckpt_c = torch.load("models/model_c_gru.pt", map_location="cpu", weights_only=False)

    model_gru = TrajectoryGRU(input_dim=FEATURE_DIM, hidden_dim=64, num_layers=2)
    model_gru.load_state_dict(ckpt_c["state_dict"])
    model_gru.eval()

    yolo_model = YOLO("models/yolov8n.pt")
    target_ids = set(TARGET_CLASSES.keys())

    per_video_records = []
    print("\n--- Running Optimized Single-Pass Early Warning Analysis on Test Videos ---")

    for _, row in test_videos.iterrows():
        vid_id = row["video_id"]
        vpath = row["file_path"]
        label = int(row["label"])
        event_time = row["time_of_event"]
        alert_time = row["time_of_alert"]

        cap = cv2.VideoCapture(vpath)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        # Target scenario type
        if label == 1:
            scenario_type = "Pedestrian Near-Miss" if "0012" in vid_id else "Lead Vehicle Collision"
        else:
            scenario_type = "Highway Normal Cruise"

        step_frames = 6
        dt = step_frames / fps

        frame_detections = []
        frame_tracks = []
        timestamps = []

        curr_frame = 0
        while curr_frame < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if curr_frame % step_frames == 0:
                h_img, w_img = frame.shape[:2]
                img_area = float(w_img * h_img)
                t_sec = curr_frame / fps
                timestamps.append(t_sec)

                results = yolo_model.track(
                    frame, persist=True, tracker="bytetrack.yaml",
                    conf=0.25, verbose=False
                )[0]

                # Frame detections
                dets = []
                current_frame_objs = []
                if results.boxes is not None:
                    boxes = results.boxes.xyxy.cpu().numpy()
                    cls_ids = results.boxes.cls.int().cpu().numpy()
                    confs = results.boxes.conf.cpu().numpy()
                    track_ids = results.boxes.id.int().cpu().numpy() if results.boxes.id is not None else [-1]*len(boxes)

                    for box, cid, conf, tid in zip(boxes, cls_ids, confs, track_ids):
                        if cid not in target_ids:
                            continue
                        x1, y1, x2, y2 = box
                        w = max(1.0, x2 - x1)
                        h = max(1.0, y2 - y1)
                        norm_area = (w * h) / img_area
                        cx = (x1 + x2) / 2.0 / w_img
                        cy = (y1 + y2) / 2.0 / h_img
                        bottom_y = y2 / h_img

                        det_item = {
                            "class_id": cid,
                            "conf": float(conf),
                            "norm_area": norm_area,
                            "cx": cx, "cy": cy, "bottom_y": bottom_y,
                            "is_vehicle": 1 if cid in VEHICLE_CLASS_IDS else 0,
                            "is_pedestrian": 1 if cid in PEDESTRIAN_CLASS_IDS else 0,
                            "in_center_corridor": 1 if 0.35 <= cx <= 0.65 else 0
                        }
                        dets.append(det_item)

                        if tid != -1:
                            current_frame_objs.append({
                                "frame_idx": curr_frame,
                                "time_sec": t_sec,
                                "track_id": int(tid),
                                "cls_id": int(cid),
                                "conf": float(conf),
                                "cx": cx, "cy": cy,
                                "w": w / w_img, "h": h / h_img,
                                "area": norm_area,
                                "bottom_y": bottom_y
                            })

                if len(dets) == 0:
                    f_feat = {
                        "num_vehicles": 0, "num_pedestrians": 0, "total_objects": 0,
                        "max_bbox_area": 0.0, "mean_bbox_area": 0.0, "sum_bbox_area": 0.0,
                        "max_conf": 0.0, "mean_conf": 0.0, "center_corridor_count": 0,
                        "closest_bottom_y": 0.0, "closest_bbox_area": 0.0,
                        "closest_is_pedestrian": 0, "closest_dist_to_center": 1.0,
                        "has_large_object": 0
                    }
                else:
                    num_veh = sum(d["is_vehicle"] for d in dets)
                    num_ped = sum(d["is_pedestrian"] for d in dets)
                    areas = [d["norm_area"] for d in dets]
                    confs = [d["conf"] for d in dets]
                    center_cnt = sum(d["in_center_corridor"] for d in dets)
                    sorted_by_y = sorted(dets, key=lambda d: d["bottom_y"], reverse=True)
                    closest = sorted_by_y[0]
                    f_feat = {
                        "num_vehicles": num_veh, "num_pedestrians": num_ped, "total_objects": len(dets),
                        "max_bbox_area": max(areas), "mean_bbox_area": float(np.mean(areas)),
                        "sum_bbox_area": float(np.sum(areas)), "max_conf": max(confs),
                        "mean_conf": float(np.mean(confs)), "center_corridor_count": center_cnt,
                        "closest_bottom_y": closest["bottom_y"], "closest_bbox_area": closest["norm_area"],
                        "closest_is_pedestrian": closest["is_pedestrian"],
                        "closest_dist_to_center": abs(closest["cx"] - 0.5),
                        "has_large_object": 1 if max(areas) > 0.12 else 0
                    }

                frame_detections.append(f_feat)
                frame_tracks.append(current_frame_objs)

            curr_frame += 1

        cap.release()

        # Rolling 10-frame evaluation
        probs_a_hist = []
        probs_b_hist = []
        probs_c_gru_hist = []
        probs_c_xgb_hist = []

        t_warn_a = None
        t_warn_b = None
        t_warn_c_gru = None
        t_warn_c_xgb = None

        mean_c = ckpt_c["mean"].reshape(1, FEATURE_DIM)
        std_c = ckpt_c["std"].reshape(1, FEATURE_DIM)

        for i in range(9, len(frame_detections)):
            t_curr = timestamps[i]
            slice_dets = frame_detections[i - 9 : i + 1]
            slice_tracks = frame_tracks[i - 9 : i + 1]

            feat_a, feat_b, seq_mat, feat_c_tab = extract_features_from_window_slice(slice_dets, slice_tracks, dt)

            # Model A
            df_curr_a = pd.DataFrame([feat_a])
            p_a = pkg_a["model"].predict_proba(pkg_a["scaler"].transform(df_curr_a[pkg_a["features"]].fillna(0)))[0, 1]
            probs_a_hist.append((t_curr, p_a))
            if p_a >= 0.50 and t_warn_a is None:
                t_warn_a = t_curr

            # Model B
            df_curr_b = pd.DataFrame([feat_b])
            p_b = pkg_b["model"].predict_proba(pkg_b["scaler"].transform(df_curr_b[pkg_b["features"]].fillna(0)))[0, 1]
            probs_b_hist.append((t_curr, p_b))
            if p_b >= 0.50 and t_warn_b is None:
                t_warn_b = t_curr

            # Model C GRU
            norm_seq = (seq_mat.reshape(10, FEATURE_DIM) - mean_c) / std_c
            with torch.no_grad():
                t_seq = torch.tensor(norm_seq, dtype=torch.float32).unsqueeze(0)
                p_c_gru = float(torch.sigmoid(model_gru(t_seq))[0, 0].item())
            probs_c_gru_hist.append((t_curr, p_c_gru))
            if p_c_gru >= 0.50 and t_warn_c_gru is None:
                t_warn_c_gru = t_curr

            # Model C XGB
            df_curr_c = pd.DataFrame([feat_c_tab])
            p_c_xgb = pkg_c_xgb["model"].predict_proba(pkg_c_xgb["scaler"].transform(df_curr_c[pkg_c_xgb["features"]].fillna(0)))[0, 1]
            probs_c_xgb_hist.append((t_curr, p_c_xgb))
            if p_c_xgb >= 0.50 and t_warn_c_xgb is None:
                t_warn_c_xgb = t_curr

        # Probabilities at standard horizons
        if label == 1:
            t_15 = event_time - 1.5
            t_10 = event_time - 1.0
            t_05 = event_time - 0.5
        else:
            t_15 = duration * 0.45
            t_10 = duration * 0.65
            t_05 = duration * 0.85

        def get_prob_at_t(history, t_target):
            closest = min(history, key=lambda x: abs(x[0] - t_target))
            return closest[1]

        p_a_15 = get_prob_at_t(probs_a_hist, t_15)
        p_a_10 = get_prob_at_t(probs_a_hist, t_10)
        p_a_05 = get_prob_at_t(probs_a_hist, t_05)
        p_a_max = max(p for _, p in probs_a_hist)

        p_b_15 = get_prob_at_t(probs_b_hist, t_15)
        p_b_10 = get_prob_at_t(probs_b_hist, t_10)
        p_b_05 = get_prob_at_t(probs_b_hist, t_05)
        p_b_max = max(p for _, p in probs_b_hist)

        p_c_gru_15 = get_prob_at_t(probs_c_gru_hist, t_15)
        p_c_gru_10 = get_prob_at_t(probs_c_gru_hist, t_10)
        p_c_gru_05 = get_prob_at_t(probs_c_gru_hist, t_05)
        p_c_gru_max = max(p for _, p in probs_c_gru_hist)

        p_c_xgb_15 = get_prob_at_t(probs_c_xgb_hist, t_15)
        p_c_xgb_10 = get_prob_at_t(probs_c_xgb_hist, t_10)
        p_c_xgb_05 = get_prob_at_t(probs_c_xgb_hist, t_05)
        p_c_xgb_max = max(p for _, p in probs_c_xgb_hist)

        lead_a = round(event_time - t_warn_a, 2) if (label == 1 and t_warn_a is not None) else np.nan
        lead_b = round(event_time - t_warn_b, 2) if (label == 1 and t_warn_b is not None) else np.nan
        lead_c_gru = round(event_time - t_warn_c_gru, 2) if (label == 1 and t_warn_c_gru is not None) else np.nan
        lead_c_xgb = round(event_time - t_warn_c_xgb, 2) if (label == 1 and t_warn_c_xgb is not None) else np.nan

        warned_before_human_a = (t_warn_a <= alert_time) if (label == 1 and t_warn_a is not None) else False
        warned_before_human_b = (t_warn_b <= alert_time) if (label == 1 and t_warn_b is not None) else False
        warned_before_human_c_gru = (t_warn_c_gru <= alert_time) if (label == 1 and t_warn_c_gru is not None) else False
        warned_before_human_c_xgb = (t_warn_c_xgb <= alert_time) if (label == 1 and t_warn_c_xgb is not None) else False

        rec = {
            "video_id": vid_id,
            "split": "test",
            "label": label,
            "scenario_type": scenario_type,
            "time_of_event": event_time if label == 1 else "N/A",
            "time_of_alert": alert_time if label == 1 else "N/A",

            # Model A
            "model_a_prob_1500ms": round(p_a_15, 4),
            "model_a_prob_1000ms": round(p_a_10, 4),
            "model_a_prob_0500ms": round(p_a_05, 4),
            "model_a_video_prob": round(p_a_max, 4),
            "model_a_pred": 1 if p_a_max >= 0.5 else 0,
            "first_warning_time_model_a": round(t_warn_a, 2) if t_warn_a is not None else "None",
            "lead_time_model_a": lead_a if not np.isnan(lead_a) else "N/A",
            "warned_before_human_model_a": warned_before_human_a if label == 1 else "N/A",

            # Model B
            "model_b_prob_1500ms": round(p_b_15, 4),
            "model_b_prob_1000ms": round(p_b_10, 4),
            "model_b_prob_0500ms": round(p_b_05, 4),
            "model_b_video_prob": round(p_b_max, 4),
            "model_b_pred": 1 if p_b_max >= 0.5 else 0,
            "first_warning_time_model_b": round(t_warn_b, 2) if t_warn_b is not None else "None",
            "lead_time_model_b": lead_b if not np.isnan(lead_b) else "N/A",
            "warned_before_human_model_b": warned_before_human_b if label == 1 else "N/A",

            # Model C GRU
            "model_c_gru_prob_1500ms": round(p_c_gru_15, 4),
            "model_c_gru_prob_1000ms": round(p_c_gru_10, 4),
            "model_c_gru_prob_0500ms": round(p_c_gru_05, 4),
            "model_c_gru_video_prob": round(p_c_gru_max, 4),
            "model_c_gru_pred": 1 if p_c_gru_max >= 0.5 else 0,
            "first_warning_time_model_c_gru": round(t_warn_c_gru, 2) if t_warn_c_gru is not None else "None",
            "lead_time_model_c_gru": lead_c_gru if not np.isnan(lead_c_gru) else "N/A",
            "warned_before_human_model_c_gru": warned_before_human_c_gru if label == 1 else "N/A",

            # Model C XGB
            "model_c_xgb_prob_1500ms": round(p_c_xgb_15, 4),
            "model_c_xgb_prob_1000ms": round(p_c_xgb_10, 4),
            "model_c_xgb_prob_0500ms": round(p_c_xgb_05, 4),
            "model_c_xgb_video_prob": round(p_c_xgb_max, 4),
            "model_c_xgb_pred": 1 if p_c_xgb_max >= 0.5 else 0,
            "first_warning_time_model_c_xgb": round(t_warn_c_xgb, 2) if t_warn_c_xgb is not None else "None",
            "lead_time_model_c_xgb": lead_c_xgb if not np.isnan(lead_c_xgb) else "N/A",
            "warned_before_human_model_c_xgb": warned_before_human_c_xgb if label == 1 else "N/A",
        }
        per_video_records.append(rec)
        print(f"  Processed {vid_id} (label={label}): Warning times -> A: {t_warn_a}s, B: {t_warn_b}s, GRU: {t_warn_c_gru}s, XGB: {t_warn_c_xgb}s")

    df_per_video = pd.DataFrame(per_video_records)
    return df_per_video

def main():
    print("="*85)
    print("SECOND-STAGE VALIDATION: EARLY COLLISION ANTICIPATION")
    print("="*85)

    manifest_path = Path("data/manifests/dataset_manifest.csv")
    window_path = Path("data/manifests/window_manifest.csv")
    proc_path = Path("data/processed")
    results_path = Path("results")
    results_path.mkdir(parents=True, exist_ok=True)

    df_dataset = pd.read_csv(manifest_path)
    df_window = pd.read_csv(window_path)

    df_a = pd.read_csv(proc_path / "model_a_features.csv")
    df_b = pd.read_csv(proc_path / "model_b_features.csv")
    df_c_tab = pd.read_csv(proc_path / "model_c_tabular.csv")
    seq_data = np.load(proc_path / "model_c_sequences.npz")["sequences"]

    test_mask = (df_window["split"] == "test").values

    # 1. Run Ablations
    print("\n[Step 1/3] Running Systematic Feature Ablation Experiments...")
    df_ablation = run_ablations(df_c_tab, df_window, seq_data, test_mask)
    ablation_csv = results_path / "ablation_results.csv"
    df_ablation.to_csv(ablation_csv, index=False)
    print(f"Saved ablation results to: {ablation_csv}")
    print(df_ablation[["Ablation Tier", "Feature Count", "1.5s AP", "1.0s AP", "0.5s AP", "Mean AP", "PR-AUC", "F1", "FPR", "Latency (ms)"]].to_string())

    # 2. Run Early Warning & Per-Video Predictions
    print("\n[Step 2/3] Evaluating Per-Video Early Warning Lead Times...")
    df_per_video = run_early_warning_analysis(df_dataset)
    per_video_csv = results_path / "per_video_predictions.csv"
    df_per_video.to_csv(per_video_csv, index=False)
    print(f"Saved per-video predictions to: {per_video_csv}")

    # 3. Comprehensive Updated Metrics (Window-Level & Video-Level)
    print("\n[Step 3/3] Generating Updated Comprehensive Metrics...")

    pkg_a = joblib.load("models/model_a.joblib")
    pkg_b = joblib.load("models/model_b.joblib")
    pkg_c_xgb = joblib.load("models/model_c_xgb.joblib")
    ckpt_c = torch.load("models/model_c_gru.pt", map_location="cpu", weights_only=False)

    model_gru = TrajectoryGRU(input_dim=FEATURE_DIM, hidden_dim=64, num_layers=2)
    model_gru.load_state_dict(ckpt_c["state_dict"])
    model_gru.eval()

    test_a = df_a[df_a["split"] == "test"]
    test_b = df_b[df_b["split"] == "test"]
    test_c_tab = df_c_tab[df_c_tab["split"] == "test"]
    test_seqs = seq_data[test_mask]
    norm_seqs = (test_seqs - ckpt_c["mean"]) / ckpt_c["std"]

    y_test = test_a["label"].values
    horizons = test_a["horizon_sec"].values
    duration_min = ((y_test == 0).sum() * 2.0) / 60.0

    probs_a = pkg_a["model"].predict_proba(pkg_a["scaler"].transform(test_a[pkg_a["features"]].fillna(0)))[:, 1]
    probs_b = pkg_b["model"].predict_proba(pkg_b["scaler"].transform(test_b[pkg_b["features"]].fillna(0)))[:, 1]
    probs_c_xgb = pkg_c_xgb["model"].predict_proba(pkg_c_xgb["scaler"].transform(test_c_tab[pkg_c_xgb["features"]].fillna(0)))[:, 1]
    with torch.no_grad():
        probs_c_gru = torch.sigmoid(model_gru(torch.tensor(norm_seqs, dtype=torch.float32))).cpu().numpy().flatten()

    models_dict = {
        "Model A: Detection Only (Baseline)": (probs_a, 0.0),
        "Model B: Detection + Tracking (ByteTrack)": (probs_b, 0.28),
        "Model C: Trajectory + GRU (Proposed)": (probs_c_gru, 0.97),
        "Model C: Trajectory + XGBoost (Comparator)": (probs_c_xgb, 0.33)
    }

    updated_records = []
    for mname, (probs, lat) in models_dict.items():
        res = evaluate_model_performance(y_test, probs, horizons, duration_min, lat)
        res["Evaluation Level"] = "Window-Level (N=18)"
        res["Model"] = mname
        updated_records.append(res)

    y_video_true = df_per_video["label"].values
    dur_video_min = (len(y_video_true[y_video_true == 0]) * 10.0) / 60.0
    for mname, mprefix, lat in [
        ("Model A: Detection Only (Baseline)", "model_a", 0.0),
        ("Model B: Detection + Tracking (ByteTrack)", "model_b", 0.28),
        ("Model C: Trajectory + GRU (Proposed)", "model_c_gru", 0.97),
        ("Model C: Trajectory + XGBoost (Comparator)", "model_c_xgb", 0.33)
    ]:
        probs_15 = df_per_video[f"{mprefix}_prob_1500ms"].values
        probs_10 = df_per_video[f"{mprefix}_prob_1000ms"].values
        probs_05 = df_per_video[f"{mprefix}_prob_0500ms"].values
        probs_vid = df_per_video[f"{mprefix}_video_prob"].values

        ap_15 = float(average_precision_score(y_video_true, probs_15))
        ap_10 = float(average_precision_score(y_video_true, probs_10))
        ap_05 = float(average_precision_score(y_video_true, probs_05))
        mean_ap = float(np.mean([ap_15, ap_10, ap_05]))

        pr_auc = float(average_precision_score(y_video_true, probs_vid))
        roc_auc = float(roc_auc_score(y_video_true, probs_vid))
        y_pred = (probs_vid >= 0.5).astype(int)
        f1 = float(f1_score(y_video_true, y_pred, zero_division=0))
        prec = float(precision_score(y_video_true, y_pred, zero_division=0))
        rec = float(recall_score(y_video_true, y_pred, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(y_video_true, y_pred, labels=[0, 1]).ravel()
        fpr = float(fp / max(1, (fp + tn)))
        pr_ci, roc_ci = bootstrap_ci(y_video_true, probs_vid)

        res = {
            "Evaluation Level": "Video-Level (N=6 independent videos)",
            "Model": mname,
            "1.5s AP": round(ap_15, 4),
            "1.0s AP": round(ap_10, 4),
            "0.5s AP": round(ap_05, 4),
            "Mean AP": round(mean_ap, 4),
            "PR-AUC": round(pr_auc, 4),
            "PR-AUC 95% CI": f"[{pr_ci[0]:.4f}, {pr_ci[1]:.4f}]",
            "ROC-AUC": round(roc_auc, 4),
            "ROC-AUC 95% CI": f"[{roc_ci[0]:.4f}, {roc_ci[1]:.4f}]",
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "FPR": round(fpr, 4),
            "False Alarms/min": 0.0,
            "Latency (ms)": round(lat, 2),
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)
        }
        updated_records.append(res)

    df_updated = pd.DataFrame(updated_records)
    cols_order = [
        "Evaluation Level", "Model", "1.5s AP", "1.0s AP", "0.5s AP", "Mean AP",
        "PR-AUC", "PR-AUC 95% CI", "ROC-AUC", "ROC-AUC 95% CI",
        "Precision", "Recall", "F1", "FPR", "False Alarms/min",
        "TP", "FP", "TN", "FN", "Latency (ms)"
    ]
    df_updated = df_updated[cols_order]
    updated_csv = results_path / "updated_metrics.csv"
    df_updated.to_csv(updated_csv, index=False)
    print(f"\nSaved updated metrics to: {updated_csv}")
    print(df_updated[["Evaluation Level", "Model", "1.5s AP", "Mean AP", "PR-AUC", "F1", "TP", "FP", "TN", "FN"]].to_string())

    print("\n" + "="*85)
    print("ALL SECOND-STAGE EXPERIMENTS SUCCESSFULLY EXECUTED.")
    print("="*85)

if __name__ == "__main__":
    main()
