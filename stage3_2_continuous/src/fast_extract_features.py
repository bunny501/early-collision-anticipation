"""
fast_extract_features.py
High-speed video-batched feature extractor for Stage 3.2.
1. Reuses 300 windows from Stage 3 cache.
2. For each remaining video, opens the video ONCE, seeks to min_frame ONCE,
   and reads the 15-18 frames covering all 3 horizons sequentially.
3. Runs YOLOv8n + ByteTrack once through the frames and slices the 3 windows.
4. Generates complete features for all 1,200 windows in ~4-5 minutes.
"""

import os
import sys
import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import torch
torch.set_num_threads(8)
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
BASE_DIR = Path(__file__).resolve().parent.parent
STAGE3_DIR = ROOT_DIR / "stage3_real_world"

TARGET_CLASSES = {0: "pedestrian", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}
PEDESTRIAN_CLASS_IDS = {0}
FEATURE_DIM = 14
SEQUENCE_LENGTH = 10

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

def extract_window_features(frame_detections, track_history, ego_motions):
    # 1. Model A Features
    feat_a = {}
    all_objs = [d for fd in frame_detections for d in fd]

    feat_a["mean_vehicle_count"] = float(np.mean([sum(d["is_vehicle"] for d in fd) for fd in frame_detections])) if frame_detections else 0.0
    feat_a["max_vehicle_count"] = float(np.max([sum(d["is_vehicle"] for d in fd) for fd in frame_detections])) if frame_detections else 0.0
    feat_a["mean_pedestrian_count"] = float(np.mean([sum(d["is_pedestrian"] for d in fd) for fd in frame_detections])) if frame_detections else 0.0
    feat_a["max_pedestrian_count"] = float(np.max([sum(d["is_pedestrian"] for d in fd) for fd in frame_detections])) if frame_detections else 0.0
    feat_a["total_objects_mean"] = float(np.mean([len(fd) for fd in frame_detections])) if frame_detections else 0.0

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

    last_dets = frame_detections[-1] if frame_detections else []
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

    feat_a["center_corridor_count_mean"] = float(np.mean([sum(1 for d in fd if d["dist_to_center"] < 0.15) for fd in frame_detections])) if frame_detections else 0.0

    # 2. Model B Features
    feat_b = {**feat_a}
    track_durations = [len(s) for s in track_history.values()]
    feat_b["num_active_tracks"] = float(len(track_history))
    feat_b["mean_track_duration"] = float(np.mean(track_durations)) if track_durations else 0.0
    feat_b["max_track_duration"] = float(np.max(track_durations)) if track_durations else 0.0

    velocities_x, velocities_y, speeds, expansion_rates = [], [], [], []
    for tid, states in track_history.items():
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

    # 3. Model C Features
    primary_tid = None
    best_crit = -1.0
    for tid, states in track_history.items():
        if len(states) >= 2:
            crit = max(s["bottom_y"] for s in states) * 0.6 + max(s["norm_area"] for s in states) * 0.4
            if crit > best_crit:
                best_crit = crit
                primary_tid = tid

    if primary_tid is None and len(track_history) > 0:
        primary_tid = list(track_history.keys())[0]

    seq_mat = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
    feat_c_tab = {**feat_b}

    if primary_tid is not None and primary_tid in track_history:
        states = track_history[primary_tid]
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
            ego_dx, ego_dy = ego_motions[i] if i < len(ego_motions) else (0.0, 0.0)

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
        feat_c_tab["ego_motion_dy_mean"] = float(np.mean([em[1] for em in ego_motions]))
    else:
        for k in [
            "traj_cx_mean", "traj_cx_trend", "traj_cy_mean", "traj_cy_trend",
            "traj_bottom_y_max", "traj_bottom_y_mean", "traj_area_max", "traj_area_mean",
            "traj_vx_mean", "traj_vx_max", "traj_vy_mean", "traj_vy_max",
            "traj_ax_mean", "traj_ay_mean", "traj_ay_max",
            "traj_exp_mean", "traj_exp_max", "traj_min_ttc", "ego_motion_dy_mean"
        ]:
            feat_c_tab[k] = 0.0

    return feat_a, feat_b, feat_c_tab, seq_mat

def main():
    parser = argparse.ArgumentParser(description="Fast video-batched feature extractor")
    parser.add_argument("--manifest_path", type=str, default="data/manifests/window_manifest.csv")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    parser.add_argument("--step_frames", type=int, default=6)
    args = parser.parse_args()

    df_manifest = pd.read_csv(BASE_DIR / args.manifest_path)
    proc_dir = BASE_DIR / args.out_dir
    proc_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Stage 3 pre-extracted cache
    old_proc = STAGE3_DIR / "data" / "processed"
    cached_a, cached_b, cached_c, cached_seq = {}, {}, {}, {}
    if (old_proc / "model_a_features.csv").exists():
        old_df_a = pd.read_csv(old_proc / "model_a_features.csv")
        old_df_b = pd.read_csv(old_proc / "model_b_features.csv")
        old_df_c = pd.read_csv(old_proc / "model_c_tabular.csv")
        old_seq = np.load(old_proc / "model_c_sequences.npz")["sequences"]

        meta_drop = ["window_id", "video_id", "split", "label", "horizon_sec", "scene", "weather"]
        for i, wid in enumerate(old_df_a["window_id"]):
            cached_a[wid] = old_df_a.iloc[i].drop(labels=[c for c in meta_drop if c in old_df_a.columns]).to_dict()
            cached_b[wid] = old_df_b.iloc[i].drop(labels=[c for c in meta_drop if c in old_df_b.columns]).to_dict()
            cached_c[wid] = old_df_c.iloc[i].drop(labels=[c for c in meta_drop if c in old_df_c.columns]).to_dict()
            cached_seq[wid] = old_seq[i]
        print(f"Loaded {len(cached_a)} pre-extracted windows from Stage 3 cache.", flush=True)

    # 2. Identify remaining windows to extract
    remaining_manifest = df_manifest[~df_manifest["window_id"].isin(cached_a)].copy()
    unique_vids = remaining_manifest["video_id"].unique()
    print(f"Total windows: {len(df_manifest)}. Cached: {len(cached_a)}. Remaining to extract: {len(remaining_manifest)} across {len(unique_vids)} videos.", flush=True)

    # Load YOLO
    model_path = BASE_DIR / "models" / "yolov8n.pt"
    model = YOLO(str(model_path))

    extracted_a, extracted_b, extracted_c, extracted_seq = {}, {}, {}, {}

    target_ids = set(TARGET_CLASSES.keys())

    print("\n--- Running Fast Video-Level Batch Extraction ---", flush=True)
    for vid in tqdm(unique_vids, desc="Processing videos"):
        vid_wins = remaining_manifest[remaining_manifest["video_id"] == vid].sort_values("start_frame")
        vpath = vid_wins["file_path"].values[0]

        cap = cv2.VideoCapture(vpath)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        min_f = int(vid_wins["start_frame"].min())
        max_f = int(vid_wins["end_frame"].max())

        # Single seek
        cap.set(cv2.CAP_PROP_POS_FRAMES, min_f)

        sampled_frames = []
        sampled_fidxs = []
        curr_f = min_f

        # Read frames sequentially
        while curr_f <= max_f:
            ret, frame = cap.read()
            if not ret:
                break
            if (curr_f - min_f) % args.step_frames == 0:
                sampled_frames.append(frame)
                sampled_fidxs.append(curr_f)
            curr_f += 1
        cap.release()

        # Run YOLO + ByteTrack across the sequence
        frame_detections = []
        track_history = defaultdict(list)
        prev_gray = None
        ego_motions = []

        for frame, f_idx in zip(sampled_frames, sampled_fidxs):
            h_img, w_img = frame.shape[:2]
            img_area = float(w_img * h_img)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            results = model.track(
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
                        "frame_idx": f_idx,
                        "time_sec": f_idx / fps,
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
                        track_history[int(tid)].append(det_item)

            frame_detections.append(curr_dets)

            if prev_gray is not None:
                ego_dx, ego_dy = compute_background_flow(prev_gray, gray, boxes_xyxy)
            else:
                ego_dx, ego_dy = 0.0, 0.0
            ego_motions.append((ego_dx, ego_dy))
            prev_gray = gray

        # Now slice each of the 3 windows from this single pass!
        for _, w_row in vid_wins.iterrows():
            wid = w_row["window_id"]
            sf = int(w_row["start_frame"])
            ef = int(w_row["end_frame"])

            # Find matching frame indices
            slice_indices = [i for i, f_idx in enumerate(sampled_fidxs) if sf <= f_idx <= ef]
            if len(slice_indices) > SEQUENCE_LENGTH:
                slice_indices = slice_indices[:SEQUENCE_LENGTH]

            sub_dets = [frame_detections[i] for i in slice_indices]
            sub_ego = [ego_motions[i] for i in slice_indices]

            # Sub-track history
            sub_track_history = defaultdict(list)
            for fd in sub_dets:
                for d in fd:
                    if d["track_id"] >= 0:
                        sub_track_history[d["track_id"]].append(d)

            fa, fb, fc, seq = extract_window_features(sub_dets, sub_track_history, sub_ego)
            extracted_a[wid] = fa
            extracted_b[wid] = fb
            extracted_c[wid] = fc
            extracted_seq[wid] = seq

    # 3. Assemble full 1,200 window datasets in manifest order
    records_a = []
    records_b = []
    records_c_tab = []
    records_center = []
    sequences = []

    for idx, row in df_manifest.iterrows():
        wid = row["window_id"]
        meta = {
            "window_id": wid,
            "video_id": str(row["video_id"]),
            "split": row["split"],
            "label": int(row["label"]),
            "horizon_sec": float(row["horizon_sec"]),
            "scene": row.get("scene", "Unknown"),
            "weather": row.get("weather", "Unknown")
        }

        if wid in cached_a:
            fa = cached_a[wid]
            fb = cached_b[wid]
            fc = cached_c[wid]
            seq = cached_seq[wid]
        else:
            fa = extracted_a[wid]
            fb = extracted_b[wid]
            fc = extracted_c[wid]
            seq = extracted_seq[wid]

        records_a.append({**meta, **fa})
        records_b.append({**meta, **fb})
        records_c_tab.append({**meta, **fc})
        records_center.append({
            **meta,
            "closest_dist_to_center_mean": fa["closest_dist_to_center_mean"],
            "closest_dist_to_center_min": fa["closest_dist_to_center_min"],
            "center_corridor_count_mean": fa["center_corridor_count_mean"]
        })
        sequences.append(seq)

    df_a = pd.DataFrame(records_a)
    df_b = pd.DataFrame(records_b)
    df_c_tab = pd.DataFrame(records_c_tab)
    df_center = pd.DataFrame(records_center)
    seq_arr = np.array(sequences, dtype=np.float32)

    df_a.to_csv(proc_dir / "model_a_features.csv", index=False)
    df_b.to_csv(proc_dir / "model_b_features.csv", index=False)
    df_c_tab.to_csv(proc_dir / "model_c_tabular.csv", index=False)
    df_center.to_csv(proc_dir / "center_features.csv", index=False)
    np.savez_compressed(proc_dir / "model_c_sequences.npz", sequences=seq_arr)

    print(f"\n[SUCCESS] Saved 1,200 window features to {proc_dir.resolve()}!", flush=True)

if __name__ == "__main__":
    main()
