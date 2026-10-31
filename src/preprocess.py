"""
preprocess.py
Preprocesses dashcam videos into standardized temporal anticipation windows.
Extracts pre-collision windows ending at 1.5s, 1.0s, and 0.5s before the event for positive videos,
and comparable non-collision driving windows for negative videos.
Ensures zero data leakage across train/val/test splits.
"""

import os
import argparse
import pandas as pd
import numpy as np
import cv2
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Extract temporal analysis windows from dashcam dataset")
    parser.add_argument("--manifest_path", type=str, default="data/manifests/dataset_manifest.csv", help="Dataset manifest CSV")
    parser.add_argument("--output_manifest", type=str, default="data/manifests/window_manifest.csv", help="Window manifest output path")
    parser.add_argument("--history_sec", type=float, default=2.0, help="Duration of temporal observation history in seconds")
    parser.add_argument("--horizons", nargs="+", type=float, default=[1.5, 1.0, 0.5], help="Prediction horizons in seconds before collision")
    parser.add_argument("--sample_fps", type=float, default=5.0, help="Sample frame rate for feature processing (saves computation & memory)")
    return parser.parse_args()

def extract_windows_for_dataset(manifest_df, horizons, history_sec, sample_fps):
    """
    Generate window metadata ensuring strict split isolation and multi-horizon coverage.
    """
    records = []

    for _, row in manifest_df.iterrows():
        vid_id = row["video_id"]
        label = int(row["label"])
        split = row["split"]
        vpath = row["file_path"]
        event_time = row["time_of_event"]
        alert_time = row["time_of_alert"]

        cap = cv2.VideoCapture(vpath)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()

        if label == 1:
            # Positive video: generate windows for each anticipation horizon
            # Event time is known
            if pd.isna(event_time) or event_time <= (history_sec + max(horizons)):
                event_time = duration - 1.0  # fallback safe event point

            for h in horizons:
                end_sec = event_time - h
                start_sec = max(0.0, end_sec - history_sec)
                
                start_frame = int(round(start_sec * fps))
                end_frame = int(round(end_sec * fps))
                
                win_id = f"{vid_id}_h{int(h*1000):04d}ms"
                records.append({
                    "window_id": win_id,
                    "video_id": vid_id,
                    "split": split,
                    "label": 1,
                    "horizon_sec": h,
                    "start_sec": round(start_sec, 3),
                    "end_sec": round(end_sec, 3),
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "time_of_event": event_time,
                    "time_of_alert": alert_time,
                    "file_path": vpath,
                    "duration_sec": duration,
                    "video_fps": fps,
                    "target_sample_fps": sample_fps
                })
        else:
            # Negative video: generate 3 comparable non-event windows
            # e.g., sampled at equivalent temporal offsets
            anchor_points = [duration * 0.45, duration * 0.65, duration * 0.85]
            for idx, end_sec in enumerate(anchor_points):
                h_proxy = horizons[idx % len(horizons)]
                start_sec = max(0.0, end_sec - history_sec)
                start_frame = int(round(start_sec * fps))
                end_frame = int(round(end_sec * fps))

                win_id = f"{vid_id}_w{idx+1}_h{int(h_proxy*1000):04d}ms"
                records.append({
                    "window_id": win_id,
                    "video_id": vid_id,
                    "split": split,
                    "label": 0,
                    "horizon_sec": h_proxy,
                    "start_sec": round(start_sec, 3),
                    "end_sec": round(end_sec, 3),
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "time_of_event": np.nan,
                    "time_of_alert": np.nan,
                    "file_path": vpath,
                    "duration_sec": duration,
                    "video_fps": fps,
                    "target_sample_fps": sample_fps
                })

    window_df = pd.DataFrame(records)
    return window_df

def main():
    args = parse_args()
    manifest_path = Path(args.manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Run download_dataset.py first.")

    print(f"Reading dataset manifest from: {manifest_path}")
    manifest_df = pd.read_csv(manifest_path)

    print(f"Extracting temporal windows (history={args.history_sec}s, horizons={args.horizons})...")
    window_df = extract_windows_for_dataset(manifest_df, args.horizons, args.history_sec, args.sample_fps)

    out_path = Path(args.output_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    window_df.to_csv(out_path, index=False)
    print(f"Generated {len(window_df)} total temporal windows.")

    print("\nWindow Breakdown by Split and Label:")
    for split in ["train", "val", "test"]:
        sub = window_df[window_df["split"] == split]
        pos = (sub["label"] == 1).sum()
        neg = (sub["label"] == 0).sum()
        print(f"  [{split.upper()}]: Total={len(sub)}, Pos={pos}, Neg={neg}")

    print("\nHorizon Breakdown for Positive Anticipation:")
    pos_sub = window_df[window_df["label"] == 1]
    for h in sorted(pos_sub["horizon_sec"].unique()):
        cnt = (pos_sub["horizon_sec"] == h).sum()
        print(f"  Horizon {h}s before event: {cnt} windows")

    print(f"\nWindow manifest successfully saved to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
