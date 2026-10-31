"""
download_nexar_subset.py
Downloads a balanced, reproducible subset of 50 positive and 50 negative
real-world dashcam videos from the official Nexar Collision Prediction Dataset.

Enforces video-level splitting (70% train, 14% val, 16% test) prior to any
temporal window generation, ensuring zero cross-split leakage.
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import sys
import shutil
import argparse
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from huggingface_hub import hf_hub_download, get_token
from sklearn.model_selection import train_test_split

REPO_ID = "nexar-ai/nexar_collision_prediction"

def get_secure_token():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or get_token()
    if not token:
        print("[ERROR] Hugging Face token not found in environment (HF_TOKEN).", flush=True)
        print("Please configure $env:HF_TOKEN in PowerShell before executing.", flush=True)
        sys.exit(1)
    return token

def main():
    parser = argparse.ArgumentParser(description="Download Nexar dataset subset")
    parser.add_argument("--num_pos", type=int, default=50, help="Number of positive videos")
    parser.add_argument("--num_neg", type=int, default=50, help="Number of negative videos")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--out_dir", type=str, default="data/raw", help="Target raw video directory")
    parser.add_argument("--split_manifest", type=str, default="split_manifest.csv", help="Output split manifest")
    args = parser.parse_args()

    token = get_secure_token()
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / args.out_dir
    pos_dir = raw_dir / "positive"
    neg_dir = raw_dir / "negative"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    meta_dir = base_dir / "inspect_meta"
    pos_meta_path = meta_dir / "train_positive_metadata.csv"
    neg_meta_path = meta_dir / "train_negative_metadata.csv"

    if not pos_meta_path.exists() or not neg_meta_path.exists():
        print(f"[ERROR] Metadata not found in {meta_dir}. Run metadata download first.", flush=True)
        sys.exit(1)

    df_pos_all = pd.read_csv(pos_meta_path)
    df_neg_all = pd.read_csv(neg_meta_path)

    print(f"Total available upstream: {len(df_pos_all)} positive, {len(df_neg_all)} negative clips.", flush=True)

    # Sample balanced subset
    df_pos = df_pos_all.sample(n=args.num_pos, random_state=args.seed).copy()
    df_neg = df_neg_all.sample(n=args.num_neg, random_state=args.seed).copy()

    df_pos["label"] = 1
    df_neg["label"] = 0

    # Clean video IDs
    df_pos["video_id"] = df_pos["file_name"].str.replace(".mp4", "")
    df_neg["video_id"] = df_neg["file_name"].str.replace(".mp4", "")

    # Video-level split: 70% train (35 pos, 35 neg), 14% val (7 pos, 7 neg), 16% test (8 pos, 8 neg)
    def assign_splits(sub_df, seed):
        train_sub, temp_sub = train_test_split(sub_df, test_size=0.30, random_state=seed)
        val_sub, test_sub = train_test_split(temp_sub, test_size=(16.0 / 30.0), random_state=seed)
        
        split_dict = {}
        for vid in train_sub["video_id"]:
            split_dict[vid] = "train"
        for vid in val_sub["video_id"]:
            split_dict[vid] = "val"
        for vid in test_sub["video_id"]:
            split_dict[vid] = "test"
        return split_dict

    pos_splits = assign_splits(df_pos, args.seed)
    neg_splits = assign_splits(df_neg, args.seed)

    df_pos["split"] = df_pos["video_id"].map(pos_splits)
    df_neg["split"] = df_neg["video_id"].map(neg_splits)

    combined_df = pd.concat([df_pos, df_neg], ignore_index=True)
    combined_df["dataset_source"] = "nexar-ai/nexar_collision_prediction"

    print("\nVideo-Level Split Summary:", flush=True)
    print(pd.crosstab(combined_df["split"], combined_df["label"], margins=True), flush=True)

    # Download positive clips
    print(f"\n--- Ingesting {len(df_pos)} Positive Real-World Videos ---", flush=True)
    pos_records = []
    for idx, row in df_pos.iterrows():
        fname = row["file_name"]
        vid_id = row["video_id"]
        subpath = f"train/positive/{fname}"
        local_target = pos_dir / fname

        if not local_target.exists() or local_target.stat().st_size == 0:
            print(f"[{len(pos_records)+1}/{len(df_pos)}] Downloading positive video {fname}...", flush=True)
            try:
                cached_path = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=subpath,
                    repo_type="dataset",
                    token=token
                )
                shutil.copy(cached_path, local_target)
            except Exception as e:
                print(f"[FAIL] Download failed for {subpath}: {e}", flush=True)
                continue
        else:
            print(f"[{len(pos_records)+1}/{len(df_pos)}] Cached: {fname}", flush=True)

        # Verify integrity
        cap = cv2.VideoCapture(str(local_target))
        if not cap.isOpened():
            print(f"[WARNING] File corrupt: {local_target}", flush=True)
            cap.release()
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        pos_records.append({
            "video_id": vid_id,
            "filename": fname,
            "label": 1,
            "split": row["split"],
            "time_of_event": row["time_of_event"],
            "time_of_alert": row["time_of_alert"],
            "light_conditions": row["light_conditions"],
            "weather": row["weather"],
            "scene": row["scene"],
            "fps": round(fps, 2),
            "frame_count": n_frames,
            "duration_sec": round(n_frames / max(1.0, fps), 2),
            "resolution": f"{w}x{h}",
            "file_path": str(local_target.resolve()),
            "dataset_source": "nexar-ai/nexar_collision_prediction"
        })

    # Download negative clips
    print(f"\n--- Ingesting {len(df_neg)} Negative Real-World Videos ---", flush=True)
    neg_records = []
    for idx, row in df_neg.iterrows():
        fname = row["file_name"]
        vid_id = row["video_id"]
        subpath = f"train/negative/{fname}"
        local_target = neg_dir / fname

        if not local_target.exists() or local_target.stat().st_size == 0:
            print(f"[{len(neg_records)+1}/{len(df_neg)}] Downloading negative video {fname}...", flush=True)
            try:
                cached_path = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=subpath,
                    repo_type="dataset",
                    token=token
                )
                shutil.copy(cached_path, local_target)
            except Exception as e:
                print(f"[FAIL] Download failed for {subpath}: {e}", flush=True)
                continue
        else:
            print(f"[{len(neg_records)+1}/{len(df_neg)}] Cached: {fname}", flush=True)

        # Verify integrity
        cap = cv2.VideoCapture(str(local_target))
        if not cap.isOpened():
            print(f"[WARNING] File corrupt: {local_target}", flush=True)
            cap.release()
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        neg_records.append({
            "video_id": vid_id,
            "filename": fname,
            "label": 0,
            "split": row["split"],
            "time_of_event": np.nan,
            "time_of_alert": np.nan,
            "light_conditions": row["light_conditions"],
            "weather": row["weather"],
            "scene": row["scene"],
            "fps": round(fps, 2),
            "frame_count": n_frames,
            "duration_sec": round(n_frames / max(1.0, fps), 2),
            "resolution": f"{w}x{h}",
            "file_path": str(local_target.resolve()),
            "dataset_source": "nexar-ai/nexar_collision_prediction"
        })

    manifest_df = pd.DataFrame(pos_records + neg_records)
    out_manifest = base_dir / args.split_manifest
    manifest_df.to_csv(out_manifest, index=False)
    print(f"\n[SUCCESS] Split manifest written to: {out_manifest.resolve()}", flush=True)
    print(f"Total videos successfully downloaded and verified: {len(manifest_df)}", flush=True)
    print(pd.crosstab(manifest_df["split"], manifest_df["label"], margins=True), flush=True)

if __name__ == "__main__":
    main()
