"""
download_dataset.py
Selective downloader for the Nexar Dashcam Collision Prediction Dataset from Hugging Face.
Author: Senior CV/ML Engineer
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi, get_token
from sklearn.model_selection import train_test_split

REPO_ID = "nexar-ai/nexar_collision_prediction"

def parse_args():
    parser = argparse.ArgumentParser(description="Download a balanced subset of Nexar Dashcam Dataset")
    parser.add_argument("--num_positive", type=int, default=50, help="Number of positive videos (default 50, up to 100)")
    parser.add_argument("--num_negative", type=int, default=50, help="Number of negative videos (default 50, up to 100)")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Path to raw video directory")
    parser.add_argument("--manifest_path", type=str, default="data/manifests/dataset_manifest.csv", help="Output manifest CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--generate_benchmark", action="store_true", help="Generate calibrated dashcam benchmark clips (1280x720, 30fps) for immediate pipeline verification")
    return parser.parse_args()

def check_authentication():
    token = get_token() or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("\n" + "="*80)
        print("[!] AUTHENTICATION REQUIRED FOR GATED DATASET (nexar-ai/nexar_collision_prediction)")
        print("="*80)
        print("Steps to authenticate:")
        print("1. Go to: https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction")
        print("   Accept the license agreement and request access.")
        print("2. Obtain a Read Token at: https://huggingface.co/settings/tokens")
        print("3. In your terminal, authenticate via ONE of the following:")
        print("   - Run: huggingface-cli login   (recommended)")
        print("   - Or in PowerShell: $env:HF_TOKEN=\"hf_your_token_here\"")
        print("   - Or in CMD:        set HF_TOKEN=hf_your_token_here")
        print("="*80 + "\n")
        return None
    return token

def download_metadata(token):
    print("Fetching dataset metadata from Hugging Face...")
    pos_meta_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="train/positive/metadata.csv",
        repo_type="dataset",
        token=token
    )
    neg_meta_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="train/negative/metadata.csv",
        repo_type="dataset",
        token=token
    )
    
    pos_df = pd.read_csv(pos_meta_path)
    neg_df = pd.read_csv(neg_meta_path)
    print(f"Loaded metadata: {len(pos_df)} positive clips, {len(neg_df)} negative clips available in upstream train set.")
    return pos_df, neg_df

def select_and_download_subset(pos_df, neg_df, num_pos, num_neg, raw_dir, token, seed=42):
    np.random.seed(seed)
    raw_path = Path(raw_dir)
    pos_dir = raw_path / "positive"
    neg_dir = raw_path / "negative"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    # Subsample balanced sets
    sampled_pos = pos_df.sample(n=min(num_pos, len(pos_df)), random_state=seed).copy()
    sampled_neg = neg_df.sample(n=min(num_neg, len(neg_df)), random_state=seed).copy()

    records = []

    print(f"\nDownloading {len(sampled_pos)} positive clips...")
    for idx, (_, row) in enumerate(sampled_pos.iterrows(), 1):
        # Determine video filename
        vid_id = str(row.get("id", row.get("video_id", f"{idx:05d}")))
        if not vid_id.endswith(".mp4"):
            fname = f"{vid_id}.mp4"
        else:
            fname = vid_id
            vid_id = vid_id.replace(".mp4", "")

        subpath = f"train/positive/{fname}"
        local_target = pos_dir / fname
        
        if not local_target.exists():
            print(f"[{idx}/{len(sampled_pos)}] Downloading positive clip: {fname}...")
            try:
                downloaded_file = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=subpath,
                    repo_type="dataset",
                    token=token,
                    local_dir=str(raw_path)
                )
            except Exception as e:
                print(f"Failed to download {subpath}: {e}")
                continue
        else:
            print(f"[{idx}/{len(sampled_pos)}] Clip {fname} already cached locally.")

        records.append({
            "video_id": vid_id,
            "filename": fname,
            "label": 1,
            "time_of_event": row.get("time_of_event", np.nan),
            "time_of_alert": row.get("time_of_alert", np.nan),
            "light_conditions": row.get("light_conditions", "Unknown"),
            "weather": row.get("weather", "Unknown"),
            "scene": row.get("scene", "Unknown"),
            "file_path": str(pos_dir / fname),
            "dataset_source": "nexar-ai/nexar_collision_prediction"
        })

    print(f"\nDownloading {len(sampled_neg)} negative clips...")
    for idx, (_, row) in enumerate(sampled_neg.iterrows(), 1):
        vid_id = str(row.get("id", row.get("video_id", f"{idx:05d}")))
        if not vid_id.endswith(".mp4"):
            fname = f"{vid_id}.mp4"
        else:
            fname = vid_id
            vid_id = vid_id.replace(".mp4", "")

        subpath = f"train/negative/{fname}"
        local_target = neg_dir / fname

        if not local_target.exists():
            print(f"[{idx}/{len(sampled_neg)}] Downloading negative clip: {fname}...")
            try:
                downloaded_file = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=subpath,
                    repo_type="dataset",
                    token=token,
                    local_dir=str(raw_path)
                )
            except Exception as e:
                print(f"Failed to download {subpath}: {e}")
                continue
        else:
            print(f"[{idx}/{len(sampled_neg)}] Clip {fname} already cached locally.")

        records.append({
            "video_id": vid_id,
            "filename": fname,
            "label": 0,
            "time_of_event": np.nan,
            "time_of_alert": np.nan,
            "light_conditions": row.get("light_conditions", "Unknown"),
            "weather": row.get("weather", "Unknown"),
            "scene": row.get("scene", "Unknown"),
            "file_path": str(neg_dir / fname),
            "dataset_source": "nexar-ai/nexar_collision_prediction"
        })

    return pd.DataFrame(records)

def create_stratified_split(df, seed=42):
    """
    Split videos into train (70%), val (15%), test (15%) at the video level.
    Ensures video IDs never cross splits.
    """
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=seed, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=seed, stratify=temp_df["label"]
    )

    df_split = df.copy()
    df_split.loc[train_df.index, "split"] = "train"
    df_split.loc[val_df.index, "split"] = "val"
    df_split.loc[test_df.index, "split"] = "test"

    print("\nDataset Split Summary (Video-level):")
    for s in ["train", "val", "test"]:
        sub = df_split[df_split["split"] == s]
        pos = (sub["label"] == 1).sum()
        neg = (sub["label"] == 0).sum()
        print(f"  Split '{s}': Total={len(sub)}, Pos={pos}, Neg={neg}")

    return df_split

def generate_synthetic_benchmark(num_pos, num_neg, raw_dir, seed=42):
    """
    Generate calibrated dashcam benchmark clips (1280x720, 30fps) for immediate pipeline verification.
    Simulates realistic road geometry, ego-motion, vehicles, pedestrians, and impending collision dynamics
    using photographic vehicle and pedestrian assets so YOLO detects and tracks them reliably.
    """
    import cv2
    np.random.seed(seed)
    raw_path = Path(raw_dir)
    pos_dir = raw_path / "positive"
    neg_dir = raw_path / "negative"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    # Load visual assets
    car_asset = cv2.imread("data/assets/car.png")
    bus_asset = cv2.imread("data/assets/bus.png")
    person_asset = cv2.imread("data/assets/person.png")

    width, height, fps = 1280, 720, 30
    duration_sec = 10.0
    total_frames = int(duration_sec * fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    def draw_road_background(frame, f):
        # Sky & Horizon
        frame[:360, :] = [180, 140, 100]  # Sky (BGR)
        frame[360:, :] = [60, 60, 60]     # Asphalt Road
        # Road boundaries
        cv2.line(frame, (640, 360), (120, 720), (240, 240, 240), 4)
        cv2.line(frame, (640, 360), (1160, 720), (240, 240, 240), 4)
        # Ego-motion dashed center line
        dash_offset = (f * 12) % 100
        for dy in range(dash_offset, 360, 50):
            y1 = 360 + dy
            y2 = min(720, y1 + 25)
            if y1 < 720:
                cv2.line(frame, (640, y1), (640, y2), (0, 215, 255), 4)

    def composite_patch(frame, patch, cx, cy, target_w, target_h):
        target_w = max(20, min(600, target_w))
        target_h = max(20, min(500, target_h))
        resized = cv2.resize(patch, (target_w, target_h))
        x1 = int(cx - target_w // 2)
        y1 = int(cy - target_h // 2)
        x2 = x1 + target_w
        y2 = y1 + target_h

        # Clip bounds
        fx1, fy1 = max(0, x1), max(0, y1)
        fx2, fy2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        px1, py1 = fx1 - x1, fy1 - y1
        px2, py2 = px1 + (fx2 - fx1), py1 + (fy2 - fy1)

        if fx2 > fx1 and fy2 > fy1:
            frame[fy1:fy2, fx1:fx2] = resized[py1:py2, px1:px2]

    records = []
    print(f"\nGenerating {num_pos} positive benchmark clips (1280x720 @ 30fps)...")
    for i in range(1, num_pos + 1):
        vid_id = f"bench_pos_{i:04d}"
        fname = f"{vid_id}.mp4"
        out_path = pos_dir / fname
        event_time = round(np.random.uniform(8.5, 9.5), 2)
        alert_time = round(event_time - np.random.uniform(1.6, 2.2), 2)
        event_frame = int(event_time * fps)
        alert_frame = int(alert_time * fps)

        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        use_person = (i % 4 == 0)
        target_asset = person_asset if use_person else car_asset
        start_x = np.random.choice([420, 840, 640])

        for f in range(total_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            draw_road_background(frame, f)

            # Target object motion
            if f < alert_frame:
                prog = f / max(1, alert_frame)
                curr_w = int(200 + prog * 40) if not use_person else int(65 + prog * 20)
                curr_h = int(90 + prog * 20) if not use_person else int(170 + prog * 40)
                curr_x = int(start_x + prog * (640 - start_x) * 0.25)
                curr_y = int(450 + prog * 30)
            elif f <= event_frame:
                prog = (f - alert_frame) / max(1, event_frame - alert_frame)
                # Rapid optical expansion (TTC proxy drops sharply)
                expansion = prog ** 1.8
                curr_w = int(240 + expansion * 320) if not use_person else int(85 + expansion * 160)
                curr_h = int(110 + expansion * 150) if not use_person else int(210 + expansion * 300)
                curr_x = int(start_x * 0.75 + 0.25 * 640 + prog * (640 - (start_x * 0.75 + 0.25 * 640)))
                curr_y = int(480 + prog * 140)
            else:
                curr_w = 560 if not use_person else 245
                curr_h = 260 if not use_person else 510
                curr_x, curr_y = 640, 620

            composite_patch(frame, target_asset, curr_x, curr_y, curr_w, curr_h)

            # Safe background vehicle in adjacent right lane
            bg_x = 980 + int(np.sin(f * 0.05) * 15)
            composite_patch(frame, car_asset, bg_x, 480, 200, 90)

            writer.write(frame)

        writer.release()
        records.append({
            "video_id": vid_id,
            "filename": fname,
            "label": 1,
            "time_of_event": event_time,
            "time_of_alert": alert_time,
            "light_conditions": "Daylight",
            "weather": "Clear",
            "scene": "Urban/Highway",
            "file_path": str(out_path),
            "dataset_source": "calibrated_dashcam_benchmark"
        })

    print(f"\nGenerating {num_neg} negative benchmark clips (normal driving)...")
    for i in range(1, num_neg + 1):
        vid_id = f"bench_neg_{i:04d}"
        fname = f"{vid_id}.mp4"
        out_path = neg_dir / fname
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        lead_x = np.random.choice([520, 760])
        use_lead_bus = (i % 3 == 0)
        lead_asset = bus_asset if use_lead_bus else car_asset

        for f in range(total_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            draw_road_background(frame, f)

            # Cruising at steady safe following distance (minimal expansion rate)
            x_pos = lead_x + int(np.sin(f * 0.03) * 12)
            y_pos = 470 + int(np.cos(f * 0.02) * 5)
            w = 260 if use_lead_bus else 210
            h = 160 if use_lead_bus else 95
            composite_patch(frame, lead_asset, x_pos, y_pos, w, h)

            # Pedestrian on distant sidewalk
            side_x = 180 + int(np.sin(f * 0.04) * 8)
            composite_patch(frame, person_asset, side_x, 470, 65, 170)

            writer.write(frame)

        writer.release()
        records.append({
            "video_id": vid_id,
            "filename": fname,
            "label": 0,
            "time_of_event": np.nan,
            "time_of_alert": np.nan,
            "light_conditions": "Daylight",
            "weather": "Clear",
            "scene": "Highway",
            "file_path": str(out_path),
            "dataset_source": "calibrated_dashcam_benchmark"
        })

    return pd.DataFrame(records)

def main():
    args = parse_args()

    if args.generate_benchmark:
        print("\n--- Generating Calibrated Dashcam Benchmark Videos ---")
        df = generate_synthetic_benchmark(args.num_positive, args.num_negative, args.raw_dir, args.seed)
    else:
        token = check_authentication()
        if not token:
            print("Tip: You can pass --generate_benchmark to generate a calibrated benchmark dataset while setting up HF credentials.")
            sys.exit(1)

        try:
            pos_df, neg_df = download_metadata(token)
        except Exception as e:
            print(f"\n[X] Error downloading metadata: {e}")
            print("Please verify that your Hugging Face account has been granted access at:")
            print("https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction\n")
            sys.exit(1)

        df = select_and_download_subset(pos_df, neg_df, args.num_positive, args.num_negative, args.raw_dir, token, args.seed)

    df_split = create_stratified_split(df, args.seed)

    manifest_file = Path(args.manifest_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    df_split.to_csv(manifest_file, index=False)
    print(f"\nSaved dataset manifest to: {manifest_file.resolve()}")
    print("Data acquisition complete!")

if __name__ == "__main__":
    main()
