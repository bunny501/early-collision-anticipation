"""
create_windows.py
Generates reproducible temporal observation windows for Stage 3 real-world validation.
Constructs 3 observation windows per video:
- Positive videos: observation windows ending at 1.5s, 1.0s, and 0.5s prior to collision event.
- Negative videos: observation windows matching the equivalent temporal duration and range.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Create temporal window manifest")
    parser.add_argument("--split_manifest", type=str, default="split_manifest.csv", help="Input split manifest")
    parser.add_argument("--out_manifest", type=str, default="data/manifests/window_manifest.csv", help="Output window manifest")
    parser.add_argument("--obs_duration", type=float, default=2.0, help="Observation window duration in seconds")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    split_file = base_dir / args.split_manifest
    if not split_file.exists():
        print(f"[ERROR] Split manifest not found: {split_file}")
        sys.exit(1)

    df_splits = pd.read_csv(split_file)
    print(f"Loaded split manifest with {len(df_splits)} videos.")

    out_file = base_dir / args.out_manifest
    out_file.parent.mkdir(parents=True, exist_ok=True)

    records = []
    horizons = [1.5, 1.0, 0.5]

    for _, row in df_splits.iterrows():
        vid_id = str(row["video_id"])
        label = int(row["label"])
        split = row["split"]
        fps = float(row["fps"])
        vpath = str(row["file_path"])
        dur = float(row["duration_sec"])

        if label == 1:
            t_event = float(row["time_of_event"])
            t_alert = float(row["time_of_alert"]) if pd.notnull(row["time_of_alert"]) else np.nan

            for h in horizons:
                t_end = max(args.obs_duration, t_event - h)
                t_start = max(0.0, t_end - args.obs_duration)
                sf = int(round(t_start * fps))
                ef = int(round(t_end * fps))

                records.append({
                    "window_id": f"{vid_id}_h{int(h*1000):04d}ms",
                    "video_id": vid_id,
                    "label": 1,
                    "split": split,
                    "horizon_sec": h,
                    "start_time_sec": round(t_start, 2),
                    "end_time_sec": round(t_end, 2),
                    "start_frame": sf,
                    "end_frame": ef,
                    "time_of_event": t_event,
                    "time_of_alert": t_alert,
                    "duration_sec": dur,
                    "fps": fps,
                    "file_path": vpath,
                    "scene": row.get("scene", "Unknown"),
                    "weather": row.get("weather", "Unknown"),
                    "light_conditions": row.get("light_conditions", "Unknown")
                })
        else:
            # Negative video: sample 3 comparable windows around typical event timeframe
            # Positive events occur between 9s and 23s (median ~19.9s).
            # We sample windows ending at 16.0s (1.5s horizon), 18.0s (1.0s horizon), 20.0s (0.5s horizon)
            neg_ends = {1.5: 16.0, 1.0: 18.0, 0.5: 20.0}
            for h in horizons:
                target_end = neg_ends[h]
                t_end = min(dur - 0.5, target_end)
                t_start = max(0.0, t_end - args.obs_duration)
                sf = int(round(t_start * fps))
                ef = int(round(t_end * fps))

                records.append({
                    "window_id": f"{vid_id}_h{int(h*1000):04d}ms",
                    "video_id": vid_id,
                    "label": 0,
                    "split": split,
                    "horizon_sec": h,
                    "start_time_sec": round(t_start, 2),
                    "end_time_sec": round(t_end, 2),
                    "start_frame": sf,
                    "end_frame": ef,
                    "time_of_event": np.nan,
                    "time_of_alert": np.nan,
                    "duration_sec": dur,
                    "fps": fps,
                    "file_path": vpath,
                    "scene": row.get("scene", "Unknown"),
                    "weather": row.get("weather", "Unknown"),
                    "light_conditions": row.get("light_conditions", "Unknown")
                })

    df_windows = pd.DataFrame(records)
    df_windows.to_csv(out_file, index=False)
    print(f"\n[SUCCESS] Generated {len(df_windows)} temporal observation windows.")
    print(f"Saved window manifest to: {out_file.resolve()}")
    print("\nWindow Count Breakdown by Split & Label:")
    print(pd.crosstab(df_windows["split"], [df_windows["label"], df_windows["horizon_sec"]]))

if __name__ == "__main__":
    main()
