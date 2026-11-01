"""
run_parallel_continuous.py
Parallel orchestrator for continuous sliding-window inference across 100 test videos.
Spawns 3 subprocess workers, merges predictions, and computes:
- continuous_predictions.csv
- false_alarm_metrics.csv
- lead_time_metrics.csv
"""

import sys
import subprocess
import time
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"

def main():
    print("="*75)
    print("STAGE 3.2: PARALLEL CONTINUOUS SLIDING-WINDOW INFERENCE (3 WORKERS)")
    print("="*75)

    script_path = SRC_DIR / "run_continuous_inference.py"

    partitions = [
        (0, 34, DATA_DIR / "preds_part0.csv"),
        (34, 67, DATA_DIR / "preds_part1.csv"),
        (67, 100, DATA_DIR / "preds_part2.csv")
    ]

    processes = []
    t0 = time.time()

    for idx_start, idx_end, out_file in partitions:
        cmd = [
            sys.executable, "-u", str(script_path),
            "--idx_start", str(idx_start),
            "--idx_end", str(idx_end),
            "--out_csv", str(out_file),
            "--only_predict"
        ]
        print(f"Launching worker for videos {idx_start} to {idx_end} -> {out_file.name}...", flush=True)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        processes.append((p, idx_start, idx_end, out_file))

    # Monitor processes
    while any(p.poll() is None for p, _, _, _ in processes):
        time.sleep(2)

    for p, idx_start, idx_end, out_file in processes:
        stdout, _ = p.communicate()
        if p.returncode != 0:
            print(f"[ERROR] Worker {idx_start}-{idx_end} failed with exit code {p.returncode}:")
            print(stdout)
            sys.exit(1)
        else:
            print(f"Worker {idx_start}-{idx_end} finished successfully ({out_file.name}).")

    elapsed = time.time() - t0
    print(f"\nAll 3 workers completed in {elapsed:.1f}s ({elapsed/60.0:.2f} minutes)!", flush=True)

    # Merge partition CSVs
    print("Merging continuous predictions...", flush=True)
    dfs = [pd.read_csv(out_file) for _, _, out_file in partitions]
    df_merged = pd.concat(dfs, ignore_index=True)
    out_merged = BASE_DIR / "continuous_predictions.csv"
    df_merged.to_csv(out_merged, index=False)
    print(f"Saved merged predictions: {len(df_merged)} rows -> {out_merged}", flush=True)

    # Clean up part files
    for _, _, out_file in partitions:
        if out_file.exists():
            out_file.unlink()

    # Now compute metrics on merged dataset
    df_test = pd.read_csv(BASE_DIR / "test_manifest.csv")
    df_neg_preds = df_merged[df_merged["ground_truth"] == 0].copy()
    total_neg_vids = df_test[df_test["label"] == 0]["video_id"].nunique()
    total_duration_sec = df_test[df_test["label"] == 0]["duration_sec"].sum()
    total_duration_min = total_duration_sec / 60.0

    print(f"\n--- Calculating Debounced False Alarm Rates (Negative Test Driving) ---", flush=True)
    print(f"Total Negative Videos: {total_neg_vids} | Total Continuous Duration: {total_duration_sec:.1f} s ({total_duration_min:.2f} minutes)", flush=True)

    fa_records = []
    models_list = df_merged["model_name"].unique()

    for m_name in models_list:
        sub_m = df_neg_preds[df_neg_preds["model_name"] == m_name]
        th = sub_m["val_threshold"].iloc[0]

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
                    if not in_alarm and consec >= 2:  # Min 2 consecutive steps (>=0.4s)
                        in_alarm = True
                        ep_start = t - 0.5
                        ep_peak = sc
                    elif in_alarm:
                        ep_peak = max(ep_peak, sc)
                else:
                    if in_alarm:
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

    # Early Warning Lead Time Analysis (Section 8)
    print("\n--- Calculating Early Warning Lead Time (Positive Collision Videos) ---", flush=True)
    df_pos_preds = df_merged[df_merged["ground_truth"] == 1].copy()
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

    print("\n[SUCCESS] Parallel continuous evaluation completed!", flush=True)

if __name__ == "__main__":
    main()
