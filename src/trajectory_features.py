"""
trajectory_features.py
Model C: Proposed Trajectory-Based Temporal Feature Extractor.
Extracts consecutive temporal feature sequences:
- Position over time (cx, cy, bottom_y, area)
- Image-space velocity and acceleration (vx, vy, ax, ay)
- Relative motion and pairwise closing rates
- Bounding box optical expansion rate
- Heading direction changes and sudden lateral cut-ins
- Abrupt deceleration indicators
- Time-to-Contact (TTC) proxy (strictly image-space proxy, not metric)
- Camera ego-motion compensation
Outputs temporal sequences (T x D) for PyTorch GRU/LSTM and tabular summaries for XGBoost.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.tracker import MultiObjectTracker

FEATURE_DIM = 14
SEQUENCE_LENGTH = 10

class TrajectoryFeatureExtractor:
    def __init__(self, model_path="models/yolov8n.pt", conf_thresh=0.25, device="cpu", seq_len=10):
        self.tracker = MultiObjectTracker(model_path=model_path, conf_thresh=conf_thresh, device=device)
        self.seq_len = seq_len

    def extract_trajectory_sequence(self, video_path, start_frame, end_frame, step_frames=6):
        """
        Extracts temporal sequence (T x D) and tabular summary vector for a temporal window.
        """
        tracking_summary, track_history = self.tracker.process_window(
            video_path, start_frame, end_frame, step_frames=step_frames
        )

        cap = cv2_cap = None
        # Determine number of sampled steps
        total_steps = ((end_frame - start_frame) // step_frames) + 1
        dt = step_frames / 30.0  # approximate dt

        # Identify primary (most critical) track:
        # Chosen based on combination of largest area, lowest bottom_y (closest to ego vehicle), or highest expansion
        primary_tid = None
        best_crit = -1.0

        for tid, states in track_history.items():
            if len(states) >= 2:
                max_bottom = max(s["bottom_y"] for s in states)
                max_area = max(s["area"] for s in states)
                crit = max_bottom * 0.6 + max_area * 0.4
                if crit > best_crit:
                    best_crit = crit
                    primary_tid = tid

        # If no primary track with >=2 states, pick any track
        if primary_tid is None and len(track_history) > 0:
            primary_tid = list(track_history.keys())[0]

        # Build sequence of frame states
        # Sequence matrix shape: (seq_len, FEATURE_DIM)
        seq_matrix = np.zeros((self.seq_len, FEATURE_DIM), dtype=np.float32)

        if primary_tid is not None and primary_tid in track_history:
            states = track_history[primary_tid]
            # Map states to time indices
            n_states = len(states)

            vx_prev, vy_prev = 0.0, 0.0
            heading_prev = 0.0

            # Compute kinematics per state
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
                    
                    # Optical expansion rate
                    prev_area = max(1e-5, states[i-1]["area"])
                    expansion_rate = (area - prev_area) / (prev_area * dt_actual)
                    
                    # Heading angle & change
                    heading_curr = np.arctan2(vy, vx)
                    heading_diff = heading_curr - heading_prev
                    # normalize heading to [-pi, pi]
                    heading_diff = (heading_diff + np.pi) % (2 * np.pi) - np.pi
                    heading_change_rate = heading_diff / dt_actual

                    # Sudden lateral movement proxy
                    lateral_ratio = abs(vx) / (abs(vy) + 1e-4)

                    # Abrupt deceleration indicator: downward vertical expansion surge
                    abrupt_decel = max(0.0, expansion_rate) * max(0.0, vy)

                    # Time-to-Contact (TTC) proxy: inverse expansion rate
                    # Note: Labeled strictly as an image-space proxy per research physical limitations
                    if expansion_rate > 1e-3:
                        ttc_proxy = 1.0 / expansion_rate
                    else:
                        ttc_proxy = 10.0  # safe large default

                    # Camera ego-motion compensation:
                    # Subtract median background flow/motion estimated from other tracks if present
                    ego_comp_vy = vy
                    if len(track_history) > 1:
                        other_vys = []
                        for otid, ostates in track_history.items():
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

                f_vec = [
                    cx, cy, bottom_y, area,
                    vx, vy, ax, ay,
                    expansion_rate,
                    heading_change_rate,
                    lateral_ratio,
                    abrupt_decel,
                    ttc_proxy,
                    ego_comp_vy
                ]
                state_feats.append(f_vec)

            # Interpolate / resample to fixed seq_len
            state_feats = np.array(state_feats, dtype=np.float32)
            if len(state_feats) == self.seq_len:
                seq_matrix = state_feats
            elif len(state_feats) > self.seq_len:
                # Subsample evenly
                indices = np.linspace(0, len(state_feats) - 1, self.seq_len).astype(int)
                seq_matrix = state_feats[indices]
            else:
                # Pad earlier frames with repetition of first frame
                pad_len = self.seq_len - len(state_feats)
                pad_block = np.repeat(state_feats[0:1], pad_len, axis=0)
                seq_matrix = np.vstack([pad_block, state_feats])

        # Summary statistics for Tabular XGBoost comparison
        summary_dict = {}
        feature_names = [
            "cx", "cy", "bottom_y", "area",
            "vx", "vy", "ax", "ay",
            "expansion_rate", "heading_change_rate", "lateral_ratio",
            "abrupt_decel", "ttc_proxy", "ego_comp_vy"
        ]
        for idx, fname in enumerate(feature_names):
            vals = seq_matrix[:, idx]
            summary_dict[f"traj_{fname}_mean"] = float(np.mean(vals))
            summary_dict[f"traj_{fname}_max"] = float(np.max(vals))
            summary_dict[f"traj_{fname}_min"] = float(np.min(vals))
            summary_dict[f"traj_{fname}_std"] = float(np.std(vals))
            summary_dict[f"traj_{fname}_last"] = float(vals[-1])
            # Trend: slope from start to end
            summary_dict[f"traj_{fname}_trend"] = float(vals[-1] - vals[0])

        # Merge with tracker summary
        for k, v in tracking_summary.items():
            summary_dict[k] = v

        return seq_matrix, summary_dict

if __name__ == "__main__":
    print("Testing TrajectoryFeatureExtractor...")
    extractor = TrajectoryFeatureExtractor(seq_len=10)
    seq, summary = extractor.extract_trajectory_sequence(
        "data/raw/positive/bench_pos_0001.mp4", 0, 90, step_frames=6
    )
    print(f"Sequence Matrix Shape: {seq.shape} (T x D)")
    print("\nSample Tabular Trajectory Features (Model C XGB):")
    sample_keys = [
        "traj_expansion_rate_max", "traj_ttc_proxy_min",
        "traj_abrupt_decel_max", "traj_vy_max", "traj_ego_comp_vy_max",
        "active_track_count"
    ]
    for k in sample_keys:
        print(f"  {k}: {summary.get(k, 0.0):.4f}")
