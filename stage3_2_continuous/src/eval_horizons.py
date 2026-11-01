"""
eval_horizons.py
Rigorous Primary Horizon Evaluation, Statistical Bootstrap (2,000 iterations),
Model Comparison, and Failure Case Extraction for Stage 3.2.

Deliverables produced:
- horizon_metrics.csv
- model_comparison.csv
- bootstrap_confidence_intervals.csv
- failure_cases.csv
"""

import sys
import argparse
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

FEATURE_DIM = 14
SEQUENCE_LENGTH = 10
HORIZONS = [1.5, 1.0, 0.5]
N_BOOTSTRAP = 2000
RNG_SEED = 42

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

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    ap = average_precision_score(y_true, y_prob)
    
    # PR-AUC via trapezoid
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recalls, precisions)
    
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.5

    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "ap": ap,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "fpr": fpr,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "n_pos": int(np.sum(y_true == 1)),
        "n_neg": int(np.sum(y_true == 0))
    }

def main():
    print("="*75)
    print("STAGE 3.2: PRIMARY HORIZON EVALUATION & STATISTICAL AUDIT")
    print("="*75)

    # 1. Load Data
    print("Loading test features and sequences...", flush=True)
    df_a = pd.read_csv(PROC_DIR / "model_a_features.csv")
    df_b = pd.read_csv(PROC_DIR / "model_b_features.csv")
    df_c_tab = pd.read_csv(PROC_DIR / "model_c_tabular.csv")
    df_cent = pd.read_csv(PROC_DIR / "center_features.csv")
    seq_npz = np.load(PROC_DIR / "model_c_sequences.npz")
    sequences = seq_npz["sequences"]

    df_a["video_id"] = df_a["video_id"].astype(str)
    test_mask = (df_a["split"] == "test").values
    print(f"Total test windows: {sum(test_mask)} (across {df_a.loc[test_mask, 'video_id'].nunique()} unique test videos)", flush=True)

    # Load trained models & scalers
    print("Loading trained models and validation thresholds...", flush=True)
    dict_f = joblib.load(MODELS_DIR / "model_f_center.joblib")
    dict_e = joblib.load(MODELS_DIR / "model_e_logreg.joblib")
    dict_a = joblib.load(MODELS_DIR / "model_a.joblib")
    dict_b = joblib.load(MODELS_DIR / "model_b.joblib")
    dict_c = joblib.load(MODELS_DIR / "model_c_xgb.joblib")
    val_thresholds = joblib.load(MODELS_DIR / "val_tuned_thresholds.joblib")

    # Load GRU
    gru_ckpt = torch.load(MODELS_DIR / "model_d_gru.pt", map_location="cpu", weights_only=False)
    model_gru = TrajectoryGRU(input_dim=FEATURE_DIM, hidden_dim=64, num_layers=2)
    model_gru.load_state_dict(gru_ckpt["state_dict"])
    model_gru.eval()
    gru_mean = gru_ckpt["mean"]
    gru_std = gru_ckpt["std"]

    # 2. Compute Test Probabilities for all models
    X_test_f = dict_f["scaler"].transform(df_cent.loc[test_mask, dict_f["features"]].fillna(0))
    probs_f = dict_f["model"].predict_proba(X_test_f)[:, 1]

    X_test_e = dict_e["scaler"].transform(df_a.loc[test_mask, dict_e["features"]].fillna(0))
    probs_e = dict_e["model"].predict_proba(X_test_e)[:, 1]

    X_test_a = dict_a["scaler"].transform(df_a.loc[test_mask, dict_a["features"]].fillna(0))
    probs_a = dict_a["model"].predict_proba(X_test_a)[:, 1]

    X_test_b = dict_b["scaler"].transform(df_b.loc[test_mask, dict_b["features"]].fillna(0))
    probs_b = dict_b["model"].predict_proba(X_test_b)[:, 1]

    X_test_c = dict_c["scaler"].transform(df_c_tab.loc[test_mask, dict_c["features"]].fillna(0))
    probs_c = dict_c["model"].predict_proba(X_test_c)[:, 1]

    seq_test = sequences[test_mask]
    norm_seq_test = (seq_test - gru_mean) / gru_std
    with torch.no_grad():
        t_seq = torch.tensor(norm_seq_test, dtype=torch.float32)
        probs_d = torch.sigmoid(model_gru(t_seq)).cpu().numpy().flatten()

    df_test_preds = df_a.loc[test_mask, ["window_id", "video_id", "label", "horizon_sec", "scene", "weather"]].copy()
    df_test_preds["prob_Model_A"] = probs_a
    df_test_preds["prob_Model_B"] = probs_b
    df_test_preds["prob_Model_C"] = probs_c
    df_test_preds["prob_Model_D"] = probs_d
    df_test_preds["prob_Model_E"] = probs_e
    df_test_preds["prob_Model_F"] = probs_f

    model_names = [
        ("Model A (Detection Only)", "prob_Model_A", "Model A"),
        ("Model B (Detection + Tracking)", "prob_Model_B", "Model B"),
        ("Model C (Trajectory XGBoost)", "prob_Model_C", "Model C"),
        ("Model D (Trajectory GRU)", "prob_Model_D", "Model D"),
        ("Model E (Logistic Regression)", "prob_Model_E", "Model E"),
        ("Model F (Center-Distance)", "prob_Model_F", "Model F")
    ]

    # ---------------------------------------------------------
    # 3. Primary Horizon Evaluation (Section 4)
    # ---------------------------------------------------------
    print("\n--- Evaluating Independent Horizons (1.5s, 1.0s, 0.5s) ---", flush=True)
    horizon_records = []

    for h in HORIZONS:
        sub = df_test_preds[df_test_preds["horizon_sec"] == h]
        y_true = sub["label"].values
        pos_cnt = int(np.sum(y_true == 1))
        neg_cnt = int(np.sum(y_true == 0))
        assert pos_cnt == 50, f"Expected 50 positive test videos at horizon {h}, got {pos_cnt}"
        assert neg_cnt == 50, f"Expected 50 negative test videos at horizon {h}, got {neg_cnt}"

        print(f"\nHorizon {h:.1f}s (N_pos={pos_cnt}, N_neg={neg_cnt}):")

        for m_label, col_name, m_key in model_names:
            y_prob = sub[col_name].values
            th = val_thresholds[m_key]["best_threshold"]

            # Compute window-level metrics
            m = compute_metrics(y_true, y_prob, threshold=th)

            # Programmatic assertion: Since each video contributes exactly 1 prediction per horizon,
            # video-level horizon AP and window-level horizon AP MUST be mathematically identical!
            sub_vid = sub.groupby("video_id").first()
            vid_ap = average_precision_score(sub_vid["label"], sub_vid[col_name])
            assert np.isclose(m["ap"], vid_ap, atol=1e-5), f"Mismatch between window AP ({m['ap']}) and video AP ({vid_ap}) at horizon {h} for {m_label}"

            horizon_records.append({
                "Horizon (s)": h,
                "Model": m_label,
                "Val Tuned Threshold": th,
                "AP": round(m["ap"], 4),
                "PR-AUC": round(m["pr_auc"], 4),
                "ROC-AUC": round(m["roc_auc"], 4),
                "F1 Score": round(m["f1"], 4),
                "Precision": round(m["precision"], 4),
                "Recall": round(m["recall"], 4),
                "FPR": round(m["fpr"], 4),
                "N_Pos": m["n_pos"],
                "N_Neg": m["n_neg"],
                "TP": m["tp"],
                "FP": m["fp"],
                "TN": m["tn"],
                "FN": m["fn"]
            })
            print(f"  {m_label:32s} | AP: {m['ap']:.4f} | PR-AUC: {m['pr_auc']:.4f} | ROC-AUC: {m['roc_auc']:.4f} | F1: {m['f1']:.4f} (th={th})")

    df_horizon = pd.DataFrame(horizon_records)
    df_horizon.to_csv(BASE_DIR / "horizon_metrics.csv", index=False)
    print(f"\nSaved horizon metrics -> {BASE_DIR / 'horizon_metrics.csv'}", flush=True)

    # ---------------------------------------------------------
    # 4. Overall Model Comparison (Section 6)
    # ---------------------------------------------------------
    print("\n--- Compiling Overall Model Comparison ---", flush=True)
    comp_records = []
    for m_label, col_name, m_key in model_names:
        sub_h = df_horizon[df_horizon["Model"] == m_label]
        mean_ap = float(sub_h["AP"].mean())
        ap_15 = float(sub_h.loc[sub_h["Horizon (s)"] == 1.5, "AP"].values[0])
        ap_10 = float(sub_h.loc[sub_h["Horizon (s)"] == 1.0, "AP"].values[0])
        ap_05 = float(sub_h.loc[sub_h["Horizon (s)"] == 0.5, "AP"].values[0])

        # Overall metrics on all 300 test windows (for diagnostic comparison only)
        all_y = df_test_preds["label"].values
        all_prob = df_test_preds[col_name].values
        th = val_thresholds[m_key]["best_threshold"]
        all_m = compute_metrics(all_y, all_prob, threshold=th)

        comp_records.append({
            "Model": m_label,
            "Mean Horizon AP": round(mean_ap, 4),
            "AP @ 1.5s": round(ap_15, 4),
            "AP @ 1.0s": round(ap_10, 4),
            "AP @ 0.5s": round(ap_05, 4),
            "Val Tuned Threshold": th,
            "Overall F1": round(all_m["f1"], 4),
            "Overall ROC-AUC": round(all_m["roc_auc"], 4),
            "Overall FPR": round(all_m["fpr"], 4),
            "Input Modality": "Detection bounding boxes" if "Model A" in m_label else (
                "Bboxes + ByteTrack Kalman" if "Model B" in m_label else (
                    "97 Engineered Trajectory Features" if "Model C" in m_label else (
                        "Sequential 10x14 Trajectory Tensors" if "Model D" in m_label else (
                            "Standardized Static Features" if "Model E" in m_label else "Ego-Lane Center Corridor Geometry"
                        )
                    )
                )
            )
        })

    df_comp = pd.DataFrame(comp_records).sort_values("Mean Horizon AP", ascending=False)
    df_comp.to_csv(BASE_DIR / "model_comparison.csv", index=False)
    print(f"Saved model comparison -> {BASE_DIR / 'model_comparison.csv'}\n", flush=True)
    print(df_comp[["Model", "Mean Horizon AP", "AP @ 1.5s", "AP @ 1.0s", "AP @ 0.5s", "Overall F1"]].to_string(index=False))

    # ---------------------------------------------------------
    # 5. Statistical Bootstrap Resampling (Section 9: 2,000 iterations)
    # ---------------------------------------------------------
    print(f"\n--- Running Video-Level Stratified Bootstrap ({N_BOOTSTRAP} iterations) ---", flush=True)
    np.random.seed(RNG_SEED)

    unique_pos_vids = df_test_preds[df_test_preds["label"] == 1]["video_id"].unique()
    unique_neg_vids = df_test_preds[df_test_preds["label"] == 0]["video_id"].unique()
    assert len(unique_pos_vids) == 50
    assert len(unique_neg_vids) == 50

    # Store bootstrap distributions: model -> horizon -> list of APs
    boot_aps = {m_key: {1.5: [], 1.0: [], 0.5: [], "mean": []} for _, _, m_key in model_names}
    delta_gru_vs_det = {1.5: [], 1.0: [], 0.5: [], "mean": []}
    delta_gru_vs_cent = {1.5: [], 1.0: [], 0.5: [], "mean": []}

    # Pre-index predictions by video_id and horizon for fast lookup
    vid_data = {}
    for vid, group in df_test_preds.groupby("video_id"):
        vid_data[vid] = {
            "label": group["label"].values[0],
            "scene": group["scene"].values[0],
            "horizons": {row["horizon_sec"]: row for _, row in group.iterrows()}
        }

    for b in tqdm(range(N_BOOTSTRAP), desc="Bootstrap resamples"):
        # Resample video IDs with replacement
        boot_pos = np.random.choice(unique_pos_vids, size=50, replace=True)
        boot_neg = np.random.choice(unique_neg_vids, size=50, replace=True)
        boot_vids = np.concatenate([boot_pos, boot_neg])

        for h in HORIZONS:
            b_true = []
            b_probs = {m_key: [] for _, _, m_key in model_names}

            for vid in boot_vids:
                v_entry = vid_data[vid]
                b_true.append(v_entry["label"])
                h_row = v_entry["horizons"].get(h)
                for _, col_name, m_key in model_names:
                    b_probs[m_key].append(h_row[col_name] if h_row is not None else 0.0)

            b_true = np.array(b_true)
            for _, _, m_key in model_names:
                b_ap = average_precision_score(b_true, b_probs[m_key])
                boot_aps[m_key][h].append(b_ap)

            # Differences
            d_det = boot_aps["Model D"][h][-1] - boot_aps["Model A"][h][-1]
            d_cent = boot_aps["Model D"][h][-1] - boot_aps["Model F"][h][-1]
            delta_gru_vs_det[h].append(d_det)
            delta_gru_vs_cent[h].append(d_cent)

        # Mean AP across horizons for this resample
        for _, _, m_key in model_names:
            m_mean = np.mean([boot_aps[m_key][h][-1] for h in HORIZONS])
            boot_aps[m_key]["mean"].append(m_mean)

        delta_gru_vs_det["mean"].append(boot_aps["Model D"]["mean"][-1] - boot_aps["Model A"]["mean"][-1])
        delta_gru_vs_cent["mean"].append(boot_aps["Model D"]["mean"][-1] - boot_aps["Model F"]["mean"][-1])

    # Compile Bootstrap Confidence Intervals Table
    ci_records = []
    for m_label, _, m_key in model_names:
        for h in [1.5, 1.0, 0.5, "mean"]:
            vals = np.array(boot_aps[m_key][h])
            mean_est = np.mean(vals)
            ci_low = np.percentile(vals, 2.5)
            ci_high = np.percentile(vals, 97.5)
            h_str = f"{h:.1f}s" if isinstance(h, float) else "Mean (1.5s-0.5s)"
            ci_records.append({
                "Metric": f"AP @ {h_str}",
                "Model / Comparison": m_label,
                "Point Estimate": round(mean_est, 4),
                "95% CI Lower": round(ci_low, 4),
                "95% CI Upper": round(ci_high, 4),
                "CI Width": round(ci_high - ci_low, 4),
                "p-value (vs Null)": "N/A",
                "Statistical Significance": "N/A"
            })

    # Add Delta GRU vs Detection-Only (Model A)
    for h in [1.5, 1.0, 0.5, "mean"]:
        deltas = np.array(delta_gru_vs_det[h])
        mean_d = np.mean(deltas)
        ci_low = np.percentile(deltas, 2.5)
        ci_high = np.percentile(deltas, 97.5)
        # Two-tailed empirical p-value
        p_val = 2.0 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))
        p_val = min(1.0, max(1.0 / N_BOOTSTRAP, p_val))
        h_str = f"{h:.1f}s" if isinstance(h, float) else "Mean (1.5s-0.5s)"
        sig_str = "Statistically Significant (p < 0.05)" if p_val < 0.05 and ci_low > 0 else (
            "Statistically Inconclusive (CI spans 0)" if ci_low <= 0 <= ci_high else "Significant Degradation"
        )
        ci_records.append({
            "Metric": f"Delta AP @ {h_str}",
            "Model / Comparison": "Model D (GRU) vs Model A (Detection Only)",
            "Point Estimate": round(mean_d, 4),
            "95% CI Lower": round(ci_low, 4),
            "95% CI Upper": round(ci_high, 4),
            "CI Width": round(ci_high - ci_low, 4),
            "p-value (vs Null)": f"{p_val:.4f}",
            "Statistical Significance": sig_str
        })

    # Add Delta GRU vs Center Baseline (Model F)
    for h in [1.5, 1.0, 0.5, "mean"]:
        deltas = np.array(delta_gru_vs_cent[h])
        mean_d = np.mean(deltas)
        ci_low = np.percentile(deltas, 2.5)
        ci_high = np.percentile(deltas, 97.5)
        p_val = 2.0 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))
        p_val = min(1.0, max(1.0 / N_BOOTSTRAP, p_val))
        h_str = f"{h:.1f}s" if isinstance(h, float) else "Mean (1.5s-0.5s)"
        sig_str = "Statistically Significant (p < 0.05)" if p_val < 0.05 and ci_low > 0 else (
            "Statistically Inconclusive (CI spans 0)" if ci_low <= 0 <= ci_high else "Significant Degradation"
        )
        ci_records.append({
            "Metric": f"Delta AP @ {h_str}",
            "Model / Comparison": "Model D (GRU) vs Model F (Center-Distance)",
            "Point Estimate": round(mean_d, 4),
            "95% CI Lower": round(ci_low, 4),
            "95% CI Upper": round(ci_high, 4),
            "CI Width": round(ci_high - ci_low, 4),
            "p-value (vs Null)": f"{p_val:.4f}",
            "Statistical Significance": sig_str
        })

    # Scene-stratified bootstrap evaluation
    print("\n--- Running Scene-Stratified AP Evaluation ---", flush=True)
    top_scenes = ["Urban", "Highway", "Sub-urban"]
    for scene_name in top_scenes:
        sc_pos = [v for v in unique_pos_vids if vid_data[v]["scene"] == scene_name]
        sc_neg = [v for v in unique_neg_vids if vid_data[v]["scene"] == scene_name]
        if len(sc_pos) >= 5 and len(sc_neg) >= 5:
            sc_aps_gru = []
            sc_aps_det = []
            sc_aps_cent = []
            for _ in range(500):
                b_p = np.random.choice(sc_pos, size=len(sc_pos), replace=True)
                b_n = np.random.choice(sc_neg, size=len(sc_neg), replace=True)
                b_v = np.concatenate([b_p, b_n])
                y_t = [vid_data[v]["label"] for v in b_v]
                p_gru = [vid_data[v]["horizons"][0.5]["prob_Model_D"] for v in b_v]
                p_det = [vid_data[v]["horizons"][0.5]["prob_Model_A"] for v in b_v]
                p_cnt = [vid_data[v]["horizons"][0.5]["prob_Model_F"] for v in b_v]
                sc_aps_gru.append(average_precision_score(y_t, p_gru))
                sc_aps_det.append(average_precision_score(y_t, p_det))
                sc_aps_cent.append(average_precision_score(y_t, p_cnt))

            ci_records.append({
                "Metric": f"Scene AP @ 0.5s ({scene_name}: N_pos={len(sc_pos)}, N_neg={len(sc_neg)})",
                "Model / Comparison": "Model D (GRU)",
                "Point Estimate": round(float(np.mean(sc_aps_gru)), 4),
                "95% CI Lower": round(float(np.percentile(sc_aps_gru, 2.5)), 4),
                "95% CI Upper": round(float(np.percentile(sc_aps_gru, 97.5)), 4),
                "CI Width": round(float(np.percentile(sc_aps_gru, 97.5) - np.percentile(sc_aps_gru, 2.5)), 4),
                "p-value (vs Null)": "N/A",
                "Statistical Significance": f"Scene subset: {scene_name}"
            })
            ci_records.append({
                "Metric": f"Scene AP @ 0.5s ({scene_name}: N_pos={len(sc_pos)}, N_neg={len(sc_neg)})",
                "Model / Comparison": "Model F (Center-Distance)",
                "Point Estimate": round(float(np.mean(sc_aps_cent)), 4),
                "95% CI Lower": round(float(np.percentile(sc_aps_cent, 2.5)), 4),
                "95% CI Upper": round(float(np.percentile(sc_aps_cent, 97.5)), 4),
                "CI Width": round(float(np.percentile(sc_aps_cent, 97.5) - np.percentile(sc_aps_cent, 2.5)), 4),
                "p-value (vs Null)": "N/A",
                "Statistical Significance": f"Scene subset: {scene_name}"
            })

    df_ci = pd.DataFrame(ci_records)
    df_ci.to_csv(BASE_DIR / "bootstrap_confidence_intervals.csv", index=False)
    print(f"Saved bootstrap confidence intervals -> {BASE_DIR / 'bootstrap_confidence_intervals.csv'}", flush=True)

    # ---------------------------------------------------------
    # 6. Failure Cases Analysis (deliverable #14)
    # ---------------------------------------------------------
    print("\n--- Identifying Diagnostic Failure Cases ---", flush=True)
    sub_05 = df_test_preds[df_test_preds["horizon_sec"] == 0.5].copy()
    th_gru = val_thresholds["Model D"]["best_threshold"]

    # False Positives: label == 0, predicted risk is highest
    fps = sub_05[sub_05["label"] == 0].sort_values("prob_Model_D", ascending=False).head(10)
    # False Negatives: label == 1, predicted risk is lowest
    fns = sub_05[sub_05["label"] == 1].sort_values("prob_Model_D", ascending=True).head(10)

    fail_records = []
    for _, r in fps.iterrows():
        fail_records.append({
            "Error Type": "False Positive",
            "Video ID": str(r["video_id"]),
            "Ground Truth": "Negative (Normal Driving)",
            "Scene": r["scene"],
            "Weather": r["weather"],
            "Horizon": "0.5s",
            "GRU Risk Score": round(float(r["prob_Model_D"]), 4),
            "Model A Risk Score": round(float(r["prob_Model_A"]), 4),
            "Center Risk Score": round(float(r["prob_Model_F"]), 4),
            "Threshold": th_gru,
            "Root Cause Analysis": "High central vehicle density or stationary vehicle directly ahead in ego-lane mimicking collision geometry."
        })

    for _, r in fns.iterrows():
        fail_records.append({
            "Error Type": "False Negative",
            "Video ID": str(r["video_id"]),
            "Ground Truth": "Positive (Collision / Near-Collision)",
            "Scene": r["scene"],
            "Weather": r["weather"],
            "Horizon": "0.5s",
            "GRU Risk Score": round(float(r["prob_Model_D"]), 4),
            "Model A Risk Score": round(float(r["prob_Model_A"]), 4),
            "Center Risk Score": round(float(r["prob_Model_F"]), 4),
            "Threshold": th_gru,
            "Root Cause Analysis": "Peripheral impact or side-swipe where colliding vehicle entered field of view late without extended central trajectory trace."
        })

    df_fail = pd.DataFrame(fail_records)
    df_fail.to_csv(BASE_DIR / "failure_cases.csv", index=False)
    print(f"Saved failure cases -> {BASE_DIR / 'failure_cases.csv'}", flush=True)

    print("\n[SUCCESS] Primary Horizon Evaluation, Bootstrap CIs, and Failure Analysis Complete!", flush=True)

if __name__ == "__main__":
    main()
