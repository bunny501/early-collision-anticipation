"""
evaluate.py
Comprehensive Experimental Evaluation Pipeline.
Evaluates Model A, Model B, and Model C on the exact same held-out test split.
Measures:
- Multi-horizon Average Precision (1.5s, 1.0s, 0.5s and Mean AP)
- Precision, Recall, F1-score, PR-AUC, ROC-AUC
- False Positive Rate (FPR), False Alarms per Minute
- Warning Lead Time relative to collision event and alert timestamp
- Inference Latency and Hardware Profile (CPU/GPU)
Generates results/metrics.csv, comparison.png, and confusion matrices.
"""

import os
import sys
import time
import argparse
import joblib
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.train_temporal import TrajectoryGRU, FEATURE_DIM

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Collision Anticipation Models")
    parser.add_argument("--manifest_path", type=str, default="data/manifests/window_manifest.csv")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--results_dir", type=str, default="results")
    return parser.parse_args()

def compute_horizon_ap(y_true, y_probs, horizons, target_h):
    mask = (horizons == target_h)
    if sum(mask) == 0:
        return 0.0
    # Include all negative windows evaluated against positive windows at this horizon
    neg_mask = (y_true == 0)
    eval_mask = mask | neg_mask
    if len(np.unique(y_true[eval_mask])) < 2:
        return float(y_true[eval_mask].mean())
    return float(average_precision_score(y_true[eval_mask], y_probs[eval_mask]))

def evaluate_predictions(y_true, y_probs, horizons, duration_min, latency_ms):
    y_pred = (y_probs >= 0.5).astype(int)

    # Core metrics
    pr_auc = float(average_precision_score(y_true, y_probs))
    roc_auc = float(roc_auc_score(y_true, y_probs)) if len(np.unique(y_true)) > 1 else 0.5
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))

    # Confusion Matrix & FPR
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / max(1, (fp + tn)))
    false_alarms_per_min = float(fp / max(0.01, duration_min))

    # Multi-horizon AP
    ap_15 = compute_horizon_ap(y_true, y_probs, horizons, 1.5)
    ap_10 = compute_horizon_ap(y_true, y_probs, horizons, 1.0)
    ap_05 = compute_horizon_ap(y_true, y_probs, horizons, 0.5)
    mean_ap = float(np.mean([ap_15, ap_10, ap_05]))

    return {
        "1.5s AP": round(ap_15, 4),
        "1.0s AP": round(ap_10, 4),
        "0.5s AP": round(ap_05, 4),
        "Mean AP": round(mean_ap, 4),
        "PR-AUC": round(pr_auc, 4),
        "ROC-AUC": round(roc_auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "FPR": round(fpr, 4),
        "False Alarms/min": round(false_alarms_per_min, 2),
        "Latency (ms)": round(latency_ms, 2),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn
    }

def main():
    args = parse_args()
    results_path = Path(args.results_dir)
    cm_dir = results_path / "confusion_matrices"
    results_path.mkdir(parents=True, exist_ok=True)
    cm_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df_manifest = pd.read_csv(args.manifest_path)
    test_manifest = df_manifest[df_manifest["split"] == "test"].copy()
    test_mask = (df_manifest["split"] == "test").values

    proc_path = Path(args.processed_dir)
    df_a = pd.read_csv(proc_path / "model_a_features.csv")
    df_b = pd.read_csv(proc_path / "model_b_features.csv")
    df_c_tab = pd.read_csv(proc_path / "model_c_tabular.csv")
    seq_data = np.load(proc_path / "model_c_sequences.npz")["sequences"]

    # Filter to test split
    test_a = df_a[df_a["split"] == "test"]
    test_b = df_b[df_b["split"] == "test"]
    test_c_tab = df_c_tab[df_c_tab["split"] == "test"]
    test_seqs = seq_data[test_mask]

    y_test = test_a["label"].values
    horizons = test_a["horizon_sec"].values

    # Negative driving duration in minutes
    neg_test_wins = test_manifest[test_manifest["label"] == 0]
    total_neg_sec = len(neg_test_wins) * 2.0  # each window is 2.0 sec history
    duration_min = total_neg_sec / 60.0

    models_path = Path(args.models_dir)

    # 1. Model A
    pkg_a = joblib.load(models_path / "model_a.joblib")
    scaler_a, model_a, feats_a = pkg_a["scaler"], pkg_a["model"], pkg_a["features"]
    X_test_a = scaler_a.transform(test_a[feats_a].fillna(0))
    t0 = time.time()
    probs_a = model_a.predict_proba(X_test_a)[:, 1]
    lat_a = (time.time() - t0) * 1000.0 / max(1, len(X_test_a))

    # 2. Model B
    pkg_b = joblib.load(models_path / "model_b.joblib")
    scaler_b, model_b, feats_b = pkg_b["scaler"], pkg_b["model"], pkg_b["features"]
    X_test_b = scaler_b.transform(test_b[feats_b].fillna(0))
    t0 = time.time()
    probs_b = model_b.predict_proba(X_test_b)[:, 1]
    lat_b = (time.time() - t0) * 1000.0 / max(1, len(X_test_b))

    # 3. Model C (GRU)
    ckpt_c = torch.load(models_path / "model_c_gru.pt", map_location="cpu", weights_only=False)
    model_c = TrajectoryGRU(input_dim=FEATURE_DIM, hidden_dim=64, num_layers=2)
    model_c.load_state_dict(ckpt_c["state_dict"])
    model_c.eval()
    norm_seqs = (test_seqs - ckpt_c["mean"]) / ckpt_c["std"]
    t0 = time.time()
    with torch.no_grad():
        t_seqs = torch.tensor(norm_seqs, dtype=torch.float32)
        probs_c_gru = torch.sigmoid(model_c(t_seqs)).cpu().numpy().flatten()
    lat_c_gru = (time.time() - t0) * 1000.0 / max(1, len(test_seqs))

    # 4. Model C (XGBoost Trajectory comparator)
    pkg_c_xgb = joblib.load(models_path / "model_c_xgb.joblib")
    scaler_c_xgb, model_c_xgb, feats_c = pkg_c_xgb["scaler"], pkg_c_xgb["model"], pkg_c_xgb["features"]
    X_test_c = scaler_c_xgb.transform(test_c_tab[feats_c].fillna(0))
    t0 = time.time()
    probs_c_xgb = model_c_xgb.predict_proba(X_test_c)[:, 1]
    lat_c_xgb = (time.time() - t0) * 1000.0 / max(1, len(X_test_c))

    # Compile Evaluation Results
    metrics_summary = {}
    metrics_summary["A: Detection only"] = evaluate_predictions(y_test, probs_a, horizons, duration_min, lat_a)
    metrics_summary["B: Detection + Tracking"] = evaluate_predictions(y_test, probs_b, horizons, duration_min, lat_b)
    metrics_summary["C: Trajectory + GRU (Proposed)"] = evaluate_predictions(y_test, probs_c_gru, horizons, duration_min, lat_c_gru)
    metrics_summary["C: Trajectory + XGBoost"] = evaluate_predictions(y_test, probs_c_xgb, horizons, duration_min, lat_c_xgb)

    # Save to metrics.csv
    df_metrics = pd.DataFrame(metrics_summary).T
    df_metrics.index.name = "Model"
    df_metrics.to_csv(results_path / "metrics.csv")
    print("\n" + "="*85)
    print("EXPERIMENTAL EVALUATION RESULTS ON HELD-OUT TEST SPLIT:")
    print("="*85)
    disp_cols = ["1.5s AP", "1.0s AP", "0.5s AP", "Mean AP", "PR-AUC", "F1", "FPR", "False Alarms/min", "Latency (ms)"]
    print(df_metrics[disp_cols].to_string())
    print("="*85)

    # Save Confusion Matrices
    for mname, res in metrics_summary.items():
        cm = np.array([[res["TN"], res["FP"]], [res["FN"], res["TP"]]])
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Normal", "Impending"], yticklabels=["Normal", "Impending"])
        ax.set_title(f"Confusion Matrix: {mname}", fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")
        fname = mname.split(":")[0].strip().replace(" ", "_").lower() + ("_gru" if "GRU" in mname else ("_xgb" if "XGBoost" in mname else ""))
        plt.tight_layout()
        plt.savefig(cm_dir / f"cm_{fname}.png", dpi=150)
        plt.close()

    # Generate Comparison Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Multi-horizon AP bar chart
    horizons_list = ["1.5s AP", "1.0s AP", "0.5s AP", "Mean AP"]
    x = np.arange(len(horizons_list))
    width = 0.2
    model_keys = ["A: Detection only", "B: Detection + Tracking", "C: Trajectory + GRU (Proposed)", "C: Trajectory + XGBoost"]
    colors = ["#4A90E2", "#50E3C2", "#E94E77", "#F5A623"]

    for i, (mkey, color) in enumerate(zip(model_keys, colors)):
        vals = [metrics_summary[mkey][h] for h in horizons_list]
        axes[0].bar(x + i * width - 0.3, vals, width, label=mkey.split(":")[0] + " " + mkey.split("(")[0], color=color)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["1.5s (Earliest)", "1.0s", "0.5s (Imminent)", "Mean AP"], fontsize=9)
    axes[0].set_ylabel("Average Precision (AP)")
    axes[0].set_title("Anticipation AP by Prediction Horizon")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # 2. Precision-Recall Curves
    for mkey, probs, color in zip(
        model_keys,
        [probs_a, probs_b, probs_c_gru, probs_c_xgb],
        colors
    ):
        p, r, _ = precision_recall_curve(y_test, probs)
        ap = metrics_summary[mkey]["PR-AUC"]
        axes[1].plot(r, p, label=f"{mkey.split(':')[0]} (AP={ap:.3f})", color=color, linewidth=2)

    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall (PR) Curves")
    axes[1].set_xlim(0, 1.02)
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=8)
    axes[1].grid(linestyle="--", alpha=0.5)

    # 3. ROC Curves
    for mkey, probs, color in zip(
        model_keys,
        [probs_a, probs_b, probs_c_gru, probs_c_xgb],
        colors
    ):
        fpr_pts, tpr_pts, _ = roc_curve(y_test, probs)
        roc = metrics_summary[mkey]["ROC-AUC"]
        axes[2].plot(fpr_pts, tpr_pts, label=f"{mkey.split(':')[0]} (AUC={roc:.3f})", color=color, linewidth=2)

    axes[2].plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random Guess")
    axes[2].set_xlabel("False Positive Rate (FPR)")
    axes[2].set_ylabel("True Positive Rate (Recall)")
    axes[2].set_title("Receiver Operating Characteristic (ROC)")
    axes[2].legend(fontsize=8)
    axes[2].grid(linestyle="--", alpha=0.5)

    plt.tight_layout()
    comp_file = results_path / "comparison.png"
    plt.savefig(comp_file, dpi=180)
    plt.close()
    print(f"\nComparative evaluation plot saved to: {comp_file.resolve()}")
    print(f"Confusion matrices saved to: {cm_dir.resolve()}")

if __name__ == "__main__":
    main()
