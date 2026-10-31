"""
train_baseline.py
Trains and evaluates:
- Model A: Conventional Object Detection Baseline (Detection-only features + Classifier).
- Model B: Detection + ByteTrack Multi-Object Tracking (Detection + Kinematic/Expansion/Pairwise features + Classifier).
Caches feature sets to data/processed/ for reproducible and fast experimentation.
"""

import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, classification_report

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.detector import ObjectDetector
from src.tracker import MultiObjectTracker

def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline collision prediction models (Model A & B)")
    parser.add_argument("--manifest_path", type=str, default="data/manifests/window_manifest.csv", help="Window manifest path")
    parser.add_argument("--processed_dir", type=str, default="data/processed", help="Path to save processed feature CSVs")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory to save trained models")
    parser.add_argument("--force_extract", action="store_true", help="Force re-extraction of features")
    parser.add_argument("--step_frames", type=int, default=6, help="Frame sampling stride")
    return parser.parse_args()

def extract_or_load_features(window_df, processed_dir, force=False, step_frames=6):
    proc_path = Path(processed_dir)
    proc_path.mkdir(parents=True, exist_ok=True)
    feat_a_file = proc_path / "model_a_features.csv"
    feat_b_file = proc_path / "model_b_features.csv"

    if feat_a_file.exists() and feat_b_file.exists() and not force:
        print(f"Loading cached features from:\n  {feat_a_file}\n  {feat_b_file}")
        df_a = pd.read_csv(feat_a_file)
        df_b = pd.read_csv(feat_b_file)
        return df_a, df_b

    print("Extracting Model A (Detection) and Model B (Tracking) features across all windows...")
    detector = ObjectDetector()
    tracker = MultiObjectTracker()

    records_a = []
    records_b = []

    for idx, row in tqdm(window_df.iterrows(), total=len(window_df), desc="Extracting features"):
        win_id = row["window_id"]
        vpath = row["file_path"]
        sf = int(row["start_frame"])
        ef = int(row["end_frame"])
        label = int(row["label"])
        split = row["split"]
        horizon = float(row["horizon_sec"])

        # Model A: Detection-only features
        feat_a = detector.extract_window_features(vpath, sf, ef, step_frames=step_frames)
        feat_a["window_id"] = win_id
        feat_a["split"] = split
        feat_a["label"] = label
        feat_a["horizon_sec"] = horizon
        records_a.append(feat_a)

        # Model B: Tracking features + Detection features
        track_feat, _ = tracker.process_window(vpath, sf, ef, step_frames=step_frames)
        feat_b = {**feat_a, **track_feat}
        records_b.append(feat_b)

    df_a = pd.DataFrame(records_a)
    df_b = pd.DataFrame(records_b)

    df_a.to_csv(feat_a_file, index=False)
    df_b.to_csv(feat_b_file, index=False)
    print(f"Features saved to:\n  {feat_a_file}\n  {feat_b_file}")
    return df_a, df_b

def train_and_eval_model(X_train, y_train, X_val, y_val, model_type="xgb", model_name="Model"):
    if model_type == "lr":
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    elif model_type == "rf":
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced", random_state=42)
    elif model_type == "xgb":
        clf = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=1.0,
            random_state=42,
            eval_metric="logloss"
        )
    else:
        raise ValueError(f"Unknown model_type {model_type}")

    clf.fit(X_train, y_train)

    val_probs = clf.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)

    ap = average_precision_score(y_val, val_probs)
    roc = roc_auc_score(y_val, val_probs) if len(np.unique(y_val)) > 1 else 0.5
    f1 = f1_score(y_val, val_preds, zero_division=0)

    print(f"\n[{model_name}] Val Results ({model_type.upper()}):")
    print(f"  Average Precision (PR-AUC): {ap:.4f}")
    print(f"  ROC-AUC:                   {roc:.4f}")
    print(f"  F1 Score:                  {f1:.4f}")
    return clf, ap, roc, f1

def main():
    args = parse_args()
    manifest_path = Path(args.manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Window manifest not found: {manifest_path}")

    window_df = pd.read_csv(manifest_path)
    df_a, df_b = extract_or_load_features(
        window_df, args.processed_dir, force=args.force_extract, step_frames=args.step_frames
    )

    meta_cols = ["window_id", "split", "label", "horizon_sec"]

    # ------------------ Train Model A (Detection Only) ------------------
    print("\n" + "="*70)
    print("TRAINING MODEL A: Conventional Object Detection Baseline")
    print("="*70)
    feat_cols_a = [c for c in df_a.columns if c not in meta_cols]

    train_mask_a = df_a["split"] == "train"
    val_mask_a = df_a["split"] == "val"

    scaler_a = StandardScaler()
    X_train_a = scaler_a.fit_transform(df_a.loc[train_mask_a, feat_cols_a].fillna(0))
    y_train_a = df_a.loc[train_mask_a, "label"].values

    X_val_a = scaler_a.transform(df_a.loc[val_mask_a, feat_cols_a].fillna(0))
    y_val_a = df_a.loc[val_mask_a, "label"].values

    # Train Logistic Regression and XGBoost, select best
    clf_a_lr, ap_a_lr, _, _ = train_and_eval_model(X_train_a, y_train_a, X_val_a, y_val_a, "lr", "Model A")
    clf_a_xgb, ap_a_xgb, _, _ = train_and_eval_model(X_train_a, y_train_a, X_val_a, y_val_a, "xgb", "Model A")
    best_clf_a = clf_a_xgb if ap_a_xgb >= ap_a_lr else clf_a_lr
    best_type_a = "xgb" if ap_a_xgb >= ap_a_lr else "lr"

    # ------------------ Train Model B (Detection + Tracking) ------------------
    print("\n" + "="*70)
    print("TRAINING MODEL B: Object Detection + ByteTrack Multi-Object Tracking")
    print("="*70)
    feat_cols_b = [c for c in df_b.columns if c not in meta_cols]

    train_mask_b = df_b["split"] == "train"
    val_mask_b = df_b["split"] == "val"

    scaler_b = StandardScaler()
    X_train_b = scaler_b.fit_transform(df_b.loc[train_mask_b, feat_cols_b].fillna(0))
    y_train_b = df_b.loc[train_mask_b, "label"].values

    X_val_b = scaler_b.transform(df_b.loc[val_mask_b, feat_cols_b].fillna(0))
    y_val_b = df_b.loc[val_mask_b, "label"].values

    # Train Random Forest and XGBoost, select best
    clf_b_rf, ap_b_rf, _, _ = train_and_eval_model(X_train_b, y_train_b, X_val_b, y_val_b, "rf", "Model B")
    clf_b_xgb, ap_b_xgb, _, _ = train_and_eval_model(X_train_b, y_train_b, X_val_b, y_val_b, "xgb", "Model B")
    best_clf_b = clf_b_xgb if ap_b_xgb >= ap_b_rf else clf_b_rf
    best_type_b = "xgb" if ap_b_xgb >= ap_b_rf else "rf"

    # Save models and scalers
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump({"model": best_clf_a, "scaler": scaler_a, "features": feat_cols_a, "type": best_type_a}, models_dir / "model_a.joblib")
    joblib.dump({"model": best_clf_b, "scaler": scaler_b, "features": feat_cols_b, "type": best_type_b}, models_dir / "model_b.joblib")

    print("\n" + "="*70)
    print(f"Baselines successfully trained and saved:")
    print(f"  Model A -> {models_dir / 'model_a.joblib'}")
    print(f"  Model B -> {models_dir / 'model_b.joblib'}")
    print("="*70)

if __name__ == "__main__":
    main()
