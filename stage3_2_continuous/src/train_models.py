"""
train_models.py
Trains all baseline and trajectory models on the 250 Train Videos (750 windows).
Evaluates and tunes alarm threshold and debounce parameters on the 50 Validation Videos (150 windows).
Saves models:
- Model A: Detection-Only (YOLOv8n)
- Model B: Detection + Tracking (ByteTrack)
- Model C: Trajectory-based XGBoost
- Model D: Trajectory-based GRU (PyTorch)
- Model E: Logistic Regression (Standardized static features)
- Model F: Center-Distance Baseline
"""

import sys
import argparse
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score
)

BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

FEATURE_DIM = 14
SEQUENCE_LENGTH = 10

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

class SeqDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    parser = argparse.ArgumentParser(description="Train Stage 3.2 Models")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading extracted window features...", flush=True)
    df_a = pd.read_csv(PROC_DIR / "model_a_features.csv")
    df_b = pd.read_csv(PROC_DIR / "model_b_features.csv")
    df_c_tab = pd.read_csv(PROC_DIR / "model_c_tabular.csv")
    df_cent = pd.read_csv(PROC_DIR / "center_features.csv")
    seq_npz = np.load(PROC_DIR / "model_c_sequences.npz")
    sequences = seq_npz["sequences"]

    df_a["video_id"] = df_a["video_id"].astype(str)
    meta_cols = ["window_id", "video_id", "split", "label", "horizon_sec", "scene", "weather"]
    feats_a = [c for c in df_a.columns if c not in meta_cols]
    feats_b = [c for c in df_b.columns if c not in meta_cols]
    feats_c_tab = [c for c in df_c_tab.columns if c not in meta_cols]
    feats_cent = [c for c in df_cent.columns if c not in meta_cols]

    train_mask = (df_a["split"] == "train").values
    val_mask = (df_a["split"] == "val").values
    test_mask = (df_a["split"] == "test").values

    y_train = df_a.loc[train_mask, "label"].values
    y_val = df_a.loc[val_mask, "label"].values

    print(f"Windows Breakdown: Train={sum(train_mask)} (125 Pos, 125 Neg videos), Val={sum(val_mask)} (25 Pos, 25 Neg videos), Test={sum(test_mask)} (50 Pos, 50 Neg videos)", flush=True)

    # ---------------------------------------------------------
    # 1. Model F: Center-Distance Baseline
    # ---------------------------------------------------------
    print("\n--- Training Model F: Center-Distance Baseline ---", flush=True)
    scaler_f = StandardScaler()
    X_tr_f = scaler_f.fit_transform(df_cent.loc[train_mask, feats_cent].fillna(0))
    X_va_f = scaler_f.transform(df_cent.loc[val_mask, feats_cent].fillna(0))
    clf_f = LogisticRegression(class_weight="balanced", random_state=42)
    clf_f.fit(X_tr_f, y_train)
    joblib.dump({"model": clf_f, "scaler": scaler_f, "features": feats_cent}, MODELS_DIR / "model_f_center.joblib")
    probs_val_f = clf_f.predict_proba(X_va_f)[:, 1]
    ap_val_f = average_precision_score(y_val, probs_val_f)
    print(f"Model F Validation AP: {ap_val_f:.4f}", flush=True)

    # ---------------------------------------------------------
    # 2. Model E: Logistic Regression Baseline
    # ---------------------------------------------------------
    print("--- Training Model E: Logistic Regression Baseline ---", flush=True)
    scaler_e = StandardScaler()
    X_tr_e = scaler_e.fit_transform(df_a.loc[train_mask, feats_a].fillna(0))
    X_va_e = scaler_e.transform(df_a.loc[val_mask, feats_a].fillna(0))
    clf_e = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    clf_e.fit(X_tr_e, y_train)
    joblib.dump({"model": clf_e, "scaler": scaler_e, "features": feats_a}, MODELS_DIR / "model_e_logreg.joblib")
    probs_val_e = clf_e.predict_proba(X_va_e)[:, 1]
    ap_val_e = average_precision_score(y_val, probs_val_e)
    print(f"Model E Validation AP: {ap_val_e:.4f}", flush=True)

    # ---------------------------------------------------------
    # 3. Model A: Detection-Only Baseline (XGBoost)
    # ---------------------------------------------------------
    print("--- Training Model A: Detection-Only Baseline (XGBoost) ---", flush=True)
    scaler_a = StandardScaler()
    X_tr_a = scaler_a.fit_transform(df_a.loc[train_mask, feats_a].fillna(0))
    X_va_a = scaler_a.transform(df_a.loc[val_mask, feats_a].fillna(0))
    clf_a = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    clf_a.fit(X_tr_a, y_train)
    joblib.dump({"model": clf_a, "scaler": scaler_a, "features": feats_a}, MODELS_DIR / "model_a.joblib")
    probs_val_a = clf_a.predict_proba(X_va_a)[:, 1]
    ap_val_a = average_precision_score(y_val, probs_val_a)
    print(f"Model A Validation AP: {ap_val_a:.4f}", flush=True)

    # ---------------------------------------------------------
    # 4. Model B: Detection + Tracking Baseline (ByteTrack + XGBoost)
    # ---------------------------------------------------------
    print("--- Training Model B: Detection + Tracking (XGBoost) ---", flush=True)
    scaler_b = StandardScaler()
    X_tr_b = scaler_b.fit_transform(df_b.loc[train_mask, feats_b].fillna(0))
    X_va_b = scaler_b.transform(df_b.loc[val_mask, feats_b].fillna(0))
    clf_b = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    clf_b.fit(X_tr_b, y_train)
    joblib.dump({"model": clf_b, "scaler": scaler_b, "features": feats_b}, MODELS_DIR / "model_b.joblib")
    probs_val_b = clf_b.predict_proba(X_va_b)[:, 1]
    ap_val_b = average_precision_score(y_val, probs_val_b)
    print(f"Model B Validation AP: {ap_val_b:.4f}", flush=True)

    # ---------------------------------------------------------
    # 5. Model C: Trajectory-based XGBoost
    # ---------------------------------------------------------
    print("--- Training Model C: Trajectory-based XGBoost ---", flush=True)
    scaler_c = StandardScaler()
    X_tr_c = scaler_c.fit_transform(df_c_tab.loc[train_mask, feats_c_tab].fillna(0))
    X_va_c = scaler_c.transform(df_c_tab.loc[val_mask, feats_c_tab].fillna(0))
    clf_c = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    clf_c.fit(X_tr_c, y_train)
    joblib.dump({"model": clf_c, "scaler": scaler_c, "features": feats_c_tab}, MODELS_DIR / "model_c_xgb.joblib")
    probs_val_c = clf_c.predict_proba(X_va_c)[:, 1]
    ap_val_c = average_precision_score(y_val, probs_val_c)
    print(f"Model C Validation AP: {ap_val_c:.4f}", flush=True)

    # ---------------------------------------------------------
    # 6. Model D: Trajectory-based GRU (PyTorch)
    # ---------------------------------------------------------
    print("--- Training Model D: Trajectory-based GRU (PyTorch) ---", flush=True)
    seq_tr = sequences[train_mask]
    seq_va = sequences[val_mask]

    mean_s = np.mean(seq_tr, axis=(0, 1), keepdims=True)
    std_s = np.std(seq_tr, axis=(0, 1), keepdims=True) + 1e-6
    norm_tr = (seq_tr - mean_s) / std_s
    norm_va = (seq_va - mean_s) / std_s

    ds_tr = SeqDataset(norm_tr, y_train)
    loader_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True)

    model_gru = TrajectoryGRU(input_dim=FEATURE_DIM, hidden_dim=64, num_layers=2)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model_gru.parameters(), lr=args.lr, weight_decay=1e-4)

    model_gru.train()
    for epoch in range(args.epochs):
        for bx, by in loader_tr:
            optimizer.zero_grad()
            out = model_gru(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    model_gru.eval()
    with torch.no_grad():
        t_va = torch.tensor(norm_va, dtype=torch.float32)
        probs_val_d = torch.sigmoid(model_gru(t_va)).cpu().numpy().flatten()
    ap_val_d = average_precision_score(y_val, probs_val_d)
    print(f"Model D Validation AP: {ap_val_d:.4f}", flush=True)

    torch.save({"state_dict": model_gru.state_dict(), "mean": mean_s, "std": std_s}, MODELS_DIR / "model_d_gru.pt")

    # ---------------------------------------------------------
    # 7. Validation-Set Threshold Tuning (Section 7)
    # ---------------------------------------------------------
    print("\n--- Tuning Alarm Threshold on Validation Set ---", flush=True)
    val_candidates = np.arange(0.30, 0.85, 0.05)
    best_config = {}
    
    models_val = {
        "Model A": probs_val_a,
        "Model B": probs_val_b,
        "Model C": probs_val_c,
        "Model D": probs_val_d,
        "Model E": probs_val_e,
        "Model F": probs_val_f
    }

    for mname, pvals in models_val.items():
        best_f1 = -1.0
        best_thresh = 0.50
        for th in val_candidates:
            preds = (pvals >= th).astype(int)
            f1 = f1_score(y_val, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = th
        best_config[mname] = {"best_threshold": round(best_thresh, 2), "val_f1": round(best_f1, 4)}
        print(f"  {mname}: Optimal Validation Threshold = {best_thresh:.2f} (F1 = {best_f1:.4f})", flush=True)

    joblib.dump(best_config, MODELS_DIR / "val_tuned_thresholds.joblib")
    print(f"Saved validation tuned thresholds -> {MODELS_DIR / 'val_tuned_thresholds.joblib'}", flush=True)

    print("\n[SUCCESS] Model training and validation tuning completed!", flush=True)

if __name__ == "__main__":
    main()
