"""
train_temporal.py
Trains Model C: Proposed Trajectory-Based Temporal Prediction Model.
Architecture: Lightweight GRU (PyTorch) processing temporal trajectory sequences (T x D).
Also trains and evaluates an XGBoost trajectory-feature model for direct tabular comparison.
"""

import os
import sys
import argparse
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.trajectory_features import TrajectoryFeatureExtractor, FEATURE_DIM, SEQUENCE_LENGTH

# PyTorch Dataset
class TemporalTrajectoryDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# Lightweight Temporal GRU Architecture
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
        # x: (B, T, D)
        out, _ = self.gru(x)
        # Take the last time step output for causal anticipation
        last_step = out[:, -1, :]
        logits = self.classifier(last_step)
        return logits

def parse_args():
    parser = argparse.ArgumentParser(description="Train Model C: Trajectory Temporal Prediction")
    parser.add_argument("--manifest_path", type=str, default="data/manifests/window_manifest.csv", help="Window manifest path")
    parser.add_argument("--processed_dir", type=str, default="data/processed", help="Path to processed feature data")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--force_extract", action="store_true", help="Force re-extraction of trajectory features")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs for GRU")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seq_len", type=int, default=SEQUENCE_LENGTH, help="Temporal sequence length")
    return parser.parse_args()

def extract_or_load_trajectories(window_df, processed_dir, seq_len=10, force=False):
    proc_path = Path(processed_dir)
    proc_path.mkdir(parents=True, exist_ok=True)
    seq_file = proc_path / "model_c_sequences.npz"
    tab_file = proc_path / "model_c_tabular.csv"

    if seq_file.exists() and tab_file.exists() and not force:
        print(f"Loading cached trajectory data from:\n  {seq_file}\n  {tab_file}")
        npz = np.load(seq_file)
        sequences = npz["sequences"]
        df_tab = pd.read_csv(tab_file)
        return sequences, df_tab

    print(f"Extracting Model C trajectory sequences (seq_len={seq_len}) across all windows...")
    extractor = TrajectoryFeatureExtractor(seq_len=seq_len)

    sequences_list = []
    tab_records = []

    for idx, row in tqdm(window_df.iterrows(), total=len(window_df), desc="Extracting trajectories"):
        win_id = row["window_id"]
        vpath = row["file_path"]
        sf = int(row["start_frame"])
        ef = int(row["end_frame"])
        label = int(row["label"])
        split = row["split"]
        horizon = float(row["horizon_sec"])

        seq_mat, summary = extractor.extract_trajectory_sequence(vpath, sf, ef, step_frames=6)
        sequences_list.append(seq_mat)

        summary["window_id"] = win_id
        summary["split"] = split
        summary["label"] = label
        summary["horizon_sec"] = horizon
        tab_records.append(summary)

    sequences = np.array(sequences_list, dtype=np.float32)
    df_tab = pd.DataFrame(tab_records)

    np.savez_compressed(seq_file, sequences=sequences)
    df_tab.to_csv(tab_file, index=False)
    print(f"Trajectory data saved to:\n  {seq_file}\n  {tab_file}")
    return sequences, df_tab

def train_gru(model, train_loader, val_loader, epochs=30, lr=1e-3, device="cpu"):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(seqs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs, labels = seqs.to(device), labels.to(device)
                logits = model(seqs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                val_preds.extend(probs)
                val_targets.extend(labels.cpu().numpy().flatten())

        val_loss /= len(val_loader.dataset)
        val_ap = average_precision_score(val_targets, val_preds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict().copy()

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device for Model C training: {device}")

    manifest_path = Path(args.manifest_path)
    window_df = pd.read_csv(manifest_path)

    sequences, df_tab = extract_or_load_trajectories(
        window_df, args.processed_dir, seq_len=args.seq_len, force=args.force_extract
    )

    train_mask = (df_tab["split"] == "train").values
    val_mask = (df_tab["split"] == "val").values
    test_mask = (df_tab["split"] == "test").values

    labels = df_tab["label"].values.astype(np.float32)

    # Normalize sequences using train statistics
    train_seqs = sequences[train_mask]
    mean_val = np.mean(train_seqs, axis=(0, 1), keepdims=True)
    std_val = np.std(train_seqs, axis=(0, 1), keepdims=True) + 1e-6
    sequences_norm = (sequences - mean_val) / std_val

    # DataLoaders for PyTorch GRU
    train_ds = TemporalTrajectoryDataset(sequences_norm[train_mask], labels[train_mask])
    val_ds = TemporalTrajectoryDataset(sequences_norm[val_mask], labels[val_mask])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print("\n" + "="*70)
    print("TRAINING MODEL C: Lightweight GRU Temporal Trajectory Predictor")
    print("="*70)
    model_gru = TrajectoryGRU(input_dim=FEATURE_DIM, hidden_dim=64, num_layers=2, dropout=0.2).to(device)
    model_gru = train_gru(model_gru, train_loader, val_loader, epochs=args.epochs, lr=args.lr, device=device)

    # Evaluate GRU on val set
    model_gru.eval()
    with torch.no_grad():
        val_seqs_t = torch.tensor(sequences_norm[val_mask], dtype=torch.float32).to(device)
        gru_val_probs = torch.sigmoid(model_gru(val_seqs_t)).cpu().numpy().flatten()
    gru_val_ap = average_precision_score(labels[val_mask], gru_val_probs)
    gru_val_roc = roc_auc_score(labels[val_mask], gru_val_probs) if len(np.unique(labels[val_mask])) > 1 else 0.5
    print(f"\n[Model C - GRU] Val PR-AUC (AP): {gru_val_ap:.4f} | ROC-AUC: {gru_val_roc:.4f}")

    # ------------------ Train Model C (XGBoost Trajectory Comparison) ------------------
    print("\n" + "="*70)
    print("TRAINING MODEL C (Tabular Comparison): XGBoost Trajectory Predictor")
    print("="*70)
    meta_cols = ["window_id", "split", "label", "horizon_sec"]
    tab_feat_cols = [c for c in df_tab.columns if c not in meta_cols]

    scaler_tab = StandardScaler()
    X_train_tab = scaler_tab.fit_transform(df_tab.loc[train_mask, tab_feat_cols].fillna(0))
    y_train = labels[train_mask]

    X_val_tab = scaler_tab.transform(df_tab.loc[val_mask, tab_feat_cols].fillna(0))
    y_val = labels[val_mask]

    clf_xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss")
    clf_xgb.fit(X_train_tab, y_train)

    xgb_val_probs = clf_xgb.predict_proba(X_val_tab)[:, 1]
    xgb_val_ap = average_precision_score(y_val, xgb_val_probs)
    xgb_val_roc = roc_auc_score(y_val, xgb_val_probs) if len(np.unique(y_val)) > 1 else 0.5
    print(f"[Model C - XGBoost] Val PR-AUC (AP): {xgb_val_ap:.4f} | ROC-AUC: {xgb_val_roc:.4f}")

    # Save models
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "state_dict": model_gru.state_dict(),
        "mean": mean_val,
        "std": std_val,
        "feature_dim": FEATURE_DIM,
        "seq_len": args.seq_len
    }, models_dir / "model_c_gru.pt")

    joblib.dump({
        "model": clf_xgb,
        "scaler": scaler_tab,
        "features": tab_feat_cols
    }, models_dir / "model_c_xgb.joblib")

    print("\n" + "="*70)
    print(f"Model C variants successfully trained and saved:")
    print(f"  Model C (GRU)     -> {models_dir / 'model_c_gru.pt'}")
    print(f"  Model C (XGBoost) -> {models_dir / 'model_c_xgb.joblib'}")
    print("="*70)

if __name__ == "__main__":
    main()
