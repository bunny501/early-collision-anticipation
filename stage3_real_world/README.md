# Stage 3: Real-World Validation — Nexar Collision Prediction Dataset
**Project:** Early Collision Anticipation via Vehicle and Pedestrian Trajectory Patterns  
**Evaluation Phase:** Stage 3 Authentic Real-World Dashcam Validation  
**Dataset:** Official Nexar Dashcam Collision Prediction Dataset (`nexar-ai/nexar_collision_prediction`)  
**Date:** October 2026  

---

## 1. Overview & Research Objective

Stage 3 tests whether trajectory and motion pattern modeling (kinematics, optical expansion rate, Time-to-Contact proxies, and recurrent sequence memory) genuinely outperforms conventional object detection and tracking on **authentic dashcam footage** from the official Nexar dataset.

This directly addresses the critical finding from Stage 2:
> **Stage 2 Finding:** On calibrated synthetic benchmarks, positive and negative events were linearly separable by distance to image center, producing artificial 1.00 AP scores for static bounding-box features.  
> **Stage 3 Mission:** Test on real-world urban and highway traffic to determine whether center-position shortcuts break, and whether trajectory modeling delivers a statistically validated performance gain across 1.5s, 1.0s, and 0.5s anticipation horizons.

---

## 2. Directory Structure

```
stage3_real_world/
├── README.md                      # Reproduction instructions and research summary
├── dataset_inspection.md          # Dataset structure, license, schema, and sampling
├── split_manifest.csv             # Video-level split assignments (100 independent videos)
├── stage3_metrics.csv             # Comprehensive multi-horizon metrics and bootstrap CIs
├── per_video_predictions.csv      # Video-level predictions across all models and horizons
├── ablation_results.csv           # 7-tier systematic feature ablation study
├── warning_analysis.csv           # Early warning times, lead times, and human alert comparison
├── latency_report.md              # Hardware specs and end-to-end processing latency profile
├── stage3_report.md               # Final scientific synthesis and Go/No-Go recommendation
├── data/
│   ├── raw/                       # 50 positive + 50 negative real-world MP4 videos
│   ├── manifests/                 # window_manifest.csv (temporal observation windows)
│   └── processed/                 # Extracted feature CSVs and trajectory sequence tensors
├── models/                        # Saved model checkpoints (Model A, B, C GRU, C XGB)
├── visualizations/                # Multi-horizon AP, PR curves, and shortcut distribution plots
└── src/
    ├── download_nexar_subset.py   # Authenticated download & video-level stratification
    ├── create_windows.py          # Observation window generator across 1.5s, 1.0s, 0.5s
    ├── extract_features.py        # Single-pass YOLO + ByteTrack + Ego-flow feature extraction
    └── train_and_evaluate.py      # Model training, ablations, bootstrap CIs, and metrics
```

---

## 3. Quickstart & Reproducibility Guide

### Step 1: Set Hugging Face Authentication Token
Ensure your Hugging Face account has accepted the Nexar Open Data License at `nexar-ai/nexar_collision_prediction`. Export the token:

```powershell
$env:HF_TOKEN = "your_hf_token_here"
```

### Step 2: Download the 100 Video Subset & Create Split Manifest
```powershell
python stage3_real_world/src/download_nexar_subset.py --num_pos 50 --num_neg 50 --seed 42
```

### Step 3: Generate Temporal Observation Windows
```powershell
python stage3_real_world/src/create_windows.py --obs_duration 2.0
```

### Step 4: Extract Detection, Tracking & Trajectory Features
```powershell
python stage3_real_world/src/extract_features.py --step_frames 6
```

### Step 5: Train Models, Run Ablations & Evaluate All Metrics
```powershell
python stage3_real_world/src/train_and_evaluate.py --epochs 35 --lr 0.001
```

All figures, metrics tables, and evaluation reports will be automatically generated and saved in `stage3_real_world/`.
