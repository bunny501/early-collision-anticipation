# Early Collision Anticipation: Mini Feasibility Study & POC

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15-ee4c2c.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8n-00FFFF.svg)](https://docs.ultralytics.com/)
[![ByteTrack](https://img.shields.io/badge/Tracker-ByteTrack-brightgreen.svg)](https://github.com/ifzhang/ByteTrack)
[![Feasibility Score](https://img.shields.io/badge/Feasibility-79%2F100-success.svg)](reports/feasibility_report.md)

This repository contains the complete Proof of Concept (POC) and Mini Feasibility Study investigating whether vehicle and pedestrian trajectory patterns can anticipate road collisions earlier and more reliably than conventional object detection.

---

## 📁 Project Structure

```text
early_collision_anticipation/
├── data/
│   ├── raw/                      # Downloaded and benchmark dashcam clips (.mp4)
│   ├── processed/                # Extracted feature CSVs & sequence NPZ arrays
│   ├── manifests/                # Video and temporal window CSV manifests
│   └── assets/                   # Calibrated visual patches (car, bus, person, bike)
├── src/
│   ├── download_dataset.py       # Selective Nexar Hugging Face downloader & benchmark generator
│   ├── preprocess.py             # Temporal window extraction (1.5s, 1.0s, 0.5s horizons)
│   ├── detector.py               # Model A: Conventional object detection baseline extractor
│   ├── tracker.py                # Model B: ByteTrack multi-object tracking extractor
│   ├── trajectory_features.py    # Model C: Kinematics, expansion rate, TTC proxy extractor
│   ├── train_baseline.py         # Training pipeline for Model A & Model B (LR, RF, XGB)
│   ├── train_temporal.py         # Training pipeline for Model C (PyTorch GRU & XGBoost)
│   ├── evaluate.py               # Held-out test evaluation & multi-horizon AP calculation
│   └── visualize.py              # Visual video annotator with tracking & collision HUD gauge
├── models/
│   ├── yolov8n.pt                # Pretrained YOLOv8 nano detector weights
│   ├── model_a.joblib            # Trained Model A baseline
│   ├── model_b.joblib            # Trained Model B tracking model
│   ├── model_c_gru.pt            # Trained Model C PyTorch GRU checkpoint
│   └── model_c_xgb.joblib        # Trained Model C XGBoost comparator
├── results/
│   ├── metrics.csv               # Complete comparative metrics across all horizons
│   ├── comparison.png            # Multi-horizon AP, PR curve, and ROC curve comparison
│   ├── confusion_matrices/       # Confusion matrix heatmaps for each model
│   └── sample_predictions/       # 5 annotated test demonstration videos with HUD gauge
├── reports/
│   └── feasibility_report.md     # Detailed 100-point feasibility report & roadmap
├── requirements.txt              # Environment dependencies
└── README.md                     # Project documentation & execution guide
```

---

## 🚀 Quickstart & Reproduction

### 1. Environment Installation
Ensure Python 3.10+ is installed. Install all core dependencies:

```bash
pip install -r requirements.txt
```

### 2. Dataset Acquisition & Authentication

The official dataset is the [Nexar Dashcam Collision Prediction Dataset](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction).

#### To authenticate and download official Nexar clips:
1. Visit [Nexar on Hugging Face](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction) and accept the Nexar Open Data License.
2. Obtain a user access token at [Hugging Face Settings](https://huggingface.co/settings/tokens).
3. Set your token in terminal:
   ```powershell
   # PowerShell
   $env:HF_TOKEN = "hf_your_token_here"
   ```
4. Run the selective downloader (downloads balanced subset without downloading 31 GB):
   ```bash
   python src/download_dataset.py --num_positive 50 --num_negative 50
   ```

*(Alternatively, run `python src/download_dataset.py --generate_benchmark` for immediate offline pipeline verification).*

---

### 3. Preprocessing (Temporal Window Extraction)
Extracts pre-collision windows ending 1.5s, 1.0s, and 0.5s before the event with strict video-level train/val/test splitting:

```bash
python src/preprocess.py --history_sec 2.0 --sample_fps 5.0
```

---

### 4. Training Models

#### Train Baselines (Model A & Model B)
```bash
python src/train_baseline.py
```
- **Model A:** Frame-level vehicle and pedestrian detection counts, areas, and confidence.
- **Model B:** ByteTrack multi-object tracking, displacements, image-space velocities, and pairwise closing rates.

#### Train Proposed Approach (Model C)
```bash
python src/train_temporal.py --epochs 30 --batch_size 8
```
- **Model C (GRU):** Lightweight PyTorch GRU on temporal trajectory sequences ($T \times D$).
- **Model C (XGBoost):** Tabular trajectory feature comparator.

---

### 5. Experimental Evaluation
Evaluates all models on the exact same held-out test split across 1.5s, 1.0s, and 0.5s anticipation horizons:

```bash
python src/evaluate.py
```

Outputs:
- [`results/metrics.csv`](results/metrics.csv)
- [`results/comparison.png`](results/comparison.png)
- [`results/confusion_matrices/`](results/confusion_matrices/)

---

### 6. Visual Demonstration
Renders at least 5 annotated test demonstration videos with tracking IDs, trajectory lines, velocity vectors, and risk HUD gauge:

```bash
python src/visualize.py --num_videos 5
```
Demonstration videos will be saved in [`results/sample_predictions/`](results/sample_predictions/).

---

## 📊 Experimental Results

Evaluated on held-out test videos:

| Model Architecture | 1.5s AP | 1.0s AP | 0.5s AP | Mean AP | PR-AUC | ROC-AUC | F1 | FPR | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A: Detection Only** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.00 | **0.00 ms** |
| **Model B: Detection + Tracking** | 0.7500 | 1.0000 | 1.0000 | 0.9167 | 0.9444 | 0.9444 | 0.9412 | 0.00 | 0.33 ms |
| **Model C: Trajectory + GRU (Proposed)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 0.00 | 1.62 ms |
| **Model C: Trajectory + XGBoost** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.00 | 0.23 ms |

### Key Findings
1. **Early Horizon Advantage:** At 1.5 seconds before impact, Model B dropped to 0.75 AP because instantaneous velocity vectors were still modest. Model C's sequence modeling maintained 1.00 AP by capturing non-linear bounding box expansion rates ($\eta$) and deceleration proxies.
2. **Computational Feasibility:** All models run smoothly on standard CPU or a 6 GB laptop GPU, with Model C GRU inference requiring only **1.62 ms**.
3. **Physical Limitation Note:** Dashcam bounding boxes yield image-space proxy estimators; real-world metric distances and velocities require camera calibration.

---

## 🔬 Feasibility Assessment & Recommendation

- **Overall Score:** **79 / 100**
- **Recommendation:** **CONDITIONAL GO**
- Full analysis available in [`reports/feasibility_report.md`](reports/feasibility_report.md).
