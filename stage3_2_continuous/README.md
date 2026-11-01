# Stage 3.2: Expanded Dataset & Continuous Collision Anticipation Evaluation

## Overview
This stage scales the empirical evaluation of **Early Collision Anticipation using Vehicle and Pedestrian Trajectories** on the official [Nexar Collision Prediction Dataset](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction). It directly rectifies the methodological limitations discovered during the Stage 3.1 independent audit.

### Research Question
> **Can motion and trajectory-based features predict impending vehicle collisions or near-collisions earlier and more reliably than conventional object detection on authentic dashcam video?**

---

## Key Methodological Advancements in Stage 3.2

1. **Expanded Statistical Scale:**
   - Expanded dataset from 100 to **400 authentic dashcam videos** (200 positive collisions/near-collisions, 200 negative normal driving clips).
   - Strict video-level partitioning:
     - **Train Split:** 250 videos (125 Pos, 125 Neg) $\rightarrow$ 750 observation windows.
     - **Validation Split:** 50 videos (25 Pos, 25 Neg) $\rightarrow$ 150 observation windows.
     - **Held-Out Test Split:** **100 independent videos** (50 Pos, 50 Neg) $\rightarrow$ 300 observation windows.
   - Programmatically verified zero cross-split leakage: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.

2. **Stratified Negative Sampling:**
   - Mitigated Stage 3's scene imbalance by stratifying negative videos to mirror the positive distribution:
     - Urban: 129 Pos vs 129 Neg
     - Highway: 45 Pos vs 45 Neg
     - Suburban: 16 Pos vs 16 Neg
     - Other: 6 Pos vs 6 Neg

3. **Strict Horizon-Specific Evaluation:**
   - Evaluated independently at $1.5\text{s}$, $1.0\text{s}$, and $0.5\text{s}$ prior to collision.
   - Prohibited pooling of negative windows across horizons.
   - Enforced mathematical identity assertion: window-level horizon AP $\equiv$ video-level horizon AP.

4. **Continuous Sliding-Window Inference & False Alarm Rate:**
   - Streaming sliding window ($T_{\text{obs}} = 2.0\text{s}$, 10 frames @ 5 FPS) evaluated continuously over the full 40-second duration of test videos ($33.3\text{ minutes}$ of negative footage).
   - Validation-tuned threshold $\tau$ and debounce persistence ($D_{\min} \ge 0.4\text{s}$) to measure realistic **False Alarms per Minute**.

5. **Objective Lead Time Evaluation:**
   - Measured time prior to physical impact event ($t_e - t_{\text{warn}}$).
   - Fully retracted ungrounded claims of "warning earlier than human perception."

6. **2,000-Iteration Video-Level Bootstrap:**
   - Stratified bootstrap resampling across the 100 independent test videos to produce rigorous 95% Confidence Intervals and two-tailed empirical $p$-values.

---

## Directory Structure

```
stage3_2_continuous/
├── README.md                           # This reproducibility guide
├── dataset_manifest.csv                # Complete 400-video manifest
├── train_manifest.csv                  # 250 training videos
├── validation_manifest.csv             # 50 validation videos
├── test_manifest.csv                   # 100 held-out test videos
├── sampling_bias_analysis.csv          # Scene stratification audit
├── continuous_predictions.csv          # Rolling risk scores per video timestamp
├── horizon_metrics.csv                 # Primary AP, PR-AUC, ROC-AUC at 1.5s, 1.0s, 0.5s
├── false_alarm_metrics.csv             # Debounced false alarm episodes per minute
├── lead_time_metrics.csv               # Early warning lead time relative to impact event
├── latency_benchmark.csv               # Native 720p CPU latency breakdown
├── bootstrap_confidence_intervals.csv  # 2,000 bootstrap resamples with 95% CIs
├── model_comparison.csv                # Overall model benchmark table
├── failure_cases.csv                   # Diagnostic false positive and false negative analysis
├── stage3_2_report.md                  # Comprehensive scientific report and verdict
├── data/
│   ├── raw/                            # Original MP4 clips (positive/ & negative/)
│   ├── manifests/                      # window_manifest.csv (1,200 observation windows)
│   └── processed/                      # Extracted feature CSVs and sequential NPZ
├── models/                             # Trained model checkpoints & val_tuned_thresholds.joblib
├── visualizations/                     # Publication figures (PNG)
└── src/
    ├── create_dataset_manifests.py     # Dataset download, stratification & splitting
    ├── create_windows.py               # Generates 1,200 temporal windows
    ├── fast_extract_features.py        # Video-batched YOLOv8n + ByteTrack feature extractor
    ├── train_models.py                 # Trains Models A-F and tunes thresholds on Val
    ├── eval_horizons.py                # Primary horizon evaluation & 2,000 bootstrap CIs
    ├── run_continuous_inference.py     # Full continuous sliding window inference & false alarm rate
    ├── benchmark_latency.py            # Component-by-component CPU latency benchmarking
    └── generate_visualizations.py      # Generates summary plots
```

---

## Reproduction Instructions

### 1. Environment & Token Setup
Ensure your Hugging Face authentication token with access to `nexar-ai/nexar_collision_prediction` is set:
```bash
$env:HF_TOKEN = "your_token_here"
```

### 2. Dataset Ingestion & Window Creation
```bash
python src/create_dataset_manifests.py
python src/create_windows.py
```

### 3. Feature Extraction
```bash
python src/fast_extract_features.py
```

### 4. Model Training & Validation Tuning
```bash
python src/train_models.py
```

### 5. Primary Horizon & Bootstrap Evaluation
```bash
python src/eval_horizons.py
```

### 6. Continuous Sliding-Window Evaluation
```bash
python src/run_continuous_inference.py
```

### 7. Latency Benchmarking & Visualization
```bash
python src/benchmark_latency.py
python src/generate_visualizations.py
```
