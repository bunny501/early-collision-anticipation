# Stage 3.2 Implementation Plan: Expanded Dataset & Continuous Collision Anticipation

**Research Question:**  
*Can motion and trajectory-based features predict impending vehicle collisions or near-collisions earlier and more reliably than conventional object detection on authentic dashcam video?*

**Execution Directory:** `stage3_2_continuous/`  
**Date:** November 2026  
**Auditor Alignment:** Incorporates all corrections and recommendations from the Stage 3.1 independent audit.

---

## 1. High-Level Architecture & Phase Breakdown

```
Phase 1: Dataset Expansion & Stratified Video-Level Splitting (400 videos: 200 Pos, 200 Neg)
   │
   ├── Phase 2: Horizon-Specific Dataset Manifests (Train: 250, Val: 50, Test: 100 videos)
   │
   ├── Phase 3: Model Training (Models A, B, C, D, E, F on 250 Train Videos)
   │
   ├── Phase 4: Validation Set Optimization (Tuning Alarm Threshold & Debounce Persistence on 50 Val Videos)
   │
   ├── Phase 5: Continuous Sliding-Window Inference across Full Video Durations (100 Test Videos)
   │
   ├── Phase 6: Primary Metric & Horizon Evaluation (AP at 1.5s, 1.0s, 0.5s on 100 Independent Test Videos)
   │
   ├── Phase 7: Debounced False Alarm Episode Analysis (Continuous Negative Video Footage)
   │
   ├── Phase 8: Early Warning Lead Time Analysis (Time Before Physical Impact Event)
   │
   ├── Phase 9: Statistical Bootstrap Evaluation (2,000 Resamples, 95% CIs, Scene Stratification)
   │
   └── Phase 10: Latency Benchmarking, Interactive Visualization Dashboard & Comprehensive Report
```

---

## 2. Dataset Ingestion & Video-Level Stratification

### 2.1. Target Scale
* **Positive Videos ($N=200$):** Collisions and near-collisions from `train/positive/metadata.csv` (Nexar ground truth `label=1`).
* **Negative Videos ($N=200$):** Normal, collision-free driving clips from `train/negative/metadata.csv` (Nexar ground truth `label=0`).
* **Total Videos:** **400 independent real-world dashcam videos** ($\approx 16,000\text{ seconds}$ / $4.44\text{ hours}$ of footage).

### 2.2. Stratification Strategy
To reduce scene distribution mismatch between positive and negative clips:
* Positive videos are predominantly Urban (63.3%), Highway (22.8%), and Suburban (10.0%).
* Negative videos will be stratified to match the scene distribution of the positive set as closely as possible, preserving natural highway cut-ins, urban intersections, and lighting/weather conditions.
* Seed is fixed at `seed = 42` for strict reproducibility.

### 2.3. Partitioning Protocol (Disjoint Video-Level Split)
* **Train Set (62.5%):** 125 Positive + 125 Negative = **250 videos**
* **Validation Set (12.5%):** 25 Positive + 25 Negative = **50 videos**
* **Held-Out Test Set (25.0%):** 50 Positive + 50 Negative = **100 independent videos**
* Programmatic assertion:
  $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset$$

---

## 3. Continuous Sliding-Window Inference & Feature Extraction

### 3.1. Temporal Mechanics
* **Input Frame Stride:** $step\_frames = 6$ (corresponding to $5\text{ FPS}$ sampling at nominal $30\text{ FPS}$).
* **Temporal Observation Window:** $T_{\text{obs}} = 2.0\text{ seconds}$ (10 sampled frames).
* **Sliding Stride:** Stride of $0.5\text{ seconds}$ (evaluated continuously every $0.5\text{s}$ over the complete duration of each 40-second video).
* For each 40-second video, continuous inference produces $\approx 76$ temporal prediction steps.
* Over 100 test videos, continuous evaluation yields $\approx 7,600$ full-pipeline predictions.

### 3.2. Six Comparative Models
1. **Model A (Detection Only):** YOLOv8n object bounding boxes, counts, areas, and bottom-y positions.
2. **Model B (Detection + Tracking):** YOLOv8n + ByteTrack Kalman kinematic velocities and speeds.
3. **Model C (Trajectory XGBoost):** 97 engineered trajectory features (velocities, accelerations, optical expansion rate, TTC proxy, ego-motion dy).
4. **Model D (Trajectory GRU):** PyTorch 2-layer GRU over $10 \times 14$ sequential trajectory tensors.
5. **Model E (Logistic Regression):** Regularized linear classifier on standardized static features.
6. **Model F (Center-Distance Baseline):** Direct ego-lane geometric baseline (`closest_dist_to_center_min`, `center_corridor_count_mean`).

---

## 4. Addressing Stage 3.1 Audit Weaknesses

1. **Statistical Power:** Scaling the test set from $N=16$ to $N=100$ independent videos (50 pos, 50 neg) narrows bootstrap confidence intervals by $\approx 2.5\times$, allowing statistically sound conclusions ($p < 0.05$).
2. **Proper Horizon AP Protocol:** For each horizon ($1.5\text{s}, 1.0\text{s}, 0.5\text{s}$), evaluate 50 positive videos against 50 negative videos at an aligned temporal reference. No pooling of negative windows across horizons.
3. **Continuous False Alarm Rate:** Continuously evaluate the full $50 \times 40\text{s} = 33.3\text{ minutes}$ of negative test video. Implement a hysteresis debounce filter ($\tau_{\text{high}}, \tau_{\text{low}}$) tuned strictly on validation data to report verified **alarm episodes per minute**.
4. **Objective Lead Time:** Measure lead time strictly relative to physical collision timestamp ($t_e - t_{\text{warn}}$), entirely avoiding the "human perception" terminology.
5. **Accurate Hardware Latency:** Measure component-by-component CPU latency at native 720p with warm-up procedures.

---

## 5. Required Deliverables Matrix

1. `dataset_manifest.csv`
2. `train_manifest.csv`
3. `validation_manifest.csv`
4. `test_manifest.csv`
5. `sampling_bias_analysis.csv`
6. `continuous_predictions.csv`
7. `horizon_metrics.csv`
8. `false_alarm_metrics.csv`
9. `lead_time_metrics.csv`
10. `latency_benchmark.csv`
11. `bootstrap_confidence_intervals.csv`
12. `model_comparison.csv`
13. `failure_cases.csv`
14. `stage3_2_report.md`
15. `README.md`
16. Updated visualization/dashboard scripts and saved model weights.
