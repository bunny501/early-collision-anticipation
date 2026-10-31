# Second-Stage Validation: Comprehensive Audit Report
**Project:** Early Collision Anticipation via Vehicle and Pedestrian Trajectory Patterns  
**Evaluation Stage:** Stage 2 Rigorous Validation, Data Audit, and Feature Ablation  
**Date:** October 2026  
**Status:** Completed & Empirically Verified  

---

## 1. Executive Summary & Core Conclusion

This audit re-evaluates the Proof-of-Concept (POC) and Mini Feasibility Study for Early Collision Anticipation. The central objective of this validation was to answer a fundamental scientific question:

> **Core Research Question:** Does vehicle and pedestrian trajectory information genuinely provide earlier and more reliable collision anticipation compared with conventional object detection, or were earlier findings an artifact of data leakage, evaluation mechanics, or dataset shortcuts?

### Primary Verdict: **INCONCLUSIVE ON BENCHMARK / NOT STATISTICALLY PROVEN**
1. **No Data Leakage in Code:** There is **zero programmatic data leakage** in the pipeline. Ground-truth event timestamps, alert timestamps, video filenames, and class labels were strictly isolated from the feature sets of Model A, Model B, and Model C.
2. **Identification of a Critical Dataset Shortcut:** The reported perfect scores (1.00 Mean AP across multiple models) are largely caused by an **unintended geometric shortcut in the synthetic benchmark generator**. In the benchmark dataset, positive collision scenarios place target vehicles and pedestrians directly along the host vehicle's central corridor ($\Delta x_{\text{center}} < 0.14$), whereas negative driving scenarios place lead vehicles laterally offset ($\Delta x_{\text{center}} \approx 0.36$) and sidewalk pedestrians at $x=180$. Consequently, static bounding-box features alone achieve 1.00 AP without any motion or trajectory information.
3. **Ablation Findings:** In our 6-tier ablation study, **Tier 1 (Bounding-box features only)** achieved an identical 1.00 Mean AP to **Tier 6 (Complete Trajectory GRU)**. This proves that on the current benchmark, higher-order kinematic and trajectory features (velocities, accelerations, optical expansion rates, TTC proxies) do not demonstrate an empirical performance delta over static bounding-box features.
4. **Genuine Nuance in Pedestrian Anticipation:** The only test case where baseline tracking suffered was `bench_pos_0012` (the sole pedestrian near-miss scenario in the test set). Here, Model B (instantaneous velocity features) failed at the 1.5s horizon ($P = 0.0238$, False Negative), whereas Model C (GRU sequence modeling) maintained detection ($P = 0.9998$) by accumulating gradual expansion cues over the multi-frame history.
5. **Sample Size & Split Independence Limitations:** The held-out test split contains only **6 independent videos** (3 positive, 3 negative). While evaluating 3 temporal windows per video produced 18 window samples, these windows heavily overlap temporally within each video ($75\%$ overlap between 1.5s and 1.0s horizons). Treating them as 18 independent observations inflates effective sample size and statistical confidence.
6. **Dataset Access Limitation:** The official Nexar Dashcam Dataset on Hugging Face (`nexar-ai/nexar_collision_prediction`) is strictly gated under the Nexar Open Data License and requires an authenticated user token (`HF_TOKEN`). No token was present in the execution environment; therefore, scaling to the 100 positive and 100 negative real-world video set was prevented by upstream access gating.

---

## 2. Audit of Existing Experiment & Dataset Splits (Task 1)

### 2.1. Video-Level Dataset Split Verification
We inspected `data/manifests/dataset_manifest.csv` to verify video partition integrity across the `train` (70%), `val` (15%), and `test` (15%) splits:

| Split | Positive Videos ($N$) | Negative Videos ($N$) | Total Videos | Video IDs Assigned |
| :--- | :---: | :---: | :---: | :--- |
| **Train** | 14 | 14 | 28 | `bench_pos_0003`, `0004`, `0005`, `0006`, `0007`, `0008`, `0010`, `0011`, `0013`, `0014`, `0015`, `0016`, `0017`, `0020`<br>`bench_neg_0001`, `0002`, `0003`, `0004`, `0006`, `0009`, `0010`, `0012`, `0014`, `0016`, `0017`, `0018`, `0019`, `0020` |
| **Validation** | 3 | 3 | 6 | `bench_pos_0001`, `0018`, `0019`<br>`bench_neg_0008`, `0013`, `0015` |
| **Held-Out Test** | 3 | 3 | 6 | `bench_pos_0002`, `0009`, `0012`<br>`bench_neg_0005`, `0007`, `0011` |
| **TOTAL** | **20** | **20** | **40** | **Strict Disjoint Partition Confirmed** |

**Verification Result:**
- $\text{Train} \cap \text{Val} = \emptyset$
- $\text{Val} \cap \text{Test} = \emptyset$
- $\text{Train} \cap \text{Test} = \emptyset$
- **Zero video-level contamination across splits.**

### 2.2. Temporal Window Generation & Overlap Audit
In `data/manifests/window_manifest.csv`:
- Each video was processed into 3 temporal windows of duration $T = 2.0\text{ seconds}$, sampled at $5\text{ FPS}$ (10 frames per window).
- Positive windows end at horizons of $1.5\text{s}$, $1.0\text{s}$, and $0.5\text{s}$ prior to the annotated collision timestamp:
  $$\text{Window}_{1.5\text{s}} = [t_{\text{event}} - 3.5\text{s}, \, t_{\text{event}} - 1.5\text{s}]$$
  $$\text{Window}_{1.0\text{s}} = [t_{\text{event}} - 3.0\text{s}, \, t_{\text{event}} - 1.0\text{s}]$$
  $$\text{Window}_{0.5\text{s}} = [t_{\text{event}} - 2.5\text{s}, \, t_{\text{event}} - 0.5\text{s}]$$
- **Temporal Overlap Within Positive Videos:** $\text{Window}_{1.5\text{s}}$ and $\text{Window}_{1.0\text{s}}$ overlap by $1.5\text{ seconds}$ ($75\%$ shared visual content). $\text{Window}_{1.0\text{s}}$ and $\text{Window}_{0.5\text{s}}$ also overlap by $1.5\text{ seconds}$. $\text{Window}_{1.5\text{s}}$ and $\text{Window}_{0.5\text{s}}$ overlap by $1.0\text{ second}$ ($50\%$).
- **Temporal Non-Overlap Across Splits:** Because video IDs never cross splits, no overlapping window ever crossed between train, validation, or test sets.
- **Evaluation Independence Caveat:** The evaluation protocol in Phase 1 pooled the 9 positive test windows and 9 negative test windows as $N=18$ independent samples. Because the 3 positive windows from each video share up to $75\%$ temporal overlap, the test samples are statistically dependent. A model that recognizes the road or object in video `bench_pos_0002` at $1.5\text{s}$ will trivially recognize the same object in the overlapping $1.0\text{s}$ and $0.5\text{s}$ windows.

### 2.3. Feature & Label Leakage Check
We inspected all columns in:
- `model_a_features.csv` (42 feature columns + 4 metadata columns)
- `model_b_features.csv` (55 feature columns + 4 metadata columns)
- `model_c_tabular.csv` (97 feature columns + 4 metadata columns)
- `model_c_sequences.npz` ($120 \times 10 \times 14$ array)

**Findings:**
- Excluded metadata: `window_id`, `split`, `label`, `horizon_sec`.
- Target metadata (`time_of_event`, `time_of_alert`, `file_path`, `duration_sec`, `video_fps`) were completely omitted from training feature matrices.
- The input features consist solely of image-space bounding box dimensions, normalized coordinates, tracked velocities, accelerations, optical expansion rates, and relative pairwise distances.
- **Verdict: No direct label or timestamp leakage exists.**

### 2.4. Statistical Reproducibility Check
Running `python src/evaluate.py` on the held-out test split with the saved models (`model_a.joblib`, `model_b.joblib`, `model_c_gru.pt`, `model_c_xgb.joblib`) reproduced the exact values reported in Phase 1:
- Model A: 1.5s AP = 1.0000, 1.0s AP = 1.0000, 0.5s AP = 1.0000, Mean AP = 1.0000, PR-AUC = 1.0000.
- Model B: 1.5s AP = 0.7500, 1.0s AP = 1.0000, 0.5s AP = 1.0000, Mean AP = 0.9167, PR-AUC = 0.9444.
- Model C (GRU): 1.5s AP = 1.0000, 1.0s AP = 1.0000, 0.5s AP = 1.0000, Mean AP = 1.0000, PR-AUC = 1.0000.
- Model C (XGBoost): 1.5s AP = 1.0000, 1.0s AP = 1.0000, 0.5s AP = 1.0000, Mean AP = 1.0000, PR-AUC = 1.0000.

---

## 3. Investigation into Perfect Scores & Dataset Shortcuts (Task 7)

Why did Model A (conventional detection only) achieve a perfect 1.00 Mean AP on the held-out test set?

### 3.1. Root Cause: Synthetic Benchmark Generator Shortcut
Examining `src/download_dataset.py` lines 215–350 revealed the root cause:

1. **Center Corridor Bias in Positive Videos:**
   In positive collision videos, the obstacle vehicle or pedestrian was programmed to converge directly toward the ego vehicle's center axis:
   ```python
   # download_dataset.py line 273 & 281
   curr_x = int(start_x + prog * (640 - start_x) * 0.25)
   curr_x = int(start_x * 0.75 + 0.25 * 640 + prog * (640 - (start_x * 0.75 + 0.25 * 640)))
   ```
   At a resolution of $1280 \times 720$, $x = 640$ corresponds to the exact image center ($c_x = 0.50$). Consequently, the feature `closest_dist_to_center_mean` ($|c_x - 0.5|$) in all positive test windows ranged from **0.001 to 0.133**.

2. **Lateral Offset in Negative Videos:**
   In negative non-collision driving videos, the lead cruising vehicle was placed off-center at $x = 520$ or $x = 760$ (`lead_x = np.random.choice([520, 760])`), and the distant sidewalk pedestrian was fixed at $x = 180$ ($c_x = 0.14$).
   In all negative test windows, the closest object detected by YOLO was the sidewalk pedestrian, yielding `closest_dist_to_center_mean` values of **0.358 to 0.364**.

3. **Bounding Box Area Discrepancy:**
   In negative videos, lead cruising vehicles maintained a constant safe following distance with maximum normalized bbox areas of **0.0218 to 0.0227**. In positive videos, even at the earliest 1.5s horizon, the colliding vehicle had already expanded to **0.0249 to 0.0885**.

### 3.2. Quantitative Shortcut Evidence
The test set is $100\%$ linearly separable using a single static feature:
$$\text{If } \texttt{closest\_dist\_to\_center\_mean} < 0.20 \implies \text{Label} = 1 \text{ (Collision)}$$
$$\text{If } \texttt{closest\_dist\_to\_center\_mean} \ge 0.20 \implies \text{Label} = 0 \text{ (Normal)}$$

Because this static detection feature separates all test samples with zero classification error, Model A can achieve 1.00 AP, 1.00 PR-AUC, and 1.00 ROC-AUC without ever observing inter-frame motion, displacement, or velocity vectors.

### 3.3. Why Model B Dropped to 0.75 AP at 1.5s
Model B had access to both detection features and tracking features. Why did its score drop to 0.75 AP at the 1.5s horizon?

In the held-out test split, there are three positive videos:
- `bench_pos_0002`: Vehicle scenario $\implies$ Model B $P = 0.9762$
- `bench_pos_0009`: Vehicle scenario $\implies$ Model B $P = 0.9762$
- `bench_pos_0012`: **Pedestrian near-miss scenario** ($12 \pmod 4 == 0$) $\implies$ Model B $P = 0.0238$ (False Negative)

In `bench_pos_0012_h1500ms`:
- The obstacle was a pedestrian rather than a vehicle.
- Pedestrians have a much smaller visual area ($A_{\text{norm}} = 0.0249$) compared to vehicles ($A_{\text{norm}} = 0.0885$).
- At $1.5\text{ seconds}$ before impact, the pedestrian's image-space velocity was relatively low ($v_y = 0.087$, speed = $0.108$).
- Because Model B trained an XGBoost classifier on aggregated tabular summary features (which weighted high vehicle approach speeds and large vehicle areas), the model classified this early pedestrian state as non-critical.
- **Model C (GRU)** correctly classified `bench_pos_0012_h1500ms` with probability **0.9998** because its recurrent hidden state tracked the continuous, accelerating growth rate across all 10 sequential time steps rather than evaluating a single summary threshold.

---

## 4. Feature Ablation Experiments (Task 3)

To rigorously determine which features actually contribute to anticipation capability, we conducted systematic ablation experiments across 6 incremental feature tiers on the exact same train/val/test splits:

1. **Tier 1 (Bounding-Box Features Only):** Coordinates ($c_x, c_y, y_{\text{bottom}}$), normalized areas ($A$), and static detection counts (17 features).
2. **Tier 2 (Bounding-Box + Displacement):** Tier 1 plus net trend displacements ($\Delta x_{\text{trend}}, \Delta y_{\text{trend}}$) and track duration (21 features).
3. **Tier 3 (Bounding-Box + Velocity):** Tier 2 plus instantaneous image-space velocities ($v_x, v_y$) and speed aggregates (33 features).
4. **Tier 4 (Bounding-Box + Velocity + Acceleration):** Tier 3 plus second-order accelerations ($a_x, a_y$) (41 features).
5. **Tier 5 (Complete Trajectory Features + XGBoost):** All 97 tabular features, including optical expansion rate ($\eta$), heading change rate, lateral cut-in ratio, abrupt deceleration proxy, TTC proxy, and camera ego-motion compensation.
6. **Tier 6 (Complete Trajectory Features + GRU):** Full temporal sequence matrix ($10 \times 14 = 140$ inputs) modeled with the 2-layer causal PyTorch GRU.

### Ablation Results Summary Table (`ablation_results.csv`)

| Ablation Tier | Features ($D$) | 1.5s AP | 1.0s AP | 0.5s AP | Mean AP | PR-AUC | ROC-AUC | F1-Score | FPR | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Bounding-box only** | 17 | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 0.00 | 0.14 |
| **2. Bbox + displacement** | 21 | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 0.00 | 0.11 |
| **3. Bbox + velocity** | 33 | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 0.00 | 0.17 |
| **4. Bbox + vel + acceleration** | 41 | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 0.00 | 0.11 |
| **5. Complete Trajectory + XGB** | 97 | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 0.00 | 0.11 |
| **6. Complete Trajectory + GRU** | 140 | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 0.00 | 0.33 |

### Critical Finding from Ablations
Because **Tier 1 (Bounding-box only)** achieves an identical 1.00 AP to **Tier 6 (Trajectory GRU)**, we must conclude:
> **Scientific Finding:** On this benchmark dataset, adding velocity, acceleration, expansion rate, and recurrent sequence modeling **does not produce an observable performance improvement over simple bounding-box features**. The performance ceiling is saturated by the dataset's static bounding-box shortcut.

---

## 5. Early Warning Capability & Lead Time Evaluation (Task 5)

We evaluated every held-out test video continuously across its entire 10-second duration using a rolling 2.0-second window buffer (sampled at 5 FPS) to determine the exact time of first warning ($P \ge 0.50$):

### Per-Video Early Warning Analysis Table (`per_video_predictions.csv`)

| Video ID | Scenario Type | Event Time ($t_e$) | Human Alert ($t_h$) | Model A Warn ($t_w$) | Lead Time A | Model B Warn ($t_w$) | Lead Time B | Model C GRU Warn ($t_w$) | Lead Time C | Warned Before Human? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `bench_pos_0002` | Lead Vehicle Collision | 9.28s | 7.32s | 3.0s | **6.28s** | 5.2s | 4.08s | **1.8s** | **7.48s** | **Yes** (All Models) |
| `bench_pos_0009` | Lead Vehicle Collision | 8.90s | 7.27s | 1.8s | **7.10s** | 1.8s | 7.10s | **1.8s** | **7.10s** | **Yes** (All Models) |
| `bench_pos_0012` | Pedestrian Near-Miss | 8.55s | 6.59s | 4.0s | **4.55s** | 5.0s | 3.55s | **3.6s** | **4.95s** | **Yes** (All Models) |
| `bench_neg_0005` | Normal Highway Cruise | N/A | N/A | None | N/A | None | N/A | None | N/A | No False Alarms ($P < 0.04$) |
| `bench_neg_0007` | Normal Highway Cruise | N/A | N/A | None | N/A | None | N/A | None | N/A | No False Alarms ($P < 0.04$) |
| `bench_neg_0011` | Normal Highway Cruise | N/A | N/A | None | N/A | None | N/A | None | N/A | No False Alarms ($P < 0.01$) |

### Lead Time Insights
1. **Warning Ahead of Human Reaction:** In all 3 positive test videos, Model C (GRU) crossed the warning threshold between $1.8\text{s}$ and $3.6\text{s}$ into the video, providing **4.95s to 7.48s of warning lead time** before physical impact. In all cases, the model alerted well before the human-annotated alert time (which occurred $1.6\text{s}$ to $2.0\text{s}$ prior to event).
2. **False Alarm Rate:** Across all 3 negative test videos (30 seconds of driving footage), zero false alarms occurred ($FPR = 0.00$, False Alarms/min = $0.00$).
3. **Caveat on Early Warnings:** In `bench_pos_0002` and `bench_pos_0009`, Model A and Model C issued warnings as early as $t = 1.8\text{s}$ to $3.0\text{s}$. This early trigger occurred because the colliding vehicle spawned already inside the center lane corridor. In real-world driving, an alert 7 seconds before a collision with a vehicle that is merely cruising ahead in your lane would constitute an unacceptable false positive.

---

## 6. Comprehensive Updated Metrics (Task 6)

We updated the evaluation reporting to clearly separate **Window-Level Evaluation** ($N = 18$ temporal windows) and **Video-Level Evaluation** ($N = 6$ independent video files), reporting 95% bootstrap confidence intervals ($B = 1,000$ resamples) and confusion matrix counts:

### Updated Performance Table (`updated_metrics.csv`)

| Evaluation Level | Model | 1.5s AP | 1.0s AP | 0.5s AP | Mean AP | PR-AUC (95% CI) | ROC-AUC (95% CI) | F1 | FPR | Latency | TP | FP | TN | FN |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Window-Level** ($N=18$) | **Model A (Detection Only)** | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 [1.00, 1.00] | 1.0000 [1.00, 1.00] | 1.0000 | 0.00 | **0.00 ms** | 9 | 0 | 9 | 0 |
| **Window-Level** ($N=18$) | **Model B (ByteTrack)** | 0.7500 | 1.0000 | 1.0000 | **0.9167** | 0.9444 [0.79, 1.00] | 0.9444 [0.81, 1.00] | 0.9412 | 0.00 | 0.28 ms | 8 | 0 | 9 | 1 |
| **Window-Level** ($N=18$) | **Model C (Trajectory GRU)** | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 [1.00, 1.00] | 1.0000 [1.00, 1.00] | 1.0000 | 0.00 | 0.97 ms | 9 | 0 | 9 | 0 |
| **Window-Level** ($N=18$) | **Model C (Trajectory XGB)** | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 [1.00, 1.00] | 1.0000 [1.00, 1.00] | 1.0000 | 0.00 | 0.33 ms | 9 | 0 | 9 | 0 |
| **Video-Level** ($N=6$) | **Model A (Detection Only)** | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 [1.00, 1.00] | 1.0000 [1.00, 1.00] | 1.0000 | 0.00 | **0.00 ms** | 3 | 0 | 3 | 0 |
| **Video-Level** ($N=6$) | **Model B (ByteTrack)** | 0.8333 | 1.0000 | 1.0000 | **0.9444** | 1.0000 [1.00, 1.00] | 1.0000 [1.00, 1.00] | 1.0000 | 0.00 | 0.28 ms | 3 | 0 | 3 | 0 |
| **Video-Level** ($N=6$) | **Model C (Trajectory GRU)** | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 [1.00, 1.00] | 1.0000 [1.00, 1.00] | 1.0000 | 0.00 | 0.97 ms | 3 | 0 | 3 | 0 |
| **Video-Level** ($N=6$) | **Model C (Trajectory XGB)** | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 [1.00, 1.00] | 1.0000 [1.00, 1.00] | 1.0000 | 0.00 | 0.33 ms | 3 | 0 | 3 | 0 |

---

## 7. Dataset Scale & Gating Limitations (Task 4)

### 7.1. Upstream Dataset Gating Status
Task 4 requested expanding the test dataset to at least 100 positive and 100 negative videos overall, or reporting the limitation honestly if dataset access prevents this.

We audited the environment for Hugging Face access to `nexar-ai/nexar_collision_prediction`:
- **Repository Type:** Gated Dataset under the **Nexar Open Data License**.
- **Authentication Check:** Calling `huggingface_hub.get_token()` returned `None`. Environment variables `HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN` are not set.
- **Access Status:** Unauthenticated API requests receive HTTP 401/403 (Gated Repository Access Denied).
- **Honest Assessment:** Downloading the official 100 positive and 100 negative real-world Nexar dashcam videos is **technically blocked by upstream license gating** until a valid user token is provided.

### 7.2. Why We Did Not Fabricate Synthetic Videos
While the benchmark script can synthetically synthesize additional videos (`--generate_benchmark`), doing so would merely replicate the same center-corridor bounding box shortcut. Rather than creating synthetic data to simulate 200 videos, we report the exact numbers on the audited 40 benchmark videos ($N=6$ held-out test videos) and provide the technical roadmap for real-world scaling.

---

## 8. Final Research Synthesis & Recommendations

### Does trajectory modeling genuinely outperform conventional baselines?
- **Theoretical Assessment:** **YES.** In real-world driving, a lead vehicle at a constant distance in stop-and-go traffic has a large bounding box but zero expansion rate ($\eta \approx 0$). In contrast, a vehicle undergoing emergency deceleration has a rapidly expanding bounding box ($\eta \gg 0$) and strong downward vertical acceleration ($a_y > 0$). Static detection cannot separate these two regimes without motion modeling.
- **Empirical Evidence on Benchmark:** **INCONCLUSIVE.** Because the benchmark dataset places normal cruising vehicles off-center and collision vehicles directly on-center, static detection models achieved 1.00 AP via dataset shortcuts. The trajectory GRU demonstrated superior temporal memory on the pedestrian test case (`bench_pos_0012`), but overall statistical superiority cannot be claimed on this benchmark.

### Mandatory Next Steps for Thesis / Production:
1. **Authenticate HF Credentials:** Provide a valid `HF_TOKEN` to download the official 1,500 Nexar dashcam videos, which contain diverse urban scenes, variable following distances, road curves, and genuine driving maneuvers.
2. **Implement Camera Ego-Motion Compensation:** Integrate sparse optical flow (e.g., Farneback or Lucas-Kanade) across the static background (sky/road borders) to subtract vehicle pitch dynamics during harsh braking.
3. **Incorporate Monocular Depth Models:** Replace image-space bounding box height with metric depth approximations from Depth Anything v2 or Metric3D to calculate physical metric Time-to-Collision ($\text{TTC} = \frac{d}{\dot{d}}$).
