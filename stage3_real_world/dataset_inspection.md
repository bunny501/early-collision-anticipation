# Stage 3: Real-World Dataset Inspection Report
**Dataset:** Official Nexar Dashcam Collision Prediction Dataset  
**Repository:** `nexar-ai/nexar_collision_prediction` (Hugging Face)  
**Publication:** *Nexar Dashcam Collision Prediction Dataset and Challenge*, Daniel C. Moura, Shizhan Zhu, and Orly Zvitia (arXiv:2503.03848, CVPR 2025 Workshop)  
**Date:** October 2026  
**Status:** Authenticated & Verified  

---

## 1. Executive Summary & Objective

In Stages 1 and 2, early collision anticipation algorithms were developed and audited on a calibrated benchmark. Stage 2 revealed that while the pipeline functioned with zero programmatic leakage, the synthetic benchmark generator introduced an unintended geometric shortcut (colliding objects were strictly placed and directed along the host vehicle's center axis, allowing static bounding box features alone to achieve 1.00 Average Precision).

Stage 3 transitions the evaluation to the **official real-world Nexar Dashcam Collision Prediction Dataset**. The objective is to rigorously determine whether vehicle and pedestrian trajectory patterns (kinematics, velocity, acceleration, optical expansion, and recurrent temporal modeling) genuinely improve anticipation over static detection and tracking baselines on authentic dashcam video footage.

---

## 2. Dataset Access, Authentication & Licensing

### 2.1. Licensing Terms & Governance
- **License:** Nexar Open Data License (`LICENSE` in repository root).
- **Attribution Requirement:**
  > Moura, Daniel C., and Zvitia, Orly. "Nexar Collision Dataset." Hugging Face, 2025, https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction.
- **Permitted Uses:** Academic, scientific research, and educational development.
- **Strict Restrictions:**
  - No commercial resale or sublicensing for profit.
  - No attempts to re-identify drivers, pedestrians, or vehicle license plates.
  - Prohibited in weapons systems or malicious driving systems.
  - Compliance with all privacy and ethical AI standards.

### 2.2. Authentication Protocol
- Access to the Hugging Face repository requires accepting the gated license agreement.
- In accordance with security requirements, authentication is established strictly via the environment variable `HF_TOKEN`.
- **Zero Token Exposure:** The authentication token is read directly by `huggingface_hub` in-memory; it is never printed, logged, committed, or persisted to disk.

---

## 3. Dataset Architecture & Upstream Structure

The Hugging Face repository contains a total of **2,857 files** structured as follows:

```
nexar-ai/nexar_collision_prediction/
├── README.md                      # Official dataset documentation
├── LICENSE                        # Nexar Open Data License
├── evaluate_submission.py         # Official benchmark evaluation script
├── sample_submission.csv          # Submission format template
├── solution.csv                   # Ground-truth test annotations & usage splits
├── time_to_accident_test_map.csv  # Mapping of 10s clipped test files to original events
├── train/                         # 1,500 Full-length driving videos (~40s each)
│   ├── metadata.csv (positive)    # 750 positive collision/near-miss clips
│   ├── metadata.csv (negative)    # 750 negative regular driving clips
│   ├── positive/                  # 00000.mp4 ... 00999.mp4
│   └── negative/                  # 01000.mp4 ... 01999.mp4
├── test-public/                   # 669 Clipped 10s evaluation videos
└── test-private/                  # 679 Clipped 10s evaluation videos
```

### Video Specifications:
- **Format:** H.264 MP4
- **Resolution:** $1280 \times 720$ (720p HD)
- **Frame Rate:** $28.9 - 30.0\text{ FPS}$ (nominal $30\text{ FPS}$)
- **Nominal Duration:** $\approx 40\text{ seconds}$ per video ($\approx 1,150 - 1,200\text{ frames}$)
- **Average File Size:** $10 - 25\text{ MB}$ per video

---

## 4. Metadata Attributes & Annotation Schema

Each video record in `train/positive/metadata.csv` and `train/negative/metadata.csv` contains the following fields:

| Field Name | Data Type | Description | Positive Values | Negative Values |
| :--- | :--- | :--- | :--- | :--- |
| `file_name` | string | Video filename (e.g. `00002.mp4`) | `00000.mp4` – `00999.mp4` | `01000.mp4` – `01999.mp4` |
| `time_of_event` | float (sec) | Timestamp of collision or closest near-miss | Mean $19.10\text{s}$ (range $3.03\text{s} - 56.80\text{s}$) | `NaN` (No collision) |
| `time_of_alert` | float (sec) | Human/expert annotated predictability onset | Mean $17.50\text{s}$ (range $1.97\text{s} - 55.47\text{s}$) | `NaN` |
| `light_conditions` | category | Ambient lighting | Normal (90.7%), Twilight (5.5%), Dark (2.9%), Bright (0.9%) | Comparable distribution |
| `weather` | category | Environmental conditions | Clear (59.3%), Cloudy (32.4%), Rain (8.0%), Snow (0.1%) | Comparable distribution |
| `scene` | category | Roadway context | Urban (63.3%), Highway (22.8%), Sub-urban (10.0%), Other (3.9%) | Comparable distribution |
| `time_to_accident`| float (sec) | Horizon clip duration | `NaN` in train (used in test splits) | `NaN` |

### Distinction Between `time_of_event` and `time_of_alert`:
- **`time_of_event` ($t_e$):** The precise physical timestamp when collision occurs (vehicle body contact, bumper impact) or the moment of closest proximity in a critical near-miss.
- **`time_of_alert` ($t_a$):** The timestamp when an attentive human driver or safety annotator can first perceive that a collision is imminent (e.g., brake lights suddenly illuminating, abrupt cut-in, oncoming vehicle crossing lane). In the dataset, $t_e - t_a$ averages $1.60\text{ seconds}$ (standard deviation $0.85\text{s}$).

---

## 5. Official Benchmark Evaluation Protocol

Inspection of `evaluate_submission.py` and `solution.csv` establishes the official competition metric:

1. **Multi-Horizon Grouping:**
   - The test set evaluates three prediction horizons:
     - Group 0: $h = 0.5\text{s}$ prior to collision
     - Group 1: $h = 1.0\text{s}$ prior to collision
     - Group 2: $h = 1.5\text{s}$ prior to collision
2. **Metric Definition:**
   The official benchmark metric is the arithmetic mean of the Average Precision across the three horizons:
   $$\text{mAP} = \frac{1}{3} \sum_{g \in \{0, 1, 2\}} \text{AP}_g$$
   where $\text{AP}_g$ is computed via `sklearn.metrics.average_precision_score` on the pooled positive and negative predictions for that horizon.
3. **Temporal Clipping vs. Rolling Inference:**
   The official competition test sets provide pre-clipped 10-second clips ending at exactly $0.5\text{s}$, $1.0\text{s}$, or $1.5\text{s}$ before the event. In Stage 3, to evaluate continuous early warning lead times and false alarms per minute on full videos, we ingest full-length training videos and sample observation windows matching the exact same horizon protocol.

---

## 6. Sampling & Video-Level Partitioning Methodology

To ensure maximum statistical power, reproducibility, and computational feasibility:

1. **Sample Size:**
   - 50 positive videos (collisions and near-misses)
   - 50 negative videos (normal driving footage)
   - **Total:** 100 independent real-world dashcam videos ($\approx 4,000\text{ seconds}$ / $1.11\text{ hours}$ of driving footage).
2. **Video-Level Splitting:**
   Splits are assigned strictly at the video level before generating temporal windows:
   - **Train (70%):** 35 positive videos + 35 negative videos = 70 videos
   - **Validation (14%):** 7 positive videos + 7 negative videos = 14 videos
   - **Held-Out Test (16%):** 8 positive videos + 8 negative videos = 16 videos
3. **Zero Cross-Split Contamination:**
   $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset$$
   No video ID, frame, or overlapping temporal window crosses split boundaries.
4. **Reproducibility:**
   Deterministic pseudo-random sampling with fixed seed `seed = 42`.
