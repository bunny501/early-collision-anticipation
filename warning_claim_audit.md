# Stage 3.1 Audit: Early Warning Claims & False Alarm Rate Analysis

**Audit Phase:** Stage 3.1 Methodological Verification  
**Target Deliverables:** Warning Timelines (`warning_analysis.csv`), Per-Video Scores (`per_video_predictions.csv`, `video_level_scores.csv`), and Alarm Rate Calculations  
**Date:** October 2026  
**Auditor:** Independent Scientific Audit  

---

## 1. Audit of Early Warning Claims

In the Stage 3 report (`stage3_report.md`), the following claim was highlighted:
> *"In 3 of the 8 test collision scenarios (video 00103, 00519, and 00059), the models triggered an alert before the human-annotated alert timestamp ($t_{\text{warn}} < t_{\text{alert}}$), providing critical additional reaction time for automated braking (+0.43s to +1.17s faster)."*

### 1.1. Detailed Finding & Verdict: **CLAIM IS METHODOLOGICALLY UNSUPPORTED**
This claim must be **retracted and corrected** in all scientific reporting. The audit identified five critical flaws in this claim:

1. **Nexar `time_of_alert` is NOT Human Driver Reaction:**
   - In the official Nexar benchmark (*Moura et al., CVPR 2025 Workshop / arXiv:2503.03848*), `time_of_alert` does not represent driver eye gaze, brake pedal actuation, or physical driver perception.
   - It is an **offline retrospective annotation** created by third-party human annotators watching the video replay to mark the earliest moment when an impending crash appeared foreseeable.
   - Labeling this as "human reaction" or claiming the model "beat human driver reaction" is a false equivalence.

2. **Coarse Window Discretization vs Continuous Triggering:**
   - The warning evaluation did **not** perform continuous rolling inference across the entire video.
   - Instead, the models were evaluated strictly on three static 2.0-second observation windows ending at $t_e - 1.5\text{s}$, $t_e - 1.0\text{s}$, and $t_e - 0.5\text{s}$.
   - The reported warning time $t_w$ was constrained to exactly three discrete values: $t_e - 1.5\text{s}$, $t_e - 1.0\text{s}$, or $t_e - 0.5\text{s}$.
   - If an annotator marked $t_{\text{alert}} = t_e - 0.33\text{s}$ (as in video `00059`), any model evaluated at the 1.5s window with $P \ge 0.5$ automatically scored $1.50\text{s} > 0.33\text{s}$, creating an apparent "+1.17s faster" lead time solely due to the testing grid resolution.

3. **Intermittent Predictions and Severe Signal Flickering:**
   Inspection of window-by-window probabilities across the test positive videos reveals severe instability:
   - **Video `00640` (Model C GRU):**
     - $1.5\text{s}$ Horizon: $P = 0.9260$ (Alarm Triggered)
     - $1.0\text{s}$ Horizon: $P = 0.0133$ (Alarm Collapsed to Zero!)
     - $0.5\text{s}$ Horizon: $P = 0.9987$ (Alarm Re-triggered)
     *Verdict:* A driver assistance system cannot issue an emergency warning, drop it completely 500 ms later, and re-issue it 500 ms before impact.
   - **Video `00103` (Model C GRU):**
     - $1.5\text{s}$ Horizon: $P = 0.9015$ (Alarm)
     - $1.0\text{s}$ Horizon: $P = 0.9086$ (Alarm)
     - $0.5\text{s}$ Horizon: $P = 0.2755$ (Dropped below threshold right before impact!)
   - **Video `00937` (Model C GRU):**
     - $1.5\text{s}$ Horizon: $P = 0.0170$ (No Alarm)
     - $1.0\text{s}$ Horizon: $P = 0.0080$ (No Alarm)
     - $0.5\text{s}$ Horizon: $P = 0.9139$ (Late Alarm at 0.5s only)
   *Conclusion:* The GRU model does not exhibit sustained, monotonically increasing alarm confidence as collision approaches. Without a temporal hysteresis filter, it suffers from signal flickering.

4. **Reference Baseline for Lead Time:**
   - Lead times were measured relative to physical impact ($t_{\text{event}}$), **not** relative to `time_of_alert`.
   - The reported lead times of 1.50s, 1.00s, and 0.50s are simply the predefined horizon offsets.

5. **Arbitrary Fixed Threshold:**
   - The threshold was fixed at $\tau = 0.50$ without calibration on the validation split. No precision-recall trade-off optimization was performed.

---

## 2. Audit of False Alarms Per Minute

In `stage3_metrics.csv`, the reported false alarm rates were:
- Window-Level: Model A = 8.75/min, Model B = 12.50/min, Model C GRU = 7.50/min, Tier 0 Center = 5.00/min.
- Video-Level: Model A = 0.19/min, Model B = 0.38/min, Model C GRU = 0.38/min, Tier 0 Center = 0.19/min.

### 2.1. Mathematical Audit of the Denominator & Method
The audit inspected the exact code in `train_and_evaluate.py`:

```python
# Window level (lines 224-225):
test_neg_count = sum((df_a.loc[test_mask, "label"] == 0)) # 24 windows
neg_duration_min = (test_neg_count * 2.0) / 60.0          # 0.80 minutes

# Video level (line 444):
neg_vid_duration_min = (sum(y_vid_test == 0) * 40.0) / 60.0 # 5.33 minutes
```

### 2.2. Critical Flaws in False Alarm Rate Estimation:
1. **Window-Level Denominator Artificially Deflates Duration:**
   - The denominator is $24 \text{ windows} \times 2.0\text{ seconds} = 48\text{ seconds} = 0.8\text{ minutes}$.
   - Because the denominator is less than 1 minute ($0.8\text{ min}$), dividing the 6 false positive windows of Model C GRU yields $6 / 0.8 = 7.50\text{ alarms/min}$.
   - Each window is evaluated independently, treating consecutive false positive windows as separate alarms.
2. **Video-Level Denominator Assumes Zero False Alarms Outside Windows:**
   - For video level, the denominator was set to $8 \times 40.0\text{s} = 320\text{s} = 5.33\text{ minutes}$.
   - Model A had 1 false positive video (Video `01542`), giving $1 / 5.33 = 0.1875 \approx 0.19\text{ alarms/min}$.
   - Model C GRU had 2 false positive videos (Videos `01853` and `01542`), giving $2 / 5.33 = 0.375 \approx 0.38\text{ alarms/min}$.
   - **Crucial Limitation:** Only 6 seconds out of every 40-second negative video were actually observed (the windows at 16s, 18s, and 20s). The remaining 34 seconds of driving were never fed through the model!
   - Assuming zero alarms occur in the unobserved 85% of the video artificially underestimates the true continuous false alarm rate.
3. **Absence of Debounce, Cooldown, and Episode Logic:**
   - In production automotive ADAS (ISO 26262 / Euro NCAP), false alarms must be measured as **alarm episodes** with a minimum debounce duration (e.g. 500 ms sustained confidence) and a lockout cooldown (e.g. 2.0s) to prevent a single event from generating multiple alerts.
   - No debounce logic was implemented in Stage 3.

---

## 3. Explicit False-Alarm Comparison Table

| Evaluation Protocol | Model Architecture | Evaluated Samples | False Positives ($FP$) | Evaluated Duration | Computed Alarms / Min | Methodological Validity |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Window-Level (Reported)** | Tier 0: Center Only | 24 windows | 4 | 0.80 min (48s) | **5.00** | Biased: windows are temporally clustered. |
| **Window-Level (Reported)** | Model A: YOLO Detection | 24 windows | 7 | 0.80 min (48s) | **8.75** | Biased: treats overlapping windows independently. |
| **Window-Level (Reported)** | Model B: ByteTrack | 24 windows | 10 | 0.80 min (48s) | **12.50** | Biased: treats overlapping windows independently. |
| **Window-Level (Reported)** | Model C: Trajectory GRU | 24 windows | 6 | 0.80 min (48s) | **7.50** | Biased: treats overlapping windows independently. |
| **Video-Level (Reported)** | Tier 0: Center Only | 8 videos | 1 | 5.33 min (320s) | **0.19** | Underestimated: unobserved 34s assumed clean. |
| **Video-Level (Reported)** | Model A: YOLO Detection | 8 videos | 1 | 5.33 min (320s) | **0.19** | Underestimated: unobserved 34s assumed clean. |
| **Video-Level (Reported)** | Model B: ByteTrack | 8 videos | 2 | 5.33 min (320s) | **0.38** | Underestimated: unobserved 34s assumed clean. |
| **Video-Level (Reported)** | Model C: Trajectory GRU | 8 videos | 2 | 5.33 min (320s) | **0.38** | Underestimated: unobserved 34s assumed clean. |
| **Video Observed Active Time** | Model C: Trajectory GRU | 8 videos (active) | 2 | 0.80 min (48s) | **2.50** | Valid bounded estimate on observed footage. |

---

## 4. Corrected Recommendations for Stage 4

1. **Retract Human Perception Advantage:** Replace all statements asserting the model "beat human drivers" with exact descriptions of horizon evaluation relative to the annotator timestamp.
2. **Implement Rolling Continuous Inference:** Run continuous frame-by-frame inference across all 40 seconds of each test video ($1,200\text{ frames}$ per video) rather than 3 sampled windows.
3. **Incorporate Hysteresis Debouncing:** Require an alert to maintain $P \ge \tau_{\text{high}}$ for $\ge 3$ consecutive frames (100 ms) before triggering, and drop below $\tau_{\text{low}}$ before resetting, counting discrete alarm episodes.
