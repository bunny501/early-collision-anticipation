"""
generate_visualizations.py
Generates publication-quality figures for the Stage 3.2 Report:
1. horizon_ap_comparison.png: AP across 1.5s, 1.0s, and 0.5s horizons for all 6 models.
2. bootstrap_ci_distribution.png: 95% Confidence Intervals comparing Model D (GRU) vs Model A (Detection) and Model F (Center).
3. continuous_risk_timeline.png: Continuous risk trajectories on representative positive (collision) and negative (normal driving) videos.
4. false_alarm_vs_lead_time.png: False Alarm Rate (episodes/min) vs Mean Lead Time (s) Pareto trade-off.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
VIS_DIR = BASE_DIR / "visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font="sans-serif")

def plot_horizon_comparison(df_horizon):
    plt.figure(figsize=(10, 6), dpi=300)
    palette = sns.color_palette("deep", n_colors=6)

    # Sort models consistently
    model_order = [
        "Model A (Detection Only)",
        "Model B (Detection + Tracking)",
        "Model C (Trajectory XGBoost)",
        "Model D (Trajectory GRU)",
        "Model E (Logistic Regression)",
        "Model F (Center-Distance)"
    ]

    ax = sns.barplot(
        data=df_horizon,
        x="Horizon (s)",
        y="AP",
        hue="Model",
        hue_order=model_order,
        palette="viridis"
    )

    plt.title("Primary Horizon Evaluation: Average Precision (AP) across Models", fontsize=14, weight="bold", pad=15)
    plt.xlabel("Anticipation Horizon Prior to Collision Event (seconds)", fontsize=12, labelpad=10)
    plt.ylabel("Average Precision (AP)", fontsize=12, labelpad=10)
    plt.ylim(0.0, 1.05)
    plt.axhline(0.50, color="gray", linestyle="--", linewidth=1.0, alpha=0.7, label="Chance Level (No Skill = 0.50)")

    for p in ax.patches:
        h = p.get_height()
        if h > 0.05:
            ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h / 2.0),
                        ha="center", va="center", fontsize=8, color="white", weight="bold", rotation=90)

    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0., title="Evaluated Models")
    plt.tight_layout()
    out_path = VIS_DIR / "horizon_ap_comparison.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

def plot_bootstrap_ci(df_ci):
    plt.figure(figsize=(11, 7), dpi=300)

    # Filter to main model comparisons
    sub_ci = df_ci[df_ci["Metric"].str.startswith("Delta AP")].copy()
    if len(sub_ci) == 0:
        sub_ci = df_ci.head(10).copy()

    y_pos = np.arange(len(sub_ci))
    means = sub_ci["Point Estimate"].values
    ci_low = sub_ci["95% CI Lower"].values
    ci_high = sub_ci["95% CI Upper"].values
    labels = [f"{r['Model / Comparison']}\n({r['Metric']})" for _, r in sub_ci.iterrows()]

    plt.errorbar(means, y_pos, xerr=[means - ci_low, ci_high - means], fmt='o', color="#1f77b4",
                 ecolor="#2ca02c", elinewidth=2.5, capsize=5, capthick=2, markersize=8)

    plt.axvline(0.0, color="red", linestyle="--", linewidth=1.5, alpha=0.8, label="Null Hypothesis (No Difference: Delta = 0.0)")

    plt.yticks(y_pos, labels, fontsize=10)
    plt.xlabel("AP Difference (95% Bootstrap Confidence Interval, N=2,000)", fontsize=12, labelpad=10)
    plt.title("Statistical Rigor: 95% Bootstrap Confidence Intervals of Comparative AP Gains", fontsize=14, weight="bold", pad=15)
    plt.legend(loc="lower right")
    plt.tight_layout()
    out_path = VIS_DIR / "bootstrap_ci_distribution.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

def plot_continuous_timelines(df_preds):
    plt.figure(figsize=(12, 8), dpi=300)

    # Pick 1 positive video with rich predictions and 1 negative video
    pos_vids = df_preds[df_preds["ground_truth"] == 1]["video_id"].unique()
    neg_vids = df_preds[df_preds["ground_truth"] == 0]["video_id"].unique()

    if len(pos_vids) == 0 or len(neg_vids) == 0:
        print("No predictions to plot.")
        return

    sample_pos = pos_vids[0]
    sample_neg = neg_vids[0]

    plt.subplot(2, 1, 1)
    df_p = df_preds[df_preds["video_id"] == sample_pos]
    for m in ["Model D (Trajectory GRU)", "Model A (Detection Only)", "Model F (Center-Distance)"]:
        sub = df_p[df_p["model_name"] == m].sort_values("timestamp")
        if len(sub) > 0:
            plt.plot(sub["timestamp"], sub["risk_score"], label=m, linewidth=2.0)

    # Event line
    t_ev = df_p["timestamp"].max() - 1.0
    plt.axvline(t_ev, color="red", linestyle=":", linewidth=2, label="Collision Event Timestamp (t_e)")
    plt.axhline(0.65, color="gray", linestyle="--", alpha=0.7, label="Nominal Alarm Threshold (0.65)")
    plt.title(f"Positive Video (ID: {sample_pos}) — Continuous Anticipation Risk Ramp-Up", fontsize=12, weight="bold")
    plt.xlabel("Video Timestamp (seconds)", fontsize=10)
    plt.ylabel("Model Collision Risk", fontsize=10)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc="upper left")

    plt.subplot(2, 1, 2)
    df_n = df_preds[df_preds["video_id"] == sample_neg]
    for m in ["Model D (Trajectory GRU)", "Model A (Detection Only)", "Model F (Center-Distance)"]:
        sub = df_n[df_n["model_name"] == m].sort_values("timestamp")
        if len(sub) > 0:
            plt.plot(sub["timestamp"], sub["risk_score"], label=m, linewidth=2.0)

    plt.axhline(0.65, color="gray", linestyle="--", alpha=0.7, label="Nominal Alarm Threshold (0.65)")
    plt.title(f"Negative Video (ID: {sample_neg}) — Continuous Safe Driving & False Suppression", fontsize=12, weight="bold")
    plt.xlabel("Video Timestamp (seconds)", fontsize=10)
    plt.ylabel("Model Collision Risk", fontsize=10)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc="upper left")

    plt.tight_layout()
    out_path = VIS_DIR / "continuous_risk_timeline.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

def plot_pareto_tradeoff(df_fa, df_lead):
    plt.figure(figsize=(9, 6), dpi=300)

    merged = pd.merge(df_fa, df_lead, on="Model Name")

    colors = sns.color_palette("tab10", len(merged))
    for i, row in merged.iterrows():
        plt.scatter(
            row["False Alarms / Minute"],
            row["Mean Lead Time (s)"],
            s=160,
            color=colors[i],
            label=row["Model Name"],
            edgecolor="black",
            linewidth=1.2,
            zorder=4
        )
        plt.annotate(
            row["Model Name"].split(" (")[0],
            (row["False Alarms / Minute"] + 0.01, row["Mean Lead Time (s)"] + 0.02),
            fontsize=9, weight="bold"
        )

    plt.title("Operational Trade-off: False Alarm Rate vs Early Warning Lead Time", fontsize=13, weight="bold", pad=15)
    plt.xlabel("False Alarm Rate (Episodes / Minute) [Lower is Better]", fontsize=11, labelpad=10)
    plt.ylabel("Mean Lead Time Prior to Event (Seconds) [Higher is Better]", fontsize=11, labelpad=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    out_path = VIS_DIR / "false_alarm_vs_lead_time.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

def main():
    print("="*60)
    print("GENERATING STAGE 3.2 VISUALIZATIONS")
    print("="*60)

    f_horizon = BASE_DIR / "horizon_metrics.csv"
    if f_horizon.exists():
        plot_horizon_comparison(pd.read_csv(f_horizon))

    f_ci = BASE_DIR / "bootstrap_confidence_intervals.csv"
    if f_ci.exists():
        plot_bootstrap_ci(pd.read_csv(f_ci))

    f_preds = BASE_DIR / "continuous_predictions.csv"
    if f_preds.exists():
        plot_continuous_timelines(pd.read_csv(f_preds))

    f_fa = BASE_DIR / "false_alarm_metrics.csv"
    f_lead = BASE_DIR / "lead_time_metrics.csv"
    if f_fa.exists() and f_lead.exists():
        plot_pareto_tradeoff(pd.read_csv(f_fa), pd.read_csv(f_lead))

    print("\n[SUCCESS] Visualizations created successfully!")

if __name__ == "__main__":
    main()
