# plot_all_results.py
# Unified plotter — generates all paper figures including V5
import os, json, numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

BASE = r"C:\MiniProject_Sem4\data_logs\qiskit_aer_simulation_2"
OUTDIR = os.path.join(BASE, "paper_figures")
os.makedirs(OUTDIR, exist_ok=True)


# ─── Load all results ────────────────────────────────────────────────────
def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


baseline = load_json(os.path.join(BASE, "baseline_summary.json"))
qft_v2 = load_json(os.path.join(BASE, "qft_summary.json"))
qft_v3 = load_json(os.path.join(BASE, "qft_v3_summary.json"))
qft_v4 = load_json(os.path.join(BASE, "qft_v4_summary.json"))
qft_v5 = load_json(os.path.join(BASE, "qft_v5_summary.json"))  # <-- ADDED V5

traj_v3 = load_json(os.path.join(BASE, "qft_v3_trajectories.json"))
traj_v4 = load_json(os.path.join(BASE, "qft_v4_trajectories.json"))
traj_v5 = load_json(os.path.join(BASE, "qft_v5_trajectories.json"))  # <-- ADDED V5

# Locked simulation results
SIM_RESULTS = {
    "V4 Seed 3\n(Classical)": {
        "mae": 1510,
        "mae_std": 367,
        "near": 0.948,
        "low": 0.004,
    },
    "PID\n(Gaussian)": {"mae": 3600, "mae_std": 5241, "near": 0.831, "low": 0.058},
    "V3 (RL)\n(Gaussian)": {"mae": 3333, "mae_std": 1932, "near": 0.645, "low": 0.134},
    "V3 (RL)\n(OU)": {"mae": 1241, "mae_std": 254, "near": 0.977, "low": 0.000},
    "V4 (RL+grad)\n(OU)": {"mae": 1284, "mae_std": 93, "near": 0.979, "low": 0.000},
}

# ─── Figure 1: Simulation three-way comparison bar chart ─────────────────
fig1, axes = plt.subplots(1, 3, figsize=(14, 5))
fig1.suptitle(
    "Simulation Results: PID vs V3 (RL) vs V4 (RL+Gradient)",
    fontsize=14,
    fontweight="bold",
)

labels = ["PID", "V3 (RL)", "V4 (RL+grad)"]
maes = [3600, 3333, 1510]
mae_std = [5241, 1932, 367]
nears = [83.1, 64.5, 94.8]
near_s = [18.5, 0, 7.6]
lows = [5.8, 13.4, 0.4]
colors = ["#E74C3C", "#3498DB", "#27AE60"]

axes[0].bar(labels, maes, yerr=mae_std, color=colors, capsize=6, alpha=0.85)
axes[0].set_ylabel("MAE (Hz)", fontsize=11)
axes[0].set_title("Frequency Tracking Error", fontsize=11)
axes[0].grid(axis="y", alpha=0.3)

axes[1].bar(labels, nears, yerr=near_s, color=colors, capsize=6, alpha=0.85)
axes[1].set_ylabel("Near-Resonance Fraction (%)", fontsize=11)
axes[1].set_title("Amp > 0.90 (% of Steps)", fontsize=11)
axes[1].set_ylim(0, 115)
axes[1].grid(axis="y", alpha=0.3)

axes[2].bar(labels, lows, color=colors, alpha=0.85)
axes[2].set_ylabel("Low-Amplitude Fraction (%)", fontsize=11)
axes[2].set_title("Amp < 0.70 (% of Steps)", fontsize=11)
axes[2].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTDIR, "fig1_sim_comparison.png"), dpi=200, bbox_inches="tight"
)
print("Figure 1 saved: fig1_sim_comparison.png")

# ─── Figure 2: Dual noise regime (Gaussian + OU) ─────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle(
    "Dual Noise Regime: Gaussian vs Ornstein-Uhlenbeck", fontsize=14, fontweight="bold"
)

for ax, regime, data, title in zip(
    axes2,
    ["Gaussian", "OU"],
    [
        {"PID": (3600, 5241), "V3": (3333, 1932), "V4": (1510, 367)},
        {"PID": (1414, 262), "V3": (1241, 254), "V4": (1284, 93)},
    ],
    ["Gaussian Drift (σ=500 Hz/step)", "OU Drift (mean-reverting)"],
):
    names = list(data.keys())
    means = [data[k][0] for k in names]
    stds = [data[k][1] for k in names]
    clrs = ["#E74C3C", "#3498DB", "#27AE60"]
    ax.bar(names, means, yerr=stds, color=clrs, capsize=6, alpha=0.85)
    ax.set_ylabel("MAE (Hz)", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.3)

axes2[0].text(
    2,
    1600,
    "p<0.001\n(significant)",
    ha="center",
    fontsize=9,
    color="#27AE60",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0FFF4"),
)
axes2[1].text(
    2,
    1400,
    "p=1.000\n(not significant)",
    ha="center",
    fontsize=9,
    color="#888888",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5"),
)

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "fig2_dual_noise.png"), dpi=200, bbox_inches="tight")
print("Figure 2 saved: fig2_dual_noise.png")

# ─── Figure 3: Qiskit QACT progression (Including V5) ────────────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))
fig3.suptitle(
    "Qiskit Aer QACT Progression: Zero-Shot → Fine-Tuned",
    fontsize=14,
    fontweight="bold",
)


def safe_val(d, key, default="N/A"):
    return d[key] if d else default


qk_labels = [
    "Zero-Shot",
    "QACT v2\n(wrong LR)",
    "QACT v3\n(V3 env)",
    "QACT v4\n(VN fix)",
    "QACT v5\n(Seed 3)",  # <-- ADDED V5 LABEL
]

# Extracting data for all 5 bars
qk_mae = [
    4370,
    4102,
    3292,
    safe_val(qft_v4, "mae_mean", None),
    safe_val(qft_v5, "mae_mean", None),
]
qk_std = [
    1119,
    1034,
    4176,
    safe_val(qft_v4, "mae_std", None),
    safe_val(qft_v5, "mae_std", None),
]
qk_near = [
    22.1,
    31.8,
    30.8,
    (safe_val(qft_v4, "near_mean", None) or 0) * 100,
    (safe_val(qft_v5, "near_mean", None) or 0) * 100,
]
qk_low = [
    5.7,
    5.5,
    18.3,
    (safe_val(qft_v4, "low_mean", None) or 0) * 100,
    (safe_val(qft_v5, "low_mean", None) or 0) * 100,
]

# Added purple color for V5
qk_colors = ["#C0392B", "#E67E22", "#F39C12", "#27AE60", "#8E44AD"]

# Filter out Nones in case V4 or V5 data is missing
valid = [i for i, m in enumerate(qk_mae) if m is not None]
v_lab = [qk_labels[i] for i in valid]
v_mae = [qk_mae[i] for i in valid]
v_std = [qk_std[i] for i in valid]
v_near = [qk_near[i] for i in valid]
v_low = [qk_low[i] for i in valid]
v_col = [qk_colors[i] for i in valid]

axes3[0].bar(v_lab, v_mae, yerr=v_std, color=v_col, capsize=5, alpha=0.85)
axes3[0].axhline(1510, color="#27AE60", ls="--", lw=1.5, label="Sim target (1510 Hz)")
axes3[0].set_ylabel("MAE (Hz)")
axes3[0].legend(fontsize=9)
axes3[0].set_title("MAE Progression")
axes3[0].grid(axis="y", alpha=0.3)

axes3[1].bar(v_lab, v_near, color=v_col, alpha=0.85)
axes3[1].axhline(94.8, color="#27AE60", ls="--", lw=1.5, label="Sim target (94.8%)")
axes3[1].set_ylabel("Near-Resonance (%)")
axes3[1].legend(fontsize=9)
axes3[1].set_title("Near-Resonance Progression")
axes3[1].grid(axis="y", alpha=0.3)

axes3[2].bar(v_lab, v_low, color=v_col, alpha=0.85)
axes3[2].axhline(0.4, color="#27AE60", ls="--", lw=1.5, label="Sim target (0.4%)")
axes3[2].set_ylabel("Low-Amplitude (%)")
axes3[2].legend(fontsize=9)
axes3[2].set_title("Low-Amplitude Progression")
axes3[2].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTDIR, "fig3_qiskit_progression.png"), dpi=200, bbox_inches="tight"
)
print("Figure 3 saved: fig3_qiskit_progression.png")

# ─── Figure 4: Trajectory comparison (V3 vs V4 vs V5) ────────────────────
if traj_v3 and traj_v4 and traj_v5:
    # Changed to 2 rows x 3 columns
    fig4, axes4 = plt.subplots(2, 3, figsize=(18, 9))
    fig4.suptitle(
        "Trajectory Comparison: QACT V3 vs V4 vs V5 (Episode 1)",
        fontsize=16,
        fontweight="bold",
    )

    trajectory_list = [
        (traj_v3[0], "QACT V3", "#3498DB"),
        (traj_v4[0], "QACT V4", "#27AE60"),
        (traj_v5[0], "QACT V5", "#8E44AD"),
    ]

    for col, (traj, lbl, clr) in enumerate(trajectory_list):
        steps = np.arange(len(traj["f_probe"]))

        # Row 1: Frequencies
        axes4[0, col].plot(
            steps,
            np.array(traj["f_qubit"]) / 1000,
            "r--",
            lw=1.5,
            alpha=0.8,
            label="Qubit (target)",
        )
        axes4[0, col].plot(
            steps,
            np.array(traj["f_probe"]) / 1000,
            color=clr,
            lw=1.5,
            label="Agent probe",
        )
        axes4[0, col].set_ylabel("Frequency (kHz)")
        axes4[0, col].set_title(f"{lbl}: Frequency Tracking")
        axes4[0, col].legend()
        axes4[0, col].grid(True, alpha=0.3)

        # Row 2: Amplitudes
        axes4[1, col].plot(steps, traj["amp"], color=clr, lw=1.5, label="P(|1>)")
        axes4[1, col].axhline(0.90, color="k", ls=":", label="0.90 threshold")
        axes4[1, col].set_ylabel("Excitation Prob.")
        axes4[1, col].set_xlabel("Step")
        axes4[1, col].set_ylim(0, 1.05)
        axes4[1, col].legend()
        axes4[1, col].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTDIR, "fig4_trajectory_comparison.png"),
        dpi=200,
        bbox_inches="tight",
    )
    print("Figure 4 saved: fig4_trajectory_comparison.png (Now includes V5)")
else:
    print("Figure 4 skipped or missing full trajectory data for V3, V4, and V5.")

print(f"\nAll figures saved to: {OUTDIR}")
print("Figures generated: fig1, fig2, fig3, fig4")
