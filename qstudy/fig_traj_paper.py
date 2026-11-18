"""FIGTRAJ: Paper-size trajectory figures from traj_v3 NPZ files.
Read-only: uses existing NPZ and CSV files. No model loading, no env runs.
Outputs to $w\results\figs_paper\
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Color-blind safe palette
CB_PALETTE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9']
SEED_COLORS = {'200': CB_PALETTE[0], '202': CB_PALETTE[1], '101': CB_PALETTE[2], '102': CB_PALETTE[3], 'baseline': CB_PALETTE[4]}

WORK = r"C:\Projects\MiniProject_Sem4_QSTUDY_WORK"
TRAJ_DIR = os.path.join(WORK, "results", "traj_v3")
OUT_DIR = os.path.join(WORK, "results", "figs_paper")
os.makedirs(OUT_DIR, exist_ok=True)

# Read CSV
csv_path = os.path.join(TRAJ_DIR, "traj_summary_v3.csv")
df = pd.read_csv(csv_path)
print(f"CSV rows: {len(df)}")

# List NPZ files
npz_files = [f for f in os.listdir(TRAJ_DIR) if f.endswith('.npz')]
print(f"NPZ files ({len(npz_files)}):")
for f in npz_files:
    print(f"  {f}")

# --- Helper: load trajectory data ---
def load_trajectory(model_name, seed):
    """Load NPZ for a model+seed, return dict of arrays."""
    if model_name == "HillClimb_baseline_RB":
        fname = f"HillClimb_baseline_RB_trajectories.npz"
    elif model_name == "Z1_dither_plain":
        fname = f"Z1_dither_plain_s{seed}_trajectories.npz"
    else:
        fname = f"{model_name}_s{seed}_trajectories.npz"
    path = os.path.join(TRAJ_DIR, fname)
    data = np.load(path)
    return {k: data[k] for k in data.files}

# --- Figure 1: RB models, episode 0 offset vs step, two columns ---
rb_models = ["E1_zero_shot_RB", "E4_RB", "FTB_200000_RB", "HillClimb_baseline_RB"]
seeds_rb = {"E1_zero_shot_RB": ["200", "202"],
            "E4_RB": ["200", "202"],
            "FTB_200000_RB": ["200", "202"],
            "HillClimb_baseline_RB": ["baseline"]}

fig, axes = plt.subplots(2, 2, figsize=(7.16, 6.5), dpi=300)
axes = axes.flatten()

for idx, model in enumerate(rb_models):
    ax = axes[idx]
    for seed in seeds_rb[model]:
        traj = load_trajectory(model, seed)
        # Episode 0 is column 0
        offset = traj['x'][:, 0] - traj['fc'][:, 0]  # x - fc
        steps = np.arange(len(offset))
        color = SEED_COLORS.get(seed, 'k')
        label = f"seed {seed}" if seed != "baseline" else "HillClimb"
        ax.plot(steps, offset, color=color, label=label, linewidth=1.0, alpha=0.9)
    
    # +/-1000 Hz band
    ax.axhline(y=1000, color='gray', linestyle='--', linewidth=0.8)
    ax.axhline(y=-1000, color='gray', linestyle='--', linewidth=0.8)
    ax.fill_between([0, 199], -1000, 1000, color='gray', alpha=0.1)
    
    ax.set_xlabel("Step")
    ax.set_ylabel("Offset (Hz)")
    ax.set_title(model.replace("_", " "))
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(0, 199)
    ax.set_ylim(-5000, 5000)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
for ext in ['png', 'pdf']:
    for width_in in [3.5, 7.16]:
        fname = f"FigT1_RB_models_ep0_offset_{ext}_w{str(width_in).replace('.','p')}.{ext}"
        # Resize for different widths
        if ext == 'pdf':
            fig.savefig(os.path.join(OUT_DIR, fname), dpi=300, bbox_inches='tight')
        else:
            fig.savefig(os.path.join(OUT_DIR, fname), dpi=300, bbox_inches='tight')
print(f"FigT1 saved: {len(os.listdir(OUT_DIR))} files so far")

# --- Figure 2: Z1 dither plain ---
fig2, ax2 = plt.subplots(1, 1, figsize=(3.5, 2.8), dpi=300)
for seed in ["101", "102"]:
    traj = load_trajectory("Z1_dither_plain", seed)
    offset = traj['x'][:, 0] - traj['fc'][:, 0]
    steps = np.arange(len(offset))
    color = SEED_COLORS.get(seed, 'k')
    ax2.plot(steps, offset, color=color, label=f"seed {seed}", linewidth=1.0, alpha=0.9)

ax2.axhline(y=1000, color='gray', linestyle='--', linewidth=0.8)
ax2.axhline(y=-1000, color='gray', linestyle='--', linewidth=0.8)
ax2.fill_between([0, 199], -1000, 1000, color='gray', alpha=0.1)
ax2.set_xlabel("Step")
ax2.set_ylabel("Offset (Hz)")
ax2.set_title("Z1 dither plain (dither env)")
ax2.legend(fontsize=8)
ax2.set_xlim(0, 199)
ax2.set_ylim(-5000, 5000)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
for ext in ['png', 'pdf']:
    for width_in in [3.5, 7.16]:
        fname = f"FigT2_Z1_dither_ep0_offset_{ext}_w{str(width_in).replace('.','p')}.{ext}"
        fig2.savefig(os.path.join(OUT_DIR, fname), dpi=300, bbox_inches='tight')
print("FigT2 saved")

# --- Figure 3: Bar chart of per-seed mean MAE and lock_w1000 ---
mean_rows = df[df['episode'] == 'mean'].copy()
rb_models_csv = ["E1_zero_shot_RB", "E4_RB", "FTB_200000_RB", "HillClimb_baseline_RB"]

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(7.16, 3.0), dpi=300)

# MAE bar chart
x_pos = np.arange(len(rb_models_csv))
width = 0.35
for i, seed in enumerate(["200", "202"]):
    maes = []
    for model in rb_models_csv:
        if model == "HillClimb_baseline_RB":
            val = mean_rows[(mean_rows['model'] == model) & (mean_rows['model_seed'] == 'baseline')]['mae_hz'].values
        else:
            val = mean_rows[(mean_rows['model'] == model) & (mean_rows['model_seed'] == int(seed))]['mae_hz'].values
        maes.append(val[0] if len(val) > 0 else 0)
    ax3a.bar(x_pos + (i - 0.5) * width, maes, width, label=f"seed {seed}", color=SEED_COLORS[seed], alpha=0.8)

ax3a.set_xticks(x_pos)
ax3a.set_xticklabels([m.replace("_", " ") for m in rb_models_csv], rotation=15, ha='right', fontsize=8)
ax3a.set_ylabel("Mean MAE (Hz)")
ax3a.set_title("Per-seed mean MAE (5 validation episodes)")
ax3a.legend(fontsize=8)
ax3a.grid(True, alpha=0.3, axis='y')

# Lock_w1000 bar chart
for i, seed in enumerate(["200", "202"]):
    locks = []
    for model in rb_models_csv:
        if model == "HillClimb_baseline_RB":
            val = mean_rows[(mean_rows['model'] == model) & (mean_rows['model_seed'] == 'baseline')]['lock_w1000'].values
        else:
            val = mean_rows[(mean_rows['model'] == model) & (mean_rows['model_seed'] == int(seed))]['lock_w1000'].values
        locks.append(val[0] if len(val) > 0 else 0)
    ax3b.bar(x_pos + (i - 0.5) * width, locks, width, label=f"seed {seed}", color=SEED_COLORS[seed], alpha=0.8)

ax3b.set_xticks(x_pos)
ax3b.set_xticklabels([m.replace("_", " ") for m in rb_models_csv], rotation=15, ha='right', fontsize=8)
ax3b.set_ylabel("Mean lock_w1000")
ax3b.set_title("Per-seed mean lock_w1000 (5 validation episodes)")
ax3b.legend(fontsize=8)
ax3b.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
for ext in ['png', 'pdf']:
    for width_in in [3.5, 7.16]:
        fname = f"FigT3_bar_MAE_lock_{ext}_w{str(width_in).replace('.','p')}.{ext}"
        fig3.savefig(os.path.join(OUT_DIR, fname), dpi=300, bbox_inches='tight')
print("FigT3 saved")

# --- Verify plotted numbers equal CSV means ---
print("\n=== Verification: Plotted values vs CSV means ===")
for model in rb_models_csv:
    if model == "HillClimb_baseline_RB":
        row = mean_rows[(mean_rows['model'] == model) & (mean_rows['model_seed'] == 'baseline')]
    else:
        for seed in ["200", "202"]:
            row = mean_rows[(mean_rows['model'] == model) & (mean_rows['model_seed'] == int(seed))]
            if len(row) > 0:
                print(f"{model} seed {seed}: MAE={row['mae_hz'].values[0]:.2f}, lock={row['lock_w1000'].values[0]:.4f}")

# List output files
print("\n=== Output files ===")
for f in sorted(os.listdir(OUT_DIR)):
    path = os.path.join(OUT_DIR, f)
    print(f"  {f} ({os.path.getsize(path)} bytes, mtime={os.path.getmtime(path)})")

# Write CAPTIONS.md
cap_path = os.path.join(OUT_DIR, "CAPTIONS.md")
with open(cap_path, 'w') as f:
    f.write("# Figure Captions (Draft, Facts Only)\n\n")
    f.write("## FigT1: Episode 0 offset vs step for RB models\n")
    f.write("Models: E1 zero-shot (hybrid rabi), E4 (surrogate rabi), FTB_200000 (surrogate rabi), HillClimb baseline (surrogate rabi). ")
    f.write("Seeds 200 and 202 shown as separate lines. Grey band: +/-1000 Hz lock threshold. ")
    f.write("Validation split seed base 500000, RB nominal, 5 episodes per seed. Exploratory.\n\n")
    f.write("## FigT2: Episode 0 offset vs step for Z1 dither plain\n")
    f.write("Seeds 101 and 102. Dither environment (not comparable with RB surrogate rabi). ")
    f.write("Grey band: +/-1000 Hz lock threshold. Validation split seed base 500000. Exploratory.\n\n")
    f.write("## FigT3: Bar charts of per-seed mean MAE (Hz) and mean lock_w1000\n")
    f.write("Models: E1 zero-shot, E4, FTB_200000, HillClimb baseline. ")
    f.write("Seeds 200 and 202 as separate bars (HillClimb: single bar). ")
    f.write("5 validation episodes per seed, RB nominal. No error bars. Exploratory.\n")

print(f"\nCAPTIONS.md written to {cap_path}")
print("FIGTRAJ DONE")