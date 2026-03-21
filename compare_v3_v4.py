import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Gaussian authoritative numbers ────────────────────────────────────────
v3g_mae         = 3333
v3g_mae_std     = 1932
v3g_reward      = 174.3
v3g_reward_std  = 20.2
v3g_near        = 64.5
v3g_low         = 13.4

v4g_mae_per_seed          = [2288, 1731, 2608, 1510, 2887]
v4g_reward_per_seed       = [184.5, 190.2, 181.4, 192.5, 178.9]
v4g_near_per_seed         = [85.0, 91.2, 78.9, 94.8, 74.3]
v4g_low_per_seed          = [5.8, 1.3, 8.2, 0.4, 9.8]

v4g_mae_mean  = np.mean(v4g_mae_per_seed);  v4g_mae_std  = np.std(v4g_mae_per_seed)
v4g_rew_mean  = np.mean(v4g_reward_per_seed); v4g_rew_std  = np.std(v4g_reward_per_seed)
v4g_near_mean = np.mean(v4g_near_per_seed); v4g_near_std = np.std(v4g_near_per_seed)
v4g_low_mean  = np.mean(v4g_low_per_seed);  v4g_low_std  = np.std(v4g_low_per_seed)

# ── OU authoritative numbers ───────────────────────────────────────────────
v3ou_mae        = 1241
v3ou_mae_std    = 254
v3ou_reward     = 194.7
v3ou_reward_std = 2.0
v3ou_near       = 97.7
v3ou_low        = 0.0

v4ou_mae_per_seed    = [1190, 1411, 1250]
v4ou_reward_per_seed = [194.9, 192.8, 195.3]
v4ou_near_per_seed   = [98.2, 98.1, 97.3]
v4ou_low_per_seed    = [0.0, 0.0, 0.0]

v4ou_mae_mean  = np.mean(v4ou_mae_per_seed);  v4ou_mae_std  = np.std(v4ou_mae_per_seed)
v4ou_rew_mean  = np.mean(v4ou_reward_per_seed); v4ou_rew_std  = np.std(v4ou_reward_per_seed)
v4ou_near_mean = np.mean(v4ou_near_per_seed); v4ou_near_std = np.std(v4ou_near_per_seed)
v4ou_low_mean  = np.mean(v4ou_low_per_seed);  v4ou_low_std  = np.std(v4ou_low_per_seed)

if __name__ == "__main__":
    print("=== Gaussian ===")
    print(f"MAE:        V3={v3g_mae}±{v3g_mae_std}Hz  V4={v4g_mae_mean:.0f}±{v4g_mae_std:.0f}Hz")
    print(f"Reward:     V3={v3g_reward}±{v3g_reward_std}  V4={v4g_rew_mean:.1f}±{v4g_rew_std:.1f}")
    print(f"Amp>0.90:   V3={v3g_near}%  V4={v4g_near_mean:.1f}%±{v4g_near_std:.1f}%")
    print(f"Amp<0.70:   V3={v3g_low}%  V4={v4g_low_mean:.1f}%±{v4g_low_std:.1f}%")
    print("=== OU ===")
    print(f"MAE:        V3={v3ou_mae}±{v3ou_mae_std}Hz  V4={v4ou_mae_mean:.0f}±{v4ou_mae_std:.0f}Hz")
    print(f"Reward:     V3={v3ou_reward}±{v3ou_reward_std}  V4={v4ou_rew_mean:.1f}±{v4ou_rew_std:.1f}")
    print(f"Amp>0.90:   V3={v3ou_near}%  V4={v4ou_near_mean:.1f}%±{v4ou_near_std:.1f}%")
    print(f"Amp<0.70:   V3={v3ou_low}%  V4={v4ou_low_mean:.1f}%±{v4ou_low_std:.1f}%")

    BAR_WIDTH = 0.35
    V3_COLOR  = '#d9534f'
    V4_COLOR  = '#5b9bd5'
    ERR_KW    = dict(ecolor='#333333', capsize=6, capthick=1.5, elinewidth=1.5)
    ANNOT_KW  = dict(ha='center', va='bottom', fontsize=9,
                     color='#1a7a1a', fontweight='bold')
    X         = np.array([0, 1])
    XLABELS   = ['V3', 'V4']

    fig, axs = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(
        "V3 vs V4: Gradient-Augmented Observation — Gaussian (top) and OU Noise (bottom)",
        fontsize=13, fontweight='bold'
    )

    def bar_pair(ax, v3_val, v3_std, v4_mean, v4_std,
                 title, ylabel, ylim, better_offset, lower_is_better=False,
                 v3_label='V3', v4_label='V4 aggregate'):
        # Special case: both values are zero — nothing meaningful to plot
        if v3_val == 0.0 and v4_mean == 0.0:
            ax.set_title(title, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_xlim(-1, 2)
            ax.set_ylim(0, 1)
            ax.set_xticks([0, 1])
            ax.set_xticklabels([v3_label.split()[0], v4_label.split()[0]])
            ax.text(0.5, 0.5,
                    "Both V3 and V4: 0.0%\nNo low-amplitude events detected",
                    ha='center', va='center', fontsize=10,
                    color='#1a7a1a', fontweight='bold',
                    transform=ax.transAxes)
            ax.grid(axis='y', alpha=0.4)
            return

        ax.bar(0, v3_val, width=BAR_WIDTH, color=V3_COLOR, label=v3_label)
        if v3_std > 0:
            ax.errorbar(0, v3_val, yerr=v3_std, **ERR_KW)
        ax.bar(1, v4_mean, width=BAR_WIDTH, color=V4_COLOR,
               yerr=v4_std, label=v4_label, error_kw=ERR_KW)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(XLABELS)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(ylim)
        ax.grid(axis='y', alpha=0.4)
        ax.legend(fontsize=8)
        if lower_is_better:
            better_x = 1 if v4_mean < v3_val else 0
        else:
            better_x = 1 if v4_mean > v3_val else 0
        better_y = (v4_mean + v4_std + better_offset) if better_x == 1 \
                   else (v3_val + (v3_std if v3_std > 0 else 0) + better_offset)
        ax.text(better_x, better_y, 'Better', **ANNOT_KW)

    # ── Row 0: Gaussian ────────────────────────────────────────────────────
    bar_pair(axs[0,0], v3g_mae, v3g_mae_std, v4g_mae_mean, v4g_mae_std,
             "MAE — Gaussian", "Frequency Error (Hz)",
             (0, 5500), 120, lower_is_better=True,
             v3_label='V3 (Gaussian)', v4_label='V4 (5 seeds × 1000 eps)')

    bar_pair(axs[0,1], v3g_reward, v3g_reward_std, v4g_rew_mean, v4g_rew_std,
             "Reward — Gaussian", "Cumulative Episode Reward",
             (160, 205), 1.5,
             v3_label='V3 (Gaussian)', v4_label='V4 (5 seeds × 1000 eps)')

    bar_pair(axs[0,2], v3g_near, 0, v4g_near_mean, v4g_near_std,
             "Amp > 0.90 — Gaussian", "% of Timesteps",
             (0, 115), 2.5,
             v3_label='V3 (Gaussian)', v4_label='V4 (5 seeds × 1000 eps)')

    bar_pair(axs[0,3], v3g_low, 0, v4g_low_mean, v4g_low_std,
             "Amp < 0.70 — Gaussian", "% of Timesteps",
             (0, 22), 0.4, lower_is_better=True,
             v3_label='V3 (Gaussian)', v4_label='V4 (5 seeds × 1000 eps)')

    # ── Row 1: OU ──────────────────────────────────────────────────────────
    bar_pair(axs[1,0], v3ou_mae, v3ou_mae_std, v4ou_mae_mean, v4ou_mae_std,
             "MAE — OU Noise", "Frequency Error (Hz)",
             (0, 2000), 40, lower_is_better=True,
             v3_label='V3 OU (1 seed)', v4_label='V4 OU (3 seeds × 1000 eps)')

    bar_pair(axs[1,1], v3ou_reward, v3ou_reward_std, v4ou_rew_mean, v4ou_rew_std,
             "Reward — OU Noise", "Cumulative Episode Reward",
             (188, 200), 0.4,
             v3_label='V3 OU (1 seed)', v4_label='V4 OU (3 seeds × 1000 eps)')

    bar_pair(axs[1,2], v3ou_near, 0, v4ou_near_mean, v4ou_near_std,
             "Amp > 0.90 — OU Noise", "% of Timesteps",
             (94, 101), 0.3,
             v3_label='V3 OU (1 seed)', v4_label='V4 OU (3 seeds × 1000 eps)')

    bar_pair(axs[1,3], v3ou_low, 0, v4ou_low_mean, v4ou_low_std,
             "Amp < 0.70 — OU Noise", "% of Timesteps",
             (-0.5, 2), 0.05, lower_is_better=True,
             v3_label='V3 OU (1 seed)', v4_label='V4 OU (3 seeds × 1000 eps)')

    plt.tight_layout()
    os.makedirs("data_logs", exist_ok=True)
    plt.savefig("data_logs/v3_v4_comparison.png", dpi=150, bbox_inches='tight')
    print("Comparison figure saved to data_logs/v3_v4_comparison.png")