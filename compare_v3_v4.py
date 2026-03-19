import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Hardcoded data ---
v3_mae = 3333
v3_mae_std = 1932
v3_reward = 174.3
v3_reward_std = 20.2
v3_near_resonance = 64.5
v3_low_amp = 13.4

v4_mae_per_seed = [2288, 1731, 2608, 1510, 2887]
v4_reward_per_seed = [184.5, 190.2, 181.4, 192.5, 178.9]
v4_near_resonance_per_seed = [85.0, 91.2, 78.9, 94.8, 74.3]
v4_low_amp_per_seed = [5.8, 1.3, 8.2, 0.4, 9.8]

# --- Aggregates ---
v4_mae_mean = np.mean(v4_mae_per_seed)
v4_mae_std = np.std(v4_mae_per_seed)
v4_reward_mean = np.mean(v4_reward_per_seed)
v4_reward_std = np.std(v4_reward_per_seed)
v4_near_resonance_mean = np.mean(v4_near_resonance_per_seed)
v4_near_resonance_std = np.std(v4_near_resonance_per_seed)
v4_low_amp_mean = np.mean(v4_low_amp_per_seed)
v4_low_amp_std = np.std(v4_low_amp_per_seed)

if __name__ == "__main__":
    print("=== V3 vs V4 Aggregate Comparison ===")
    print(f"MAE:        V3 = {v3_mae:.0f} ± {v3_mae_std:.0f} Hz  |  V4 = {v4_mae_mean:.0f} ± {v4_mae_std:.0f} Hz")
    print(f"Reward:     V3 = {v3_reward:.1f} ± {v3_reward_std:.1f}  |  V4 = {v4_reward_mean:.1f} ± {v4_reward_std:.1f}")
    print(f"Amp > 0.90: V3 = {v3_near_resonance:.1f}%  |  V4 = {v4_near_resonance_mean:.1f}% ± {v4_near_resonance_std:.1f}%")
    print(f"Amp < 0.70: V3 = {v3_low_amp:.1f}%  |  V4 = {v4_low_amp_mean:.1f}% ± {v4_low_amp_std:.1f}%")

    # Shared style
    BAR_WIDTH = 0.5
    V3_COLOR = '#d9534f'
    V4_COLOR = '#5b9bd5'
    ERR_KW = dict(ecolor='#333333', capsize=7, capthick=2, elinewidth=2)
    ANNOT_KW = dict(ha='center', va='center', fontsize=10,
                    color='#1a7a1a', fontweight='bold')

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("V3 vs V4: Gradient-Augmented Observation Space",
                 fontsize=14, fontweight='bold')

    # ── Subplot 1: MAE (lower is better) ──────────────────────────────────
    ax = axs[0, 0]
    ax.bar(0, v3_mae, width=BAR_WIDTH, color=V3_COLOR, label='V3 (100 eps)')
    ax.bar(1, v4_mae_mean, width=BAR_WIDTH, color=V4_COLOR,
           label='V4 (5 seeds × 1000 eps)', error_kw=ERR_KW, yerr=v4_mae_std)
    ax.set_title("Mean Tracking Error (MAE)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['V3', 'V4'])
    ax.set_ylabel("Frequency Error (Hz)")
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', alpha=0.4)
    ax.legend()
    ax.text(1, v4_mae_mean + v4_mae_std + 80, 'Better', **ANNOT_KW)

    # ── Subplot 2: Reward (higher is better) ──────────────────────────────
    ax = axs[0, 1]
    ax.bar(0, v3_reward, width=BAR_WIDTH, color=V3_COLOR, label='V3 (100 eps)')
    ax.bar(1, v4_reward_mean, width=BAR_WIDTH, color=V4_COLOR,
           label='V4 (5 seeds × 1000 eps)', error_kw=ERR_KW, yerr=v4_reward_std)
    ax.set_title("Mean Episode Reward")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['V3', 'V4'])
    ax.set_ylabel("Cumulative Episode Reward")
    ax.set_ylim(bottom=160)
    ax.grid(axis='y', alpha=0.4)
    ax.legend()
    ax.text(1, v4_reward_mean + v4_reward_std + 1.8, 'Better', **ANNOT_KW)

    # ── Subplot 3: Near-resonance time (higher is better) ─────────────────
    ax = axs[1, 0]
    ax.bar(0, v3_near_resonance, width=BAR_WIDTH, color=V3_COLOR,
           label='V3 (100 eps)')
    ax.bar(1, v4_near_resonance_mean, width=BAR_WIDTH, color=V4_COLOR,
           label='V4 (5 seeds × 1000 eps)', error_kw=ERR_KW,
           yerr=v4_near_resonance_std)
    ax.set_title("Near-Resonance Time (Amplitude > 0.90)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['V3', 'V4'])
    ax.set_ylabel("% of Timesteps")
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.4)
    ax.legend()
    ax.text(1, v4_near_resonance_mean + v4_near_resonance_std + 2.5,
            'Better', **ANNOT_KW)

    # ── Subplot 4: Low-amplitude tail (lower is better) ───────────────────
    ax = axs[1, 1]
    ax.bar(0, v3_low_amp, width=BAR_WIDTH, color=V3_COLOR,
           label='V3 (100 eps)')
    ax.bar(1, v4_low_amp_mean, width=BAR_WIDTH, color=V4_COLOR,
           label='V4 (5 seeds × 1000 eps)', error_kw=ERR_KW,
           yerr=v4_low_amp_std)
    ax.set_title("Low-Amplitude Tail (Amplitude < 0.70)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['V3', 'V4'])
    ax.set_ylabel("% of Timesteps")
    ax.set_ylim(0, 20)
    ax.grid(axis='y', alpha=0.4)
    ax.legend()
    ax.text(1, v4_low_amp_mean + v4_low_amp_std + 0.5, 'Better', **ANNOT_KW)

    plt.tight_layout()
    os.makedirs("data_logs", exist_ok=True)
    plt.savefig("data_logs/v3_v4_comparison.png", dpi=150)
    print("Comparison figure saved to data_logs/v3_v4_comparison.png")