import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    print(f"MAE:            V3 = {v3_mae:.0f} ± {v3_mae_std:.0f} Hz        |  V4 = {v4_mae_mean:.0f} ± {v4_mae_std:.0f} Hz")
    print(f"Reward:         V3 = {v3_reward:.1f} ± {v3_reward_std:.1f}        |  V4 = {v4_reward_mean:.1f} ± {v4_reward_std:.1f}")
    print(f"Amp > 0.90:     V3 = {v3_near_resonance:.1f}%        |  V4 = {v4_near_resonance_mean:.1f}% ± {v4_near_resonance_std:.1f}%")
    print(f"Amp < 0.70:     V3 = {v3_low_amp:.1f}%        |  V4 = {v4_low_amp_mean:.1f}% ± {v4_low_amp_std:.1f}%")

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("V3 vs V4: Gradient-Augmented Observation Space", fontsize=14, fontweight='bold')

    axs[0, 0].set_title("Mean Tracking Error (MAE)")
    axs[0, 0].bar(0, v3_mae, yerr=v3_mae_std, width=0.5, color='#d9534f', label='V3 (100 eps)', error_kw=dict(ecolor='black', capsize=6, capthick=1.5))
    axs[0, 0].bar(1, v4_mae_mean, yerr=v4_mae_std, width=0.5, color='#5b9bd5', label='V4 (5 seeds × 1000 eps)', error_kw=dict(ecolor='black', capsize=6, capthick=1.5))
    axs[0, 0].set_xticks([0, 1])
    axs[0, 0].set_xticklabels(['V3', 'V4'])
    axs[0, 0].set_ylabel("Frequency Error (Hz)")
    axs[0, 0].legend()
    axs[0, 0].grid(axis='y', alpha=0.5)
    axs[0, 0].text(1, v4_mae_mean * 0.05, '↓ Better', ha='center', fontsize=9, color='green')

    axs[0, 1].set_title("Mean Episode Reward")
    axs[0, 1].bar(0, v3_reward, yerr=v3_reward_std, width=0.5, color='#d9534f', label='V3 (100 eps)', error_kw=dict(ecolor='black', capsize=6, capthick=1.5))
    axs[0, 1].bar(1, v4_reward_mean, yerr=v4_reward_std, width=0.5, color='#5b9bd5', label='V4 (5 seeds × 1000 eps)', error_kw=dict(ecolor='black', capsize=6, capthick=1.5))
    axs[0, 1].set_xticks([0, 1])
    axs[0, 1].set_xticklabels(['V3', 'V4'])
    axs[0, 1].set_ylabel("Cumulative Episode Reward")
    axs[0, 1].set_ylim(bottom=160)
    axs[0, 1].legend()
    axs[0, 1].grid(axis='y', alpha=0.5)
    axs[0, 1].text(1, v4_reward_mean * 1.005, '↑ Better', ha='center', fontsize=9, color='green')

    axs[1, 0].set_title("Near-Resonance Time (Amplitude > 0.90)")
    axs[1, 0].bar(0, v3_near_resonance, yerr=0, width=0.5, color='#d9534f', label='V3 (100 eps)', error_kw=dict(ecolor='black', capsize=6, capthick=1.5))
    axs[1, 0].bar(1, v4_near_resonance_mean, yerr=v4_near_resonance_std, width=0.5, color='#5b9bd5', label='V4 (5 seeds × 1000 eps)', error_kw=dict(ecolor='black', capsize=6, capthick=1.5))
    axs[1, 0].set_xticks([0, 1])
    axs[1, 0].set_xticklabels(['V3', 'V4'])
    axs[1, 0].set_ylabel("% of Timesteps")
    axs[1, 0].set_ylim(0, 105)
    axs[1, 0].legend()
    axs[1, 0].grid(axis='y', alpha=0.5)
    axs[1, 0].text(1, v4_near_resonance_mean + 2, '↑ Better', ha='center', fontsize=9, color='green')
    for i, val in enumerate(v4_near_resonance_per_seed):
        axs[1, 0].scatter(1, val, color='navy', s=20, zorder=5, alpha=0.6)

    axs[1, 1].set_title("Low-Amplitude Tail (Amplitude < 0.70)")
    axs[1, 1].bar(0, v3_low_amp, yerr=0, width=0.5, color='#d9534f', label='V3 (100 eps)', error_kw=dict(ecolor='black', capsize=6, capthick=1.5))
    axs[1, 1].bar(1, v4_low_amp_mean, yerr=v4_low_amp_std, width=0.5, color='#5b9bd5', label='V4 (5 seeds × 1000 eps)', error_kw=dict(ecolor='black', capsize=6, capthick=1.5))
    axs[1, 1].set_xticks([0, 1])
    axs[1, 1].set_xticklabels(['V3', 'V4'])
    axs[1, 1].set_ylabel("% of Timesteps")
    axs[1, 1].set_ylim(0, 20)
    axs[1, 1].legend()
    axs[1, 1].grid(axis='y', alpha=0.5)
    axs[1, 1].text(1, v4_low_amp_mean + 0.5, '↓ Better', ha='center', fontsize=9, color='green')
    for i, val in enumerate(v4_low_amp_per_seed):
        axs[1, 1].scatter(1, val, color='navy', s=20, zorder=5, alpha=0.6)

    plt.tight_layout()
    os.makedirs("data_logs", exist_ok=True)
    plt.savefig("data_logs/v3_v4_comparison.png", dpi=150)
    print("Comparison figure saved to data_logs/v3_v4_comparison.png")