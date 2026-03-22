import os
import sys
import time
import importlib
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import argparse

# VERSION → environment module mapping (must match train.py)
ENV_MODULE_MAP = {
    "v4_gradient_obs": "rl_training.rl_environment",
    "v4a":             "rl_training.rl_environment_v4a",
    "v4b":             "rl_training.rl_environment_v4b",
}

def get_env_class(version):
    """Imports and returns the ResonatorEnv class for the given version string."""
    if version not in ENV_MODULE_MAP:
        print(f"FATAL ERROR: Unknown version '{version}'. Valid options: {list(ENV_MODULE_MAP.keys())}")
        sys.exit(1)
    module = importlib.import_module(ENV_MODULE_MAP[version])
    return module.ResonatorEnv


def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="v3_refined")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--version", type=str, default=None,
                        help="Environment version to use for evaluation. "
                             "Defaults to 'v4_gradient_obs' for v4 models, "
                             "or the base env for v3. Set explicitly for v4a/v4b.")
    parser.add_argument("--drift_sigma", type=float, default=None,
                        help="Override drift sigma (Hz) in the Gaussian resonator model. "
                             "Training default is 500 Hz. Use 750 or 1000 for robustness test. "
                             "Does NOT modify any source file — runtime override only.")
    args = parser.parse_args()

    # Resolve model directory
    if args.seed is not None:
        model_dir = os.path.join("rl_training", "trained_models", args.model, f"seed_{args.seed}")
    else:
        model_dir = os.path.join("rl_training", "trained_models", args.model)
    model_path = os.path.join(model_dir, "best_model")
    vec_norm_path = os.path.join(model_dir, "vec_normalize.pkl")

    # Resolve environment version
    # If not specified: use v4_gradient_obs for any v4 model, base env for v3
    if args.version is not None:
        version = args.version
    elif "v4a" in args.model:
        version = "v4a"
    elif "v4b" in args.model:
        version = "v4b"
    elif "v4" in args.model:
        version = "v4_gradient_obs"
    else:
        # v3 or unknown — fall back to base env (4-element obs v3 env is in rl_environment_copy.py,
        # but v3 was trained with the base rl_environment.py at the time; keep using it here)
        version = "v4_gradient_obs"

    env_class = get_env_class(version)

    # GPU Update: Check hardware and route inference accordingly
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing environment and loading policy on: {device.upper()}...", flush=True)
    print(f"Environment version: {version}", flush=True)
    if args.drift_sigma is not None:
        print(f"drift_sigma OVERRIDE: {args.drift_sigma} Hz (training default: 500 Hz)", flush=True)

    try:
        def make_eval_env():
            env = env_class()
            if args.drift_sigma is not None:
                env.set_drift_sigma(args.drift_sigma)
            return env

        env = DummyVecEnv([make_eval_env])
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False

        # Explicitly load model onto the target device
        model = PPO.load(model_path, device=device)
    except Exception as e:
        print(f"Failed to load model or normalization stats: {e}", flush=True)
        sys.exit(1)

    num_episodes = 1000
    all_maes = []
    all_rewards = []
    all_amplitudes = []
    inference_times = []

    plot_true_freqs = []
    plot_agent_freqs = []

    print(f"Executing {num_episodes} deterministic evaluation episodes...", flush=True)

    for ep in range(num_episodes):
        obs = env.reset()
        ep_true_freqs = []
        ep_agent_freqs = []
        ep_amps = []
        ep_reward = 0.0

        for step in range(200):
            t_start = time.perf_counter()
            action, _states = model.predict(obs, deterministic=True)
            t_end = time.perf_counter()
            inference_times.append((t_end - t_start) * 1000)

            obs, reward, done, info = env.step(action)

            true_f0 = env.envs[0].resonator.f0_current
            ep_true_freqs.append(true_f0)

            agent_freq = env.envs[0].current_freq
            ep_agent_freqs.append(agent_freq)

            amp = env.envs[0].ema_amp
            ep_amps.append(amp)
            all_amplitudes.append(amp)
            ep_reward += reward[0]

            if done[0]:
                break

        mae = np.mean(np.abs(np.array(ep_true_freqs) - np.array(ep_agent_freqs)))
        all_maes.append(mae)
        all_rewards.append(ep_reward)

        plot_true_freqs.append(ep_true_freqs)
        plot_agent_freqs.append(ep_agent_freqs)

    # Calculate amplitude stability metrics
    amps_array = np.array(all_amplitudes)
    amp_stability_pct = (np.sum(amps_array > 0.90) / len(amps_array)) * 100
    low_amp_tail_pct = (np.sum(amps_array < 0.70) / len(amps_array)) * 100
    avg_inference_ms = np.mean(inference_times)

    # Build label for drift sigma in output and filename
    sigma_label = f"sigma{int(args.drift_sigma)}" if args.drift_sigma is not None else "sigma500"

    print(f"\n=== {num_episodes}-Episode Evaluation | Model: {args.model} | Seed: {args.seed} | Version: {version} | drift_sigma: {args.drift_sigma or 500} Hz ===")
    print(f"Mean MAE:         {np.mean(all_maes):.0f} Hz +/- {np.std(all_maes):.0f} Hz")
    print(f"Mean Reward:      {np.mean(all_rewards):.1f} +/- {np.std(all_rewards):.1f}")
    print(f"Amplitude > 0.90: {amp_stability_pct:.1f}% of timesteps")
    print(f"Amplitude < 0.70: {low_amp_tail_pct:.1f}% of timesteps")
    print(f"Inference speed:  {avg_inference_ms:.3f} ms/step ({device.upper()})")

    print("\nGenerating performance visualizations...", flush=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=False)

    for i in range(min(3, num_episodes)):
        alpha = 0.8 if i == 0 else 0.3
        ax1.plot(plot_true_freqs[i], color="red", linestyle="--", alpha=alpha,
                 label="True Resonant Freq" if i == 0 else "")
        ax1.plot(plot_agent_freqs[i], color="blue", alpha=alpha,
                 label="Agent Controlled Freq" if i == 0 else "")

    drift_title = f"drift_sigma={int(args.drift_sigma)} Hz" if args.drift_sigma else "drift_sigma=500 Hz (training)"
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_title(f"Autonomous Resonance Tracking — {args.model} | {drift_title} (3 Sample Trajectories)")
    ax1.legend()
    ax1.grid(True)

    ax2.hist(all_amplitudes, bins=40, color="green", alpha=0.7)
    ax2.set_ylabel("Count")
    ax2.set_xlabel("Measured EMA Amplitude")
    ax2.set_title(f"Amplitude Distribution ({num_episodes} Episodes) — {drift_title}")
    ax2.axvline(x=1.0, color='r', linestyle=':', label="Ideal Max")
    ax2.axvline(x=0.90, color='orange', linestyle='--', label="Near-resonance threshold (0.90)")
    ax2.axvline(x=0.70, color='red', linestyle='--', label="Low-amplitude threshold (0.70)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    os.makedirs("data_logs", exist_ok=True)

    seed_str = f"_seed{args.seed}" if args.seed is not None else ""
    plot_filename = f"tracking_evaluation_{args.model}{seed_str}_{version}_{sigma_label}.png"
    plot_path = os.path.join("data_logs", plot_filename)
    plt.savefig(plot_path)
    print(f"Evaluation visualization saved to {plot_path}", flush=True)

if __name__ == "__main__":
    evaluate()
