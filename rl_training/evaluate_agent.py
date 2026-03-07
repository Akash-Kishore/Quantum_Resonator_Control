import os
import sys
import time
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rl_training.rl_environment import ResonatorEnv

def evaluate():
    model_dir = os.path.join("rl_training", "trained_models")
    model_path = os.path.join(model_dir, "best_model")
    vec_norm_path = os.path.join(model_dir, "vec_normalize.pkl")
    
    # GPU Update: Check hardware and route inference accordingly
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing environment and loading policy on: {device.upper()}...", flush=True)
    
    try:
        env = DummyVecEnv([lambda: ResonatorEnv()])
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False 
        env.norm_reward = False
        
        # Explicitly load model onto the target device
        model = PPO.load(model_path, device=device)
    except Exception as e:
        print(f"Failed to load model or normalization stats: {e}", flush=True)
        sys.exit(1)
        
    num_episodes = 10
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
            # GPU Update: Measure inference speed precisely
            t_start = time.perf_counter()
            action, _states = model.predict(obs, deterministic=True)
            t_end = time.perf_counter()
            inference_times.append((t_end - t_start) * 1000) # Convert to milliseconds
            
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

    # Calculate amplitude stability metric
    amps_array = np.array(all_amplitudes)
    amp_stability_pct = (np.sum(amps_array > 0.90) / len(amps_array)) * 100
    avg_inference_ms = np.mean(inference_times)

    print("\n=== 10-Episode Evaluation ===")
    print(f"Mean MAE:      {np.mean(all_maes):.0f} Hz ± {np.std(all_maes):.0f} Hz")
    print(f"Mean Reward:   {np.mean(all_rewards):.1f} ± {np.std(all_rewards):.1f}")
    print(f"Amplitude > 0.90: {amp_stability_pct:.1f}% of timesteps")
    print(f"Inference speed:  {avg_inference_ms:.3f} ms/step ({device.upper()})")

    print("\nGenerating performance visualizations...", flush=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=False)
    
    for i in range(min(3, num_episodes)):
        alpha = 0.8 if i == 0 else 0.3 
        ax1.plot(plot_true_freqs[i], color="red", linestyle="--", alpha=alpha, label="True Resonant Freq" if i==0 else "")
        ax1.plot(plot_agent_freqs[i], color="blue", alpha=alpha, label="Agent Controlled Freq" if i==0 else "")
        
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_title("Autonomous Resonance Tracking (3 Sample Trajectories)")
    ax1.legend()
    ax1.grid(True)
    
    ax2.hist(all_amplitudes, bins=40, color="green", alpha=0.7)
    ax2.set_ylabel("Count")
    ax2.set_xlabel("Measured EMA Amplitude")
    ax2.set_title(f"Amplitude Distribution ({num_episodes} Episodes)")
    ax2.axvline(x=1.0, color='r', linestyle=':', label="Ideal Max")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    os.makedirs("data_logs", exist_ok=True)
    plot_path = os.path.join("data_logs", "tracking_evaluation.png")
    plt.savefig(plot_path)
    print(f"Evaluation visualization saved successfully to {plot_path}", flush=True)

if __name__ == "__main__":
    evaluate()