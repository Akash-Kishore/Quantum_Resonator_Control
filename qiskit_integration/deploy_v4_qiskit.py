import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from qiskit_integration.qiskit_resonator_env import QiskitResonatorEnv

# Absolute paths for V4 Seed 3
MODEL_PATH = r"C:\MiniProject_Sem4\rl_training\trained_models\v4_gradient_obs\seed_3\best_model"
STATS_PATH = r"C:\MiniProject_Sem4\rl_training\trained_models\v4_gradient_obs\seed_3\vec_normalize.pkl"

def deploy():
    # 1. Initialize the NEW "Critically Damped" Environment
    # This automatically uses 4096 shots and the EMA filters
    raw_env = QiskitResonatorEnv(shots=4096)
    env = DummyVecEnv([lambda: raw_env])

    # 2. Load Normalization Stats
    if not os.path.exists(STATS_PATH):
        print(f"ERROR: Stats not found at {STATS_PATH}")
        return
    env = VecNormalize.load(STATS_PATH, env)
    env.training = False
    env.norm_reward = False

    # 3. Load Trained Model
    model = PPO.load(MODEL_PATH, env=env)
    print(f"SUCCESS: Loaded V4 Seed 3. Beginning smooth deep tracking...")

    # 4. Evaluation Loop
    obs = env.reset()
    history = {"true_f0": [], "pred_f": [], "amp": []}
    
    for step in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        
        # Access the internal state of the environment
        inner = env.envs[0]
        
        # Log True Frequency (The Red Dashed Line)
        history["true_f0"].append(inner.f0_current)
        
        # Log Predicted Frequency (The Blue Line)
        # CRUCIAL FIX 1: Must use the smoothed action (inner.prev_action_val)
        # CRUCIAL FIX 2: Must use the environment multiplier (30000.0)
        history["pred_f"].append(inner.f0_nominal + (inner.prev_action_val * inner.multiplier))
        
        # Log Amplitude (The Green Line)
        history["amp"].append(inner.prev_amp)
        
        # Inject Gaussian drift (sigma=500 Hz)
        inner.f0_current += np.random.normal(0, 500)

    # 5. Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Frequency Tracking Plot
    ax1.plot(np.array(history["true_f0"])/1000, 'r--', alpha=0.7, label="Quantum Drift (Target)")
    ax1.plot(np.array(history["pred_f"])/1000, 'b-', linewidth=2, label="Damped Agent Tracking")
    ax1.set_ylabel("Frequency (kHz)")
    ax1.set_title("V4 Seed 3: Critically Damped Qiskit Tracking (30kHz Deep Reach)")
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)

    # Amplitude Fidelity Plot
    ax2.plot(history["amp"], 'g-', linewidth=1.5, label="Smoothed P(|1>)")
    ax2.axhline(y=0.9, color='k', linestyle=':', label="0.90 Threshold")
    ax2.set_ylabel("Excitation Prob.")
    ax2.set_xlabel("Time Steps")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    # Save with a new filename for comparison
    plt.savefig("qiskit_smooth_tracking_final.png")
    print("Inference complete. Check 'qiskit_smooth_tracking_final.png'.")
    plt.show()

if __name__ == "__main__":
    deploy()