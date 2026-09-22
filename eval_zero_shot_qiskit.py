import os, sys, numpy as np
sys.path.insert(0, r'C:\Projects\MiniProject_Sem4')
os.chdir(r'C:\Projects\MiniProject_Sem4')

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from qiskit_integration.qiskit_resonator_env_v2 import QiskitResonatorEnvV2

# Load classical V4 seed3 checkpoint
ckpt_dir = os.path.join("rl_training", "trained_models", "v4_gradient_obs", "seed_3")
model_path = os.path.join(ckpt_dir, "best_model.zip")
vecnorm_path = os.path.join(ckpt_dir, "vec_normalize.pkl")

def make_env(seed):
    return QiskitResonatorEnvV2(shots=1024, seed=seed)

num_episodes = 1000
seeds = [0,1,2,3,4]
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

all_mae = []
all_near = []
all_low = []

for seed in seeds:
    print(f"\n=== Seed {seed} ===")
    env = DummyVecEnv([lambda: make_env(seed)])
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False
    model = PPO.load(model_path, device=device)

    maes = []
    rewards = []
    amps = []
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        ep_mae = []
        ep_rew = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            # info is list of dicts
            freq_err = info[0]['freq_error_hz']
            amp = info[0]['amplitude']
            ep_mae.append(freq_err)
            amps.append(amp)
            ep_rew += reward[0]
            done = done[0]
        maes.append(np.mean(ep_mae))
        rewards.append(ep_rew)
    amps = np.array(amps)
    near = (np.sum(amps>0.90)/len(amps))*100
    low = (np.sum(amps<0.70)/len(amps))*100
    print(f"MAE: {np.mean(maes):.0f}±{np.std(maes):.0f} Hz")
    print(f"Reward: {np.mean(rewards):.1f}±{np.std(rewards):.1f}")
    print(f"Near>0.90: {near:.1f}%")
    print(f"Low<0.70: {low:.1f}%")
    all_mae.extend(maes)
    all_near.append(near)
    all_low.append(low)

print("\n=== Aggregate over 5 seeds ===")
print(f"MAE: {np.mean(all_mae):.0f}±{np.std(all_mae):.0f} Hz")
print(f"Near>0.90: {np.mean(all_near):.1f}% ± {np.std(all_near):.1f}%")
print(f"Low<0.70: {np.mean(all_low):.1f}% ± {np.std(all_low):.1f}%")