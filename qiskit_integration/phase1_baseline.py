# phase1_baseline.py
import os, json, csv, numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from qiskit_integration.qiskit_resonator_env_v2 import QiskitResonatorEnvV2

# Resolve repository root (MiniProject_Sem4)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths relative to repo root
MODEL_PATH = os.path.join(REPO_ROOT, 'rl_training', 'trained_models', 'v4_gradient_obs', 'seed_3', 'best_model')
STATS_PATH = os.path.join(REPO_ROOT, 'rl_training', 'trained_models', 'v4_gradient_obs', 'seed_3', 'vec_normalize.pkl')
OUT_DIR = os.path.join(REPO_ROOT, 'data_logs', 'qiskit_aer_simulation_2')
os.makedirs(OUT_DIR, exist_ok=True)

N_EPISODES = 500

def make_env():
    return DummyVecEnv([lambda: QiskitResonatorEnvV2(shots=1024)]) # [cite: 885-886]

env = make_env()
env = VecNormalize.load(STATS_PATH, env) # [cite: 888]
env.training = False # [cite: 889]
env.norm_reward = False # [cite: 890]

model = PPO.load(MODEL_PATH, env=env) # [cite: 891]
print(f'Running {N_EPISODES} zero-shot baseline episodes...') # [cite: 892]

results = []
for ep in range(N_EPISODES): # [cite: 894]
    obs = env.reset() # [cite: 895]
    ep_amp, ep_err = [], [] # [cite: 896]
    done = False # [cite: 897]
    while not done: # [cite: 898]
        action, _ = model.predict(obs, deterministic=True) # [cite: 899]
        obs, _, done, info = env.step(action) # [cite: 900]
        ep_amp.append(float(info[0]['amplitude'])) # [cite: 901]
        ep_err.append(float(info[0]['freq_error_hz'])) # [cite: 902]
    results.append({
        'mae':   float(np.mean(ep_err)), # [cite: 904]
        'near':  float(np.mean(np.array(ep_amp) > 0.90)), # [cite: 905]
        'low':   float(np.mean(np.array(ep_amp) < 0.70)) # [cite: 906]
    })
    if (ep+1) % 100 == 0: # [cite: 908]
        print(f'  {ep+1}/{N_EPISODES} | MAE {np.mean([r["mae"] for r in results]):.0f} Hz') # [cite: 909]

summary = {
    'phase': 'zero_shot_baseline', # [cite: 911]
    'model': 'V4_Seed3_original', # [cite: 912]
    'mae_mean':  float(np.mean([r['mae']  for r in results])), # [cite: 913]
    'mae_std':   float(np.std( [r['mae']  for r in results])), # [cite: 914]
    'near_mean': float(np.mean([r['near'] for r in results])), # [cite: 915]
    'low_mean':  float(np.mean([r['low']  for r in results])) # [cite: 916]
}
with open(os.path.join(OUT_DIR,'baseline_summary.json'),'w') as f: # [cite: 918]
    json.dump(summary, f, indent=2) # [cite: 919]
print('Baseline complete:', summary) # [cite: 920]