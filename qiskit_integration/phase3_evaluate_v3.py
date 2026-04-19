# phase3_evaluate_v3.py
import os, json, numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from qiskit_integration.qiskit_resonator_env_v3 import QiskitResonatorEnvV3

QFT_MODEL = r'C:\MiniProject_Sem4\data_logs\qiskit_aer_simulation_2\v4_qft_v3\best_model'
QFT_STATS = r'C:\MiniProject_Sem4\data_logs\qiskit_aer_simulation_2\v4_qft_v3\vec_normalize_qft_v3.pkl'
OUT_DIR   = r'C:\MiniProject_Sem4\data_logs\qiskit_aer_simulation_2'
os.makedirs(OUT_DIR, exist_ok=True)

N_EPISODES = 500

env = DummyVecEnv([lambda: QiskitResonatorEnvV3(shots=1024)])
env = VecNormalize.load(QFT_STATS, env)
env.training = False
env.norm_reward = False

model = PPO.load(QFT_MODEL, env=env)
print(f'Evaluating V4_QFT_V3 over {N_EPISODES} episodes...')

results, trajectories = [], []

for ep in range(N_EPISODES):
    obs = env.reset()
    ep_amp, ep_err = [], []
    ep_fp, ep_fq = [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = env.step(action)
        ep_amp.append(float(info[0]['amplitude']))
        ep_err.append(float(info[0]['freq_error_hz']))
        ep_fp.append(float(info[0]['f_probe']))
        ep_fq.append(float(info[0]['f_qubit']))
    results.append({
        'mae':  float(np.mean(ep_err)),
        'near': float(np.mean(np.array(ep_amp) > 0.90)),
        'low':  float(np.mean(np.array(ep_amp) < 0.70))
    })
    if ep < 10:
        trajectories.append({'f_probe':ep_fp,'f_qubit':ep_fq,'amp':ep_amp})
    if (ep+1) % 100 == 0:
        print(f'  {ep+1}/{N_EPISODES} | MAE {np.mean([r["mae"] for r in results]):.0f} Hz')

summary = {
    'phase':'v3_eval','model':'V4_QFT_V3',
    'mae_mean': float(np.mean([r['mae']  for r in results])),
    'mae_std':  float(np.std( [r['mae']  for r in results])),
    'near_mean':float(np.mean([r['near'] for r in results])),
    'near_std': float(np.std( [r['near'] for r in results])),
    'low_mean': float(np.mean([r['low']  for r in results]))
}
with open(os.path.join(OUT_DIR,'qft_v3_summary.json'),'w') as f:
    import json; json.dump(summary,f,indent=2)
with open(os.path.join(OUT_DIR,'qft_v3_trajectories.json'),'w') as f:
    json.dump(trajectories,f)
print('V3 eval complete:')
print(f"  MAE:  {summary['mae_mean']:.0f} +- {summary['mae_std']:.0f} Hz")
print(f"  Near: {summary['near_mean']*100:.1f}% +- {summary['near_std']*100:.1f}%")