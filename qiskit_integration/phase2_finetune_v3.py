# phase2_finetune_v3.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from qiskit_integration.qiskit_resonator_env_v3 import QiskitResonatorEnvV3

# START FROM ORIGINAL SIMULATION CHECKPOINT — not v2
MODEL_PATH  = r'C:\MiniProject_Sem4\rl_training\trained_models\v4_gradient_obs\seed_3\best_model'
STATS_PATH  = r'C:\MiniProject_Sem4\rl_training\trained_models\v4_gradient_obs\seed_3\vec_normalize.pkl'
SAVE_DIR    = r'C:\MiniProject_Sem4\data_logs\qiskit_aer_simulation_2\v4_qft_v3'
os.makedirs(SAVE_DIR, exist_ok=True)

TOTAL_TIMESTEPS = 150000
LR_FINETUNE     = 5e-5

def make_train_env():
    return DummyVecEnv([lambda: QiskitResonatorEnvV3(shots=1024, seed=0)])

def make_eval_env():
    return DummyVecEnv([lambda: QiskitResonatorEnvV3(shots=1024, seed=99)])

train_env = make_train_env()
train_env = VecNormalize.load(STATS_PATH, train_env)
train_env.training = False
train_env.norm_reward = False

eval_env = make_eval_env()
eval_env = VecNormalize.load(STATS_PATH, eval_env)
eval_env.training = False
eval_env.norm_reward = False

model = PPO.load(MODEL_PATH, env=train_env)

def constant_lr(progress_remaining): return LR_FINETUNE
model.lr_schedule = constant_lr
model.learning_rate = LR_FINETUNE
model.policy.optimizer.param_groups[0]['lr'] = LR_FINETUNE

print('Fine-tuning V4 Seed 3 -> V4_QFT_V3 (action_scale=1000, omega_rabi=2500)')
print(f'LR: {LR_FINETUNE} | Steps: {TOTAL_TIMESTEPS} | Shots: 1024/step')

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path=SAVE_DIR,
    log_path=os.path.join(SAVE_DIR, 'logs'),
    eval_freq=5000,
    n_eval_episodes=20,
    deterministic=True,
    render=False
)

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=eval_cb,
    reset_num_timesteps=False
)

model.save(os.path.join(SAVE_DIR, 'v4_qft_v3_final'))
train_env.save(os.path.join(SAVE_DIR, 'vec_normalize_qft_v3.pkl'))
print('V3 fine-tuning complete. Saved to:', SAVE_DIR)