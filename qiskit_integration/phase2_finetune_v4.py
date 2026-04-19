# phase2_finetune_v4.py
# Final fine-tuning with VecNormalize warmup
# Phase A (25k steps): VN training=True — statistics adapt to V3 env
# Phase B (125k steps): VN training=False — locked normalisation
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from qiskit_integration.qiskit_resonator_env_v3 import QiskitResonatorEnvV3

MODEL_PATH = (
    r"C:\MiniProject_Sem4\rl_training\trained_models"
    r"\v4_gradient_obs\seed_3\best_model"
)
STATS_PATH = (
    r"C:\MiniProject_Sem4\rl_training\trained_models"
    r"\v4_gradient_obs\seed_3\vec_normalize.pkl"
)
SAVE_DIR = r"C:\MiniProject_Sem4\data_logs" r"\qiskit_aer_simulation_2\v4_qft_v4"
os.makedirs(SAVE_DIR, exist_ok=True)

WARMUP_STEPS = 25000
FINETUNE_STEPS = 125000
LR = 5e-5


def make_env(seed):
    return DummyVecEnv([lambda: QiskitResonatorEnvV3(shots=1024, seed=seed)])


# ── Phase A: VecNormalize statistics warmup (training=True) ──────────────
# This allows the normalisation running mean/variance to update toward
# the V3 environment distribution before the policy is fine-tuned.
# Without this, amp_gradient (obs element 4) is out-of-distribution
# causing the systematic upward frequency offset seen in V3 results.
train_a = make_env(0)
train_a = VecNormalize.load(STATS_PATH, train_a)
train_a.training = True  # ← KEY: stats update during warmup
train_a.norm_reward = False

eval_a = make_env(99)
eval_a = VecNormalize.load(STATS_PATH, eval_a)
eval_a.training = True  # ← also update eval stats during warmup
eval_a.norm_reward = False

model = PPO.load(MODEL_PATH, env=train_a)


def constant_lr(_):
    return LR


model.lr_schedule = constant_lr
model.learning_rate = LR
model.policy.optimizer.param_groups[0]["lr"] = LR

print("Phase A: VN Warmup — statistics adapting to V3 environment")
print(f"LR: {LR} | Warmup steps: {WARMUP_STEPS}")

model.learn(total_timesteps=WARMUP_STEPS, reset_num_timesteps=False)

warmup_stats = os.path.join(SAVE_DIR, "vec_normalize_warmup.pkl")
train_a.save(warmup_stats)
print(f"Warmup done. Updated VN stats saved to: {warmup_stats}")
# ── Phase B: Fine-tuning with locked statistics ───────────────────────────
train_b = make_env(0)
train_b = VecNormalize.load(warmup_stats, train_b)
train_b.training = False  # ← locked after warmup
train_b.norm_reward = False

eval_b = make_env(99)
eval_b = VecNormalize.load(warmup_stats, eval_b)
eval_b.training = False
eval_b.norm_reward = False

model.set_env(train_b)

eval_cb = EvalCallback(
    eval_b,
    best_model_save_path=SAVE_DIR,
    log_path=os.path.join(SAVE_DIR, "logs"),
    eval_freq=5000,
    n_eval_episodes=20,
    deterministic=True,
    render=False,
)

print("Phase B: Fine-tuning (VN locked)")
print(f"LR: {LR} | Fine-tune steps: {FINETUNE_STEPS}")

model.learn(total_timesteps=FINETUNE_STEPS, callback=eval_cb, reset_num_timesteps=False)

model.save(os.path.join(SAVE_DIR, "v4_qft_v4_final"))
train_b.save(os.path.join(SAVE_DIR, "vec_normalize_qft_v4.pkl"))
print("V4 fine-tuning complete. Saved to:", SAVE_DIR)
