import os
import sys
import json
import time
import argparse
import importlib
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback

class DriftCurriculumCallback(BaseCallback):
    """
    Gradually increases the stochastic drift difficulty during training.
    GPU Update: Annealing stretched to 750,000 steps to match the 2M total timestep run,
    giving the deeper [256, 256] network adequate time to learn fundamental mechanics.
    """
    def __init__(self, total_anneal_steps=750_000, start_sigma=100.0, end_sigma=500.0, verbose=0):
        super(DriftCurriculumCallback, self).__init__(verbose)
        self.total_anneal_steps = total_anneal_steps
        self.start_sigma = start_sigma
        self.end_sigma = end_sigma

    def _on_step(self) -> bool:
        fraction = min(1.0, self.num_timesteps / self.total_anneal_steps)
        current_sigma = self.start_sigma + fraction * (self.end_sigma - self.start_sigma)
        self.training_env.env_method("set_drift_sigma", current_sigma)

        # GPU Metric: Print VRAM allocated periodically to ensure we don't OOM
        if self.num_timesteps % 50000 == 0:
            vram_mb = torch.cuda.memory_allocated(0) / (1024**2)
            print(f"[CURRICULUM] Step {self.num_timesteps} -> drift_sigma = {current_sigma:.1f} Hz | VRAM: {vram_mb:.1f}MB")

        return True


# VERSION → environment module mapping
# Add new ablation variants here only — do not modify existing entries
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


def make_env(version):
    env_class = get_env_class(version)
    def _init():
        return env_class()
    return _init


def train():
    # GPU Verification Gate — inside train() so subprocess workers do not trigger it on re-import
    if not torch.cuda.is_available():
        print("FATAL ERROR: CUDA is not available. GPU training aborted.")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--version", type=str, default="v4_gradient_obs")
    args = parser.parse_args()
    seed = args.seed
    version = args.version

    # Validate version and confirm env class loads before touching GPU
    env_class = get_env_class(version)
    test_env = env_class()
    obs, _ = test_env.reset()
    assert len(obs) == 5, f"FATAL: obs length {len(obs)} != 5 for version {version}"
    print(f"[ENV CHECK] Version '{version}' loaded. obs shape: {obs.shape} — OK")
    del test_env

    # GPU Hardware Query
    gpu_name = torch.cuda.get_device_name(0)
    vram_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)

    print("=== GPU Training Pipeline Initialised ===")
    print(f"Device: CUDA | GPU: {gpu_name}")
    print(f"Total VRAM: {vram_total_mb:.1f} MB")
    print(f"Version: {version} | Seed: {seed}")

    # GPU-Optimized Scale Parameters
    num_envs = 16
    total_timesteps = 2_000_000

    # Ablation variants save to their own subdirectory
    if version in ("v4a", "v4b"):
        model_dir = os.path.join("rl_training", "trained_models", version + "_ablation", f"seed_{seed}")
    else:
        model_dir = os.path.join("rl_training", "trained_models", version, f"seed_{seed}")
    os.makedirs(model_dir, exist_ok=True)

    env = SubprocVecEnv([make_env(version) for _ in range(num_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = SubprocVecEnv([make_env(version)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    eval_cb = EvalCallback(eval_env, best_model_save_path=model_dir,
                           log_path=os.path.join("data_logs", "evals"),
                           eval_freq=50_000 // num_envs,
                           deterministic=True, render=False)

    checkpoint_cb = CheckpointCallback(save_freq=100_000 // num_envs, save_path=model_dir,
                                       name_prefix="ppo_resonator_ckpt")

    curriculum_cb = DriftCurriculumCallback()

    # GPU-Optimized PPO Hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        device="cuda",
        verbose=1,
        learning_rate=2.5e-4,
        n_steps=4096,
        batch_size=512,
        n_epochs=15,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=os.path.join("data_logs", "ppo_tensorboard")
    )

    # VRAM Safety Check before launch
    vram_used_mb = torch.cuda.memory_allocated(0) / (1024**2)
    print(f"VRAM Initialized: {vram_used_mb:.1f} MB / {vram_total_mb:.0f} MB")
    if vram_used_mb > vram_total_mb * 0.85:
        print("WARNING: VRAM usage above 85%. Consider reducing n_envs or batch_size.")

    print(f"\nCommencing {total_timesteps:,} step training run...")
    start_time = time.time()

    model.learn(total_timesteps=total_timesteps, callback=[curriculum_cb, eval_cb, checkpoint_cb])

    # Post-Training Metrics
    training_duration = time.time() - start_time
    mins, secs = divmod(training_duration, 60)
    peak_vram = torch.cuda.max_memory_allocated(0) / (1024**2)
    steps_per_sec = total_timesteps / training_duration

    print(f"\nTraining complete in {int(mins)}m {int(secs)}s | {steps_per_sec:,.0f} steps/sec | Peak VRAM: {peak_vram:.1f} MB")

    final_model_path = os.path.join(model_dir, "ppo_resonator_final")
    model.save(final_model_path)
    env.save(os.path.join(model_dir, "vec_normalize.pkl"))

    metadata = {
        "gpu_name": gpu_name,
        "peak_vram_mb": peak_vram,
        "total_timesteps": total_timesteps,
        "training_duration_seconds": training_duration,
        "n_steps": 4096,
        "batch_size": 512,
        "net_arch": "[256, 256]",
        "final_sigma": 500.0,
        "seed": seed,
        "version": version
    }
    with open(os.path.join(model_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    train()