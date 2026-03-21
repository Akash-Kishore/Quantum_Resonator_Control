import os
import numpy as np
from scipy.stats import mannwhitneyu
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym

NUM_EPISODES = 1000


def make_v3_env_gaussian():
    from gymnasium import spaces
    from simulation.resonator_model_gaussian import QuantumResonatorSim

    class ResonatorEnvV3Gaussian(gym.Env):
        def __init__(self):
            super().__init__()
            self.resonator = QuantumResonatorSim()
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            low  = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
            high = np.array([1.0, 1.2, 1.2,  1.0], dtype=np.float32)
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
            self.current_freq = 500e3
            self.max_steps = 200
            self.current_step = 0
            self.ema_alpha = 0.3
            self.ema_amp = 0.0
            self.prev_ema_amp = 0.0
            self.prev_action = 0.0
            self.low_amp_count = 0

        def set_drift_sigma(self, sigma):
            self.resonator.drift_sigma = sigma

        def _get_obs(self):
            norm_freq = np.clip((self.current_freq - 475e3) / 50e3, 0.0, 1.0)
            return np.array([norm_freq,
                             np.clip(self.ema_amp, 0.0, 1.2),
                             np.clip(self.prev_ema_amp, 0.0, 1.2),
                             np.clip(self.prev_action, -1.0, 1.0)], dtype=np.float32)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.resonator.reset()
            self.current_freq = 500e3
            self.current_step = 0
            self.low_amp_count = 0
            raw_amp = self.resonator.measure_amplitude(self.current_freq)
            self.ema_amp = raw_amp
            self.prev_ema_amp = raw_amp
            self.prev_action = 0.0
            return self._get_obs(), {
                "true_f0": self.resonator.f0_current,
                "current_freq": self.current_freq
            }

        def step(self, action):
            self.current_step += 1
            self.current_freq = np.clip(self.current_freq + action[0] * 3000, 475e3, 525e3)
            self.resonator.step_drift()
            raw_amp = self.resonator.measure_amplitude(self.current_freq)
            self.prev_ema_amp = self.ema_amp
            self.ema_amp = (self.ema_alpha * raw_amp) + ((1.0 - self.ema_alpha) * self.prev_ema_amp)
            gradient_bonus = 0.5 * (self.ema_amp - self.prev_ema_amp) * np.sign(action[0])
            reward = float(self.ema_amp + gradient_bonus - 0.01 * np.abs(action[0]))
            if 0.02 < self.ema_amp < 0.15:
                reward -= (0.15 - self.ema_amp) * 2.0
            if self.ema_amp < 0.02:
                self.low_amp_count += 1
            else:
                self.low_amp_count = 0
            terminated = bool(self.low_amp_count >= 3)
            if terminated:
                reward -= 5.0
            truncated = bool(self.current_step >= self.max_steps)
            self.prev_action = float(action[0])
            return self._get_obs(), reward, terminated, truncated, {
                "true_f0": self.resonator.f0_current,
                "current_freq": self.current_freq
            }

    return ResonatorEnvV3Gaussian()


def make_v3_env_ou():
    from gymnasium import spaces
    from simulation.resonator_model import QuantumResonatorSim

    class ResonatorEnvV3OU(gym.Env):
        def __init__(self):
            super().__init__()
            self.resonator = QuantumResonatorSim()
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            low  = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
            high = np.array([1.0, 1.2, 1.2,  1.0], dtype=np.float32)
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
            self.current_freq = 500e3
            self.max_steps = 200
            self.current_step = 0
            self.ema_alpha = 0.3
            self.ema_amp = 0.0
            self.prev_ema_amp = 0.0
            self.prev_action = 0.0
            self.low_amp_count = 0

        def set_drift_sigma(self, sigma):
            self.resonator.drift_sigma = sigma

        def _get_obs(self):
            norm_freq = np.clip((self.current_freq - 475e3) / 50e3, 0.0, 1.0)
            return np.array([norm_freq,
                             np.clip(self.ema_amp, 0.0, 1.2),
                             np.clip(self.prev_ema_amp, 0.0, 1.2),
                             np.clip(self.prev_action, -1.0, 1.0)], dtype=np.float32)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.resonator.reset()
            self.current_freq = 500e3
            self.current_step = 0
            self.low_amp_count = 0
            raw_amp = self.resonator.measure_amplitude(self.current_freq)
            self.ema_amp = raw_amp
            self.prev_ema_amp = raw_amp
            self.prev_action = 0.0
            return self._get_obs(), {
                "true_f0": self.resonator.f0_current,
                "current_freq": self.current_freq
            }

        def step(self, action):
            self.current_step += 1
            self.current_freq = np.clip(self.current_freq + action[0] * 3000, 475e3, 525e3)
            self.resonator.step_drift()
            raw_amp = self.resonator.measure_amplitude(self.current_freq)
            self.prev_ema_amp = self.ema_amp
            self.ema_amp = (self.ema_alpha * raw_amp) + ((1.0 - self.ema_alpha) * self.prev_ema_amp)
            gradient_bonus = 0.5 * (self.ema_amp - self.prev_ema_amp) * np.sign(action[0])
            reward = float(self.ema_amp + gradient_bonus - 0.01 * np.abs(action[0]))
            if 0.02 < self.ema_amp < 0.15:
                reward -= (0.15 - self.ema_amp) * 2.0
            if self.ema_amp < 0.02:
                self.low_amp_count += 1
            else:
                self.low_amp_count = 0
            terminated = bool(self.low_amp_count >= 3)
            if terminated:
                reward -= 5.0
            truncated = bool(self.current_step >= self.max_steps)
            self.prev_action = float(action[0])
            return self._get_obs(), reward, terminated, truncated, {
                "true_f0": self.resonator.f0_current,
                "current_freq": self.current_freq
            }

    return ResonatorEnvV3OU()


def make_v4_env_gaussian():
    from simulation.resonator_model_gaussian import QuantumResonatorSim
    from rl_training.rl_environment import ResonatorEnv

    class ResonatorEnvV4Gaussian(ResonatorEnv):
        def __init__(self):
            super().__init__()
            self.resonator = QuantumResonatorSim()

        def reset(self, seed=None, options=None):
            obs, info = super().reset(seed=seed, options=options)
            info["current_freq"] = self.current_freq
            return obs, info

        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            info["current_freq"] = self.current_freq
            return obs, reward, terminated, truncated, info

    return ResonatorEnvV4Gaussian()


def make_v4_env_ou():
    from simulation.resonator_model import QuantumResonatorSim
    from rl_training.rl_environment import ResonatorEnv

    class ResonatorEnvV4OU(ResonatorEnv):
        def __init__(self):
            super().__init__()
            self.resonator = QuantumResonatorSim()

        def reset(self, seed=None, options=None):
            obs, info = super().reset(seed=seed, options=options)
            info["current_freq"] = self.current_freq
            return obs, info

        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            info["current_freq"] = self.current_freq
            return obs, reward, terminated, truncated, info

    return ResonatorEnvV4OU()


def load_model_with_env(model_dir, env_factory):
    env = DummyVecEnv([env_factory])
    env = VecNormalize.load(os.path.join(model_dir, "vec_normalize.pkl"), env)
    env.training = False
    env.norm_reward = False
    model = PPO.load(os.path.join(model_dir, "best_model.zip"), env=env, device="cuda")
    return model, env


def collect_mae(model, env, num_episodes=NUM_EPISODES):
    maes = []
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        ep_maes = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = env.step(action)
            true_f0 = infos[0]["true_f0"]
            current_freq = infos[0]["current_freq"]
            ep_maes.append(abs(current_freq - true_f0))
            done = dones[0]
        maes.append(np.mean(ep_maes))
    return np.array(maes)


def run_experiment(label, v3_dir, v4_version, v4_seeds, v3_env_factory, v4_env_factory):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {label}")
    print(f"{'='*60}")

    print(f"Collecting V3 MAE over {NUM_EPISODES} episodes...")
    model_v3, env_v3 = load_model_with_env(v3_dir, v3_env_factory)
    mae_v3 = collect_mae(model_v3, env_v3)
    env_v3.close()
    print(f"V3 MAE: {np.mean(mae_v3):.1f} ± {np.std(mae_v3):.1f} Hz")

    all_v4_mae = []
    for seed in v4_seeds:
        model_dir = os.path.join("rl_training", "trained_models", v4_version, f"seed_{seed}")
        print(f"Collecting V4 seed {seed} MAE over {NUM_EPISODES} episodes...")
        model_v4, env_v4 = load_model_with_env(model_dir, v4_env_factory)
        mae_seed = collect_mae(model_v4, env_v4)
        env_v4.close()
        print(f"  Seed {seed} MAE: {np.mean(mae_seed):.1f} ± {np.std(mae_seed):.1f} Hz")
        all_v4_mae.append(mae_seed)

    mae_v4 = np.concatenate(all_v4_mae)
    print(f"V4 aggregate MAE: {np.mean(mae_v4):.1f} ± {np.std(mae_v4):.1f} Hz")

    stat, p = mannwhitneyu(mae_v3, mae_v4, alternative='greater')
    print(f"\nMann-Whitney U test (V3 MAE > V4 MAE):")
    print(f"  U statistic: {stat:.1f}")
    print(f"  p-value:     {p:.6f}")
    if p < 0.05:
        print(f"  Result: SIGNIFICANT (p < 0.05) — V4 significantly reduces MAE vs V3")
    else:
        print(f"  Result: NOT SIGNIFICANT (p >= 0.05)")

    return {
        "mae_v3_mean": float(np.mean(mae_v3)),
        "mae_v3_std": float(np.std(mae_v3)),
        "mae_v4_mean": float(np.mean(mae_v4)),
        "mae_v4_std": float(np.std(mae_v4)),
        "U": float(stat),
        "p_value": float(p),
        "significant": bool(p < 0.05)
    }


if __name__ == "__main__":
    import json

    from rl_training.rl_environment import ResonatorEnv
    e = ResonatorEnv()
    o, _ = e.reset()
    assert len(o) == 5, f"Expected 5-element obs, got {len(o)}. Check rl_environment.py."
    print(f"Environment check passed: obs length = {len(o)}")

    results = {}

    results["gaussian"] = run_experiment(
        label="Gaussian Noise: V3 vs V4",
        v3_dir=os.path.join("rl_training", "trained_models", "v3_refined"),
        v4_version="v4_gradient_obs",
        v4_seeds=[0, 1, 2, 3, 4],
        v3_env_factory=make_v3_env_gaussian,
        v4_env_factory=make_v4_env_gaussian
    )

    results["ou"] = run_experiment(
        label="OU Noise: V3 vs V4",
        v3_dir=os.path.join("rl_training", "trained_models", "v3_ou_noise", "seed_0"),
        v4_version="v4_ou_noise",
        v4_seeds=[0, 1, 2],
        v3_env_factory=make_v3_env_ou,
        v4_env_factory=make_v4_env_ou
    )

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for regime, r in results.items():
        print(f"\n{regime.upper()}:")
        print(f"  V3 MAE: {r['mae_v3_mean']:.1f} ± {r['mae_v3_std']:.1f} Hz")
        print(f"  V4 MAE: {r['mae_v4_mean']:.1f} ± {r['mae_v4_std']:.1f} Hz")
        print(f"  p-value: {r['p_value']:.6f} — {'SIGNIFICANT' if r['significant'] else 'NOT SIGNIFICANT'}")

    out_path = os.path.join("data_logs", "statistical_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {out_path}")