import gymnasium as gym
from gymnasium import spaces
import numpy as np
from simulation.resonator_model import QuantumResonatorSim

class ResonatorEnv(gym.Env):
    """
    Confound A: V4 observation (5-element) with V3 reward (identical to V4 reward in current code).
    """
    def __init__(self):
        super(ResonatorEnv, self).__init__()
        self.resonator = QuantumResonatorSim()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        low  = np.array([0.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32)
        high = np.array([1.0, 1.2, 1.2,  1.0,  1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.current_freq = 500e3
        self.prev_freq = 500e3
        self.amp_gradient = 0.0
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
        norm_freq = (self.current_freq - 475e3) / 50e3
        norm_freq = np.clip(norm_freq, 0.0, 1.0)
        safe_ema = np.clip(self.ema_amp, 0.0, 1.2)
        safe_prev_ema = np.clip(self.prev_ema_amp, 0.0, 1.2)
        safe_prev_action = np.clip(self.prev_action, -1.0, 1.0)
        safe_gradient = np.clip(self.amp_gradient, -1.0, 1.0)
        return np.array([norm_freq, safe_ema, safe_prev_ema, safe_prev_action, safe_gradient], dtype=np.float32)

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
        self.prev_freq = self.current_freq
        self.amp_gradient = 0.0
        info = {"true_f0": self.resonator.f0_current}
        return self._get_obs(), info

    def step(self, action):
        self.current_step += 1
        freq_shift = action[0] * 3000
        self.current_freq += freq_shift
        self.current_freq = np.clip(self.current_freq, 475e3, 525e3)
        delta_amp = self.ema_amp - self.prev_ema_amp
        freq_delta = abs(self.current_freq - self.prev_freq)
        self.amp_gradient = np.clip(delta_amp * 3000.0 / (freq_delta + 1.0), -1.0, 1.0)
        self.prev_freq = self.current_freq
        self.resonator.step_drift()
        raw_amp = self.resonator.measure_amplitude(self.current_freq)
        self.prev_ema_amp = self.ema_amp
        self.ema_amp = (self.ema_alpha * raw_amp) + ((1.0 - self.ema_alpha) * self.prev_ema_amp)
        # V3 reward (same as V4 currently)
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
        obs = self._get_obs()
        info = {"true_f0": self.resonator.f0_current}
        return obs, reward, terminated, truncated, info

if __name__ == "__main__":
    env = ResonatorEnv()
    obs, _ = env.reset()
    print(f"Confound A env initialized. obs length: {len(obs)}")