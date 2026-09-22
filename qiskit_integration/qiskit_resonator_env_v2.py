# qiskit_resonator_env_v2.py
# Aligned with classical V4 environment (rl_training/rl_environment.py)
# Observation: [norm_freq, ema_amp, prev_ema_amp, prev_action, amp_gradient]
# Reward matches classical V4 (gradient bonus, low-amp penalty, termination)
import gymnasium as gym
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


class QiskitResonatorEnvV2(gym.Env):
    """
    Qiskit‑AER environment whose API (obs, reward, action_scale, termination)
    is identical to the classical V4 environment in rl_training/rl_environment.py.
    """
    def __init__(
        self,
        f0_center=500000.0,
        f_range=25000.0,
        drift_sigma=500.0,
        action_scale=3000.0,          # match classical V4
        ema_alpha=0.3,
        omega_rabi=5000.0,            # widened so 0.90‑threshold ≈ classical
        shots=1024,
        seed=42,
    ):
        super().__init__()
        self.f0_center = f0_center
        self.f_range = f_range
        self.drift_sigma = drift_sigma
        self.action_scale = action_scale
        self.ema_alpha = ema_alpha
        self.omega_rabi = omega_rabi
        self.shots = shots
        self.rng = np.random.default_rng(seed)
        self.simulator = AerSimulator()

        # State variables
        self.f0_current = f0_center
        self.f_probe = f0_center
        self.prev_ema_amp = 0.0
        self.prev_action = 0.0
        self.ema_amp = 0.0
        self.amp_gradient = 0.0
        self.step_count = 0
        self.max_steps = 200
        self.low_amp_count = 0

        # Observation space identical to classical V4
        low = np.array([0.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32)
        high = np.array([1.0, 1.2, 1.2, 1.0, 1.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    # ---------- quantum amplitude (Rabi) ----------
    def _raw_p1(self, f_probe_hz):
        delta = f_probe_hz - self.f0_current
        theta = float(np.pi * self.omega_rabi / np.sqrt(self.omega_rabi**2 + delta**2))
        qc = QuantumCircuit(1, 1)
        qc.rx(theta, 0)
        qc.measure(0, 0)
        result = self.simulator.run(qc, shots=self.shots).result()
        return result.get_counts().get("1", 0) / self.shots

    def _quantum_amplitude(self, f_probe_hz):
        raw = self._raw_p1(f_probe_hz)
        self.ema_amp = self.ema_alpha * raw + (1 - self.ema_alpha) * self.ema_amp
        return float(self.ema_amp)

    # ---------- drift ----------
    def _inject_drift(self):
        shift = float(self.rng.normal(0.0, self.drift_sigma))
        self.f0_current = float(np.clip(
            self.f0_current + shift,
            self.f0_center - self.f_range,
            self.f0_center + self.f_range,
        ))

    # ---------- observation ----------
    def _build_obs(self):
        norm_freq = np.clip((self.f_probe - 475e3) / 50e3, 0.0, 1.0)
        safe_ema = np.clip(self.ema_amp, 0.0, 1.2)
        safe_prev_ema = np.clip(self.prev_ema_amp, 0.0, 1.2)
        safe_prev_action = np.clip(self.prev_action, -1.0, 1.0)
        safe_grad = np.clip(self.amp_gradient, -1.0, 1.0)
        return np.array([norm_freq, safe_ema, safe_prev_ema, safe_prev_action, safe_grad], dtype=np.float32)

    # ---------- gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # initialise f0 with small random offset
        self.f0_current = float(self.rng.normal(self.f0_center, self.f_range * 0.1))
        self.f0_current = float(np.clip(
            self.f0_current,
            self.f0_center - self.f_range,
            self.f0_center + self.f_range,
        ))
        self.f_probe = float(self.f0_center)
        self.prev_ema_amp = 0.0
        self.prev_action = 0.0
        self.ema_amp = 0.0
        self.amp_gradient = 0.0
        self.step_count = 0
        self.low_amp_count = 0

        # EMA warm‑up (3 measurements)
        for _ in range(3):
            self.ema_amp = self.ema_alpha * self._raw_p1(self.f_probe) + (1 - self.ema_alpha) * self.ema_amp
        self.prev_ema_amp = self.ema_amp

        return self._build_obs(), {}

    def step(self, action):
        action_val = float(np.clip(action[0], -1.0, 1.0))
        # frequency shift
        freq_shift = action_val * self.action_scale
        self.f_probe = float(np.clip(
            self.f_probe + freq_shift,
            self.f0_center - self.f_range,
            self.f0_center + self.f_range,
        ))
        self._inject_drift()

        # measure amplitude
        raw_amp = self._quantum_amplitude(self.f_probe)

        # gradient for observation
        delta_amp = self.ema_amp - self.prev_ema_amp
        freq_delta = abs(self.f_probe - getattr(self, "prev_f_probe", self.f_probe))
        self.amp_gradient = np.clip(delta_amp * 3000.0 / (freq_delta + 1.0), -1.0, 1.0)
        self.prev_f_probe = self.f_probe

        # ----- reward (identical to classical V4) -----
        gradient_bonus = 0.5 * (self.ema_amp - self.prev_ema_amp) * np.sign(action_val)
        reward = float(self.ema_amp + gradient_bonus - 0.01 * abs(action_val))
        if 0.02 < self.ema_amp < 0.15:
            reward -= (0.15 - self.ema_amp) * 2.0

        # low‑amp termination
        if self.ema_amp < 0.02:
            self.low_amp_count += 1
        else:
            self.low_amp_count = 0
        terminated = bool(self.low_amp_count >= 3)
        if terminated:
            reward -= 5.0

        truncated = bool(self.step_count >= self.max_steps)
        self.step_count += 1

        # update prev values
        self.prev_ema_amp = self.ema_amp
        self.prev_action = action_val

        info = {
            "f_probe": self.f_probe,
            "f_qubit": self.f0_current,
            "freq_error_hz": abs(self.f_probe - self.f0_current),
            "amplitude": self.ema_amp,
        }
        return self._build_obs(), reward, terminated, truncated, info