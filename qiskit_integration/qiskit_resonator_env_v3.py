# qiskit_resonator_env_v3.py
# V3 CHANGES vs V2:
#   1. action_scale reduced from 3000 to 1000 Hz  (oscillation fix)
#   2. omega_rabi reduced from 5000 to 2500 Hz    (gradient SNR fix)
#   3. EMA warmup added to reset()                (bias fix)
#   4. Default shots raised from 1024 to 1024     (unchanged, fine-tuning uses 1024)
import gymnasium as gym
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


class QiskitResonatorEnvV3(gym.Env):
    # Updated parameters for FINAL FIX in qiskit_resonator_env_v3.py
    def __init__(
        self,
        f0_center=500000.0,
        f_range=25000.0,
        drift_sigma=500.0,
        action_scale=1000.0,
        ema_alpha=0.3,
        omega_rabi=2500.0,
        shots=1024,
        seed=42,
    ):
        # ... rest of init remains the same ...
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
        self.f0_current = f0_center
        self.f_probe = f0_center
        self.prev_amp = 0.0
        self.prev_action = 0.0
        self.ema_amp = 0.0
        self.delta_amp = 0.0
        self.step_count = 0
        self.max_steps = 200

        obs_lo = np.array([-1, -1, -1, -2, -10], dtype=np.float32)
        obs_hi = np.array([2, 1, 1, 2, 10], dtype=np.float32)
        self.observation_space = gym.spaces.Box(obs_lo, obs_hi, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

    def _quantum_amplitude(self, f_probe_hz):
        delta = f_probe_hz - self.f0_current
        theta = float(np.pi * self.omega_rabi / np.sqrt(self.omega_rabi**2 + delta**2))
        qc = QuantumCircuit(1, 1)
        qc.rx(theta, 0)
        qc.measure(0, 0)
        result = self.simulator.run(qc, shots=self.shots).result()
        counts = result.get_counts()
        raw_p1 = counts.get("1", 0) / self.shots
        self.ema_amp = self.ema_alpha * raw_p1 + (1 - self.ema_alpha) * self.ema_amp
        return float(self.ema_amp)

    def _inject_drift(self):
        shift = float(self.rng.normal(0.0, self.drift_sigma))
        self.f0_current = float(
            np.clip(
                self.f0_current + shift,
                self.f0_center - self.f_range,
                self.f0_center + self.f_range,
            )
        )

    def _build_obs(self, amplitude):
        amp = float(amplitude)
        freq_err_norm = float((self.f_probe - self.f0_center) / self.f_range)
        prev_a = float(self.prev_action)
        self.delta_amp = float(amp - self.prev_amp)
        eps = 1e-6
        amp_grad = float(
            np.clip(self.delta_amp / (abs(self.prev_action) + eps), -10.0, 10.0)
        )
        return np.array(
            [amp, freq_err_norm, prev_a, self.delta_amp, amp_grad], dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.f0_current = float(self.rng.normal(self.f0_center, self.f_range * 0.1))
        self.f0_current = float(
            np.clip(
                self.f0_current,
                self.f0_center - self.f_range,
                self.f0_center + self.f_range,
            )
        )
        self.f_probe = float(self.f0_center)
        self.prev_amp = 0.0
        self.prev_action = 0.0
        self.ema_amp = 0.0
        self.delta_amp = 0.0
        self.step_count = 0

        # EMA warmup — take 3 measurements without building observations
        # This primes ema_amp to a realistic value before the policy's
        # first step, eliminating the systematic low-amplitude bias at t=0
        for _ in range(3):
            self.ema_amp = (
                self.ema_alpha * self._raw_p1(self.f_probe)
                + (1 - self.ema_alpha) * self.ema_amp
            )

        amp = self._quantum_amplitude(self.f_probe)
        self.prev_amp = amp
        obs = self._build_obs(amp)
        return obs, {}

    def _raw_p1(self, f_probe_hz):
        """Single raw measurement without updating ema_amp — used for warmup only."""
        delta = f_probe_hz - self.f0_current
        theta = float(np.pi * self.omega_rabi / np.sqrt(self.omega_rabi**2 + delta**2))
        qc = QuantumCircuit(1, 1)
        qc.rx(theta, 0)
        qc.measure(0, 0)
        result = self.simulator.run(qc, shots=self.shots).result()
        return result.get_counts().get("1", 0) / self.shots

    def step(self, action):
        action_val = float(np.clip(action[0], -1.0, 1.0))
        delta_freq = action_val * self.action_scale
        self.f_probe = float(
            np.clip(
                self.f_probe + delta_freq,
                self.f0_center - self.f_range,
                self.f0_center + self.f_range,
            )
        )
        self._inject_drift()
        amp = self._quantum_amplitude(self.f_probe)
        obs = self._build_obs(amp)
        reward = float(amp)
        self.prev_amp = amp
        self.prev_action = action_val
        self.step_count += 1
        terminated = self.step_count >= self.max_steps
        truncated = False
        info = {
            "f_probe": self.f_probe,
            "f_qubit": self.f0_current,
            "freq_error_hz": abs(self.f_probe - self.f0_current),
            "amplitude": amp,
        }
        return obs, reward, terminated, truncated, info
