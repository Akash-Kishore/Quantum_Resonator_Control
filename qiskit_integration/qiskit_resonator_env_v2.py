# qiskit_resonator_env_v2.py
# CANONICAL version — no hacks, no engineering multipliers
# All parameters match V4 training exactly
import gymnasium as gym
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

class QiskitResonatorEnvV2(gym.Env):
    def __init__(
        self,
        f0_center   = 500000.0,  # Hz — matches training
        f_range     =  25000.0,  # Hz half-range — matches training
        drift_sigma =    500.0,  # Hz/step — matches training
        action_scale=   3000.0,  # Hz/unit — matches training
        ema_alpha   =      0.3,  # — matches training
        omega_rabi  =   5000.0,  # Hz — calibrated to Q=50
        shots       =   1024,    # quantum measurements per step
        seed        =     42
    ):
        super().__init__()
        self.f0_center    = f0_center
        self.f_range      = f_range
        self.drift_sigma  = drift_sigma
        self.action_scale = action_scale
        self.ema_alpha    = ema_alpha
        self.omega_rabi   = omega_rabi
        self.shots        = shots
        self.rng          = np.random.default_rng(seed)

        self.simulator    = AerSimulator()

        # Internal state
        self.f0_current   = f0_center   # hidden qubit transition freq
        self.f_probe      = f0_center   # agent's current probe freq
        self.prev_amp     = 0.0
        self.prev_action  = 0.0         # normalised, in [-1,1]
        self.ema_amp      = 0.0
        self.delta_amp    = 0.0
        self.step_count   = 0
        self.max_steps    = 200

        # Observation and action spaces — IDENTICAL to V4
        obs_lo = np.array([-1,-1,-1,-2,-10], dtype=np.float32)
        obs_hi = np.array([ 2, 1, 1, 2, 10], dtype=np.float32)
        self.observation_space = gym.spaces.Box(obs_lo, obs_hi, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def _quantum_amplitude(self, f_probe_hz):
        """
        Rabi physics mapping — correct formula from Section 2 of
        the integration guide. Returns EMA-filtered P(|1>).
        """
        delta = f_probe_hz - self.f0_current
        theta = float(np.pi * self.omega_rabi /
                      np.sqrt(self.omega_rabi**2 + delta**2))

        qc = QuantumCircuit(1, 1)
        qc.rx(theta, 0)
        qc.measure(0, 0)

        result = self.simulator.run(qc, shots=self.shots).result()
        counts = result.get_counts()
        raw_p1 = counts.get('1', 0) / self.shots

        # EMA filter — alpha=0.3 matches V4 training
        self.ema_amp = (self.ema_alpha * raw_p1 +
                        (1 - self.ema_alpha) * self.ema_amp)
        return float(self.ema_amp)

    def _inject_drift(self):
        """
        Gaussian drift on f0_current. Called INSIDE step().
        This is the fix for Bug 2 — drift must be internal.
        """
        shift = float(self.rng.normal(0.0, self.drift_sigma))
        self.f0_current = float(np.clip(
            self.f0_current + shift,
            self.f0_center - self.f_range,
            self.f0_center + self.f_range
        ))

    def _build_obs(self, amplitude):
        """
        5-element observation — IDENTICAL formula to V4 rl_environment.py.
        Do not modify any element without also modifying the training env.
        """
        amp = float(amplitude)

        # Element 1: frequency error normalised to [-1, 1]
        freq_err_norm = float(
            (self.f_probe - self.f0_center) / self.f_range
        )

        # Element 2: previous normalised action
        prev_a = float(self.prev_action)

        # Element 3: delta_amp
        self.delta_amp = float(amp - self.prev_amp)

        # Element 4: amp_gradient — V4's key contribution
        # IDENTICAL formula to training. Do not rescale.
        eps = 1e-6
        amp_grad = float(np.clip(
            self.delta_amp / (abs(self.prev_action) + eps),
            -10.0, 10.0
        ))

        return np.array(
            [amp, freq_err_norm, prev_a, self.delta_amp, amp_grad],
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start qubit frequency near centre with small random offset
        self.f0_current = float(self.rng.normal(
            self.f0_center, self.f_range * 0.1
        ))
        self.f0_current = float(np.clip(
            self.f0_current,
            self.f0_center - self.f_range,
            self.f0_center + self.f_range
        ))
        self.f_probe      = float(self.f0_center)
        self.prev_amp     = 0.0
        self.prev_action  = 0.0
        self.ema_amp      = 0.0
        self.delta_amp    = 0.0
        self.step_count   = 0

        amp = self._quantum_amplitude(self.f_probe)
        self.prev_amp = amp
        obs = self._build_obs(amp)
        return obs, {}

    def step(self, action):
        # 1. Apply raw action — NO slew limiter
        action_val = float(np.clip(action[0], -1.0, 1.0))

        # 2. Update probe frequency
        delta_freq = action_val * self.action_scale
        self.f_probe = float(np.clip(
            self.f_probe + delta_freq,
            self.f0_center - self.f_range,
            self.f0_center + self.f_range
        ))

        # 3. Inject drift (INSIDE step — Bug 2 fix)
        self._inject_drift()

        # 4. Measure qubit excitation
        amp = self._quantum_amplitude(self.f_probe)

        # 5. Build observation
        obs = self._build_obs(amp)

        # 6. Reward — amplitude only, identical to V4
        reward = float(amp)

        # 7. Update state
        self.prev_amp    = amp
        self.prev_action = action_val
        self.step_count += 1

        terminated = self.step_count >= self.max_steps
        truncated  = False
        info = {
            'f_probe':       self.f_probe,
            'f_qubit':       self.f0_current,
            'freq_error_hz': abs(self.f_probe - self.f0_current),
            'amplitude':     amp
        }
        return obs, reward, terminated, truncated, info