import gymnasium as gym
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

class QiskitResonatorEnv(gym.Env):
    def __init__(self, shots=4096):
        super(QiskitResonatorEnv, self).__init__()
        self.simulator = AerSimulator()
        self.shots = shots
        
        # --- Physical Constants ---
        self.f0_nominal = 500000.0 
        self.omega_rabi = 10000.0  # Rabi frequency for curve width
        self.f0_current = self.f0_nominal
        
        # --- NEW "Smooth & Deep" Configuration ---
        # Change 1: High Dynamic Range (Reach: +/- 30kHz)
        self.multiplier = 30000.0  
        
        # Change 2: Heavy Slew-Rate Limiting (Action Damping)
        # beta=0.1 means movement is slow and smooth (critically damped).
        self.beta = 0.1           
        
        # Change 3: Enhanced Measurement Filter (EMA Factor)
        # alpha=0.5 smooths shot noise heavily.
        self.alpha = 0.5          
        
        # --- State Management ---
        self.prev_amp = 0.0
        self.prev_action_val = 0.0
        
        # Standard V4 Spaces
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

    def _get_quantum_amplitude(self, drive_freq):
        """Calculates heavily smoothed excited state population."""
        detuning = drive_freq - self.f0_current
        # Rabi physics Rx rotation angle mapping
        theta = np.pi * self.omega_rabi / np.sqrt(self.omega_rabi**2 + detuning**2)
        
        qc = QuantumCircuit(1, 1)
        qc.rx(theta, 0)
        qc.measure(0, 0)
        
        t_qc = transpile(qc, self.simulator)
        result = self.simulator.run(t_qc, shots=self.shots).result()
        raw_amp = result.get_counts().get('1', 0) / self.shots
        
        # Apply Exponential Moving Average (EMA) to measurement
        smoothed_amp = (self.alpha * raw_amp) + ((1 - self.alpha) * self.prev_amp)
        return smoothed_amp

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.f0_current = self.f0_nominal + np.random.uniform(-500, 500)
        self.prev_amp = self._get_quantum_amplitude(self.f0_nominal)
        self.prev_action_val = 0.0
        
        # Obs: [amplitude, freq_norm, prev_action, delta_amp, gradient]
        obs = np.array([self.prev_amp, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return obs, {}

    def step(self, action):
        # 1. Action Slew-Rate Limiter (Crucial Fix for Jaggedness)
        # Prevents the neural network's twitchy actions from making giant leaps.
        raw_action = np.clip(action[0], -1.0, 1.0)
        smoothed_action = (self.beta * raw_action) + ((1 - self.beta) * self.prev_action_val)
        
        # 2. Physics: Apply heavily smoothed frequency shift (30kHz reach)
        freq_step = smoothed_action * self.multiplier
        current_freq = self.f0_nominal + freq_step
        
        # 3. Measurement (EMA filtered to clean shot noise)
        new_amp = self._get_quantum_amplitude(current_freq)
        delta_amp = new_amp - self.prev_amp
        
        # 4. Decoupled Gradient (TRICK: Decoupling scaling factors)
        # We move at 30kHz but report the gradient to the agent with 5kHz sensitivity.
        # This matched the V4 training observation distribution.
        # Note: use abs(freq_step) in denominator to prevent gradient sign inversion.
        grad = np.clip((delta_amp * 5000.0) / (abs(freq_step) + 1.0), -1.0, 1.0)
        
        # 5. Observation Assembly (keeps normalization V4-expected range)
        obs = np.array([
            new_amp, 
            freq_step / 50000.0, 
            smoothed_action, 
            delta_amp, 
            grad
        ], dtype=np.float32)
        
        self.prev_amp = new_amp
        self.prev_action_val = smoothed_action
        
        # V4 Reward Logic
        reward = new_amp + 0.5 * delta_amp * np.sign(smoothed_action) - 0.01 * abs(smoothed_action)
        
        return obs, reward, False, False, {}