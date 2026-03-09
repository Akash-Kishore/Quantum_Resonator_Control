import gymnasium as gym
from gymnasium import spaces
import numpy as np
from simulation.resonator_model import QuantumResonatorSim

class ResonatorEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    Controls a simulated 500 kHz LC resonator with stochastic drift.
    """
    def __init__(self):
        super(ResonatorEnv, self).__init__()
        self.resonator = QuantumResonatorSim()
        
        # Action space: continuous frequency shift bounded between -1.0 and 1.0 (scales to +/- 5000 Hz)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # F-005 Fix: Tightened observation space bounds to reflect actual physical variable ranges.
        # [normalized_freq, ema_amplitude, prev_ema_amplitude, prev_action]
        low  = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        high = np.array([1.0, 1.2, 1.2,  1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        self.current_freq = 500e3
        self.max_steps = 200
        self.current_step = 0
        
        # F-002 Fix: Introduce EMA filter variables to improve Gradient SNR
        self.ema_alpha = 0.3 # Settling time approx 3.3 steps (much faster than drift)
        self.ema_amp = 0.0
        self.prev_ema_amp = 0.0
        self.prev_action = 0.0
        
        # F-004 Fix: Counter for consecutive low-amplitude measurements
        self.low_amp_count = 0

    def set_drift_sigma(self, sigma):
        """Allows external curriculum callbacks to adjust environment difficulty."""
        self.resonator.drift_sigma = sigma

    def _get_obs(self):
        """Constructs and strictly bounds the state vector."""
        norm_freq = (self.current_freq - 475e3) / 50e3
        
        # Safety clip to guarantee observations never exceed declared bounds
        norm_freq = np.clip(norm_freq, 0.0, 1.0)
        safe_ema = np.clip(self.ema_amp, 0.0, 1.2)
        safe_prev_ema = np.clip(self.prev_ema_amp, 0.0, 1.2)
        safe_prev_action = np.clip(self.prev_action, -1.0, 1.0)
        
        return np.array([norm_freq, safe_ema, safe_prev_ema, safe_prev_action], dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Resets the environment to initial conditions for a new episode."""
        super().reset(seed=seed)
        self.resonator.reset()
        self.current_freq = 500e3
        self.current_step = 0
        self.low_amp_count = 0
        
        # Initialize EMA filter with the first raw measurement (no historical lag)
        raw_amp = self.resonator.measure_amplitude(self.current_freq)
        self.ema_amp = raw_amp
        self.prev_ema_amp = raw_amp
        self.prev_action = 0.0
        
        info = {"true_f0": self.resonator.f0_current}
        return self._get_obs(), info

    def step(self, action):
        """Executes one time step within the environment."""
        self.current_step += 1
        
        # Apply action to shift frequency (+/- 5000 Hz max)
        freq_shift = action[0] * 5000 
        self.current_freq += freq_shift
        self.current_freq = np.clip(self.current_freq, 475e3, 525e3)
        
        # Progress the physical simulation
        self.resonator.step_drift()
        raw_amp = self.resonator.measure_amplitude(self.current_freq)
        
        # F-002 Fix: Apply EMA filter to incoming amplitude
        self.prev_ema_amp = self.ema_amp
        self.ema_amp = (self.ema_alpha * raw_amp) + ((1.0 - self.ema_alpha) * self.prev_ema_amp)
        
        # F-003 Fix: Reward function overhauled to include a gradient-climbing bonus.
        # This rewards actions that move amplitude in the expected direction given the Lorentzian curve,
        # explicitly breaking the static-frequency local optimum.
        gradient_bonus = 0.5 * (self.ema_amp - self.prev_ema_amp) * np.sign(action[0])
        reward = float(self.ema_amp + gradient_bonus - 0.01 * np.abs(action[0]))
        
        # F-004 Fix: Soft penalties for low amplitude rather than immediate termination.
        if 0.02 < self.ema_amp < 0.15:
            reward -= (0.15 - self.ema_amp) * 2.0
            
        # Manage hard termination for genuine loss-of-lock
        if self.ema_amp < 0.02:
            self.low_amp_count += 1
        else:
            self.low_amp_count = 0
            
        # Terminate only after 3 consecutive dead measurements, applying a strong penalty
        terminated = bool(self.low_amp_count >= 3)
        if terminated:
            reward -= 5.0 
            
        truncated = bool(self.current_step >= self.max_steps)
        
        self.prev_action = float(action[0])
        obs = self._get_obs()
        info = {"true_f0": self.resonator.f0_current}
        
        return obs, reward, terminated, truncated, info

if __name__ == "__main__":
    # Simple initialization check
    env = ResonatorEnv()
    obs, _ = env.reset()
    print(f"Environment initialized successfully. Initial state: {obs}")