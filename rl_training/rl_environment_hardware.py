import time
import serial
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from simulation.resonator_model import QuantumResonatorSim

SERIAL_PORT = "COM3"
BAUD_RATE = 921600
SERIAL_TIMEOUT = 2.0
SETTLE_DELAY = 0.05

class ResonatorEnvHardware(gym.Env):
    def __init__(self, dry_run=True):
        super().__init__()
        
        self.dry_run = dry_run
        
        if dry_run is True:
            self.resonator = QuantumResonatorSim()
            self.ser = None
        if dry_run is False:
            self.resonator = None
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT)
            print("Hardware serial connection established")
            
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        low = np.array([0.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32)
        high = np.array([1.0, 1.2, 1.2, 1.0, 1.0], dtype=np.float32)
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

    def _get_hardware_amplitude(self, freq):
        if self.dry_run is True:
            return self.resonator.measure_amplitude(freq)
        if self.dry_run is False:
            self.ser.write(f"SET_FREQ {int(freq)}\n".encode('utf-8'))
            time.sleep(SETTLE_DELAY)
            self.ser.write("MEASURE\n".encode('utf-8'))
            line = self.ser.readline().decode('utf-8').strip()
            try:
                amplitude = float(line.split(' ')[1])
            except Exception:
                amplitude = 0.0
            return amplitude

    def _step_hardware_drift(self):
        if self.dry_run is True:
            self.resonator.step_drift()
        if self.dry_run is False:
            pass

    def set_drift_sigma(self, sigma):
        if self.dry_run is True:
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
        
        if self.dry_run is True:
            self.resonator.reset()
            
        self.current_freq = 500e3
        self.prev_freq = 500e3
        self.amp_gradient = 0.0
        self.current_step = 0
        self.low_amp_count = 0
        self.prev_action = 0.0
        
        raw_amp = self._get_hardware_amplitude(self.current_freq)
        self.ema_amp = raw_amp
        self.prev_ema_amp = raw_amp
        
        info = {}
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
        
        self._step_hardware_drift()
        raw_amp = self._get_hardware_amplitude(self.current_freq)
        
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
        obs = self._get_obs()
        info = {}
        
        return obs, reward, terminated, truncated, info

    def close(self):
        if self.ser is not None:
            self.ser.close()
            print("Serial connection closed")

if __name__ == "__main__":
    print("Testing ResonatorEnvHardware in dry run mode...")
    env = ResonatorEnvHardware(dry_run=True)
    obs, _ = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print("Dry run environment initialised successfully")