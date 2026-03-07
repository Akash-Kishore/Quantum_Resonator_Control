import numpy as np

class QuantumResonatorSim:
    """
    Simulates a drifting classical resonator.
    Implements Lorentzian amplitude response and stochastic frequency drift.
    """
    def __init__(self, f0=500e3, q_factor=50, drift_sigma=500, noise_floor=0.02):
        self.f0_nominal = f0
        self.f0_current = f0
        self.q_factor = q_factor
        self.drift_sigma = drift_sigma
        self.noise_floor = noise_floor
        
    def _lorentzian_response(self, f):
        return 1.0 / (1.0 + (self.q_factor**2) * ((f - self.f0_current) / self.f0_current)**2)

    def step_drift(self):
        drift = np.random.normal(0, self.drift_sigma)
        self.f0_current += drift
        limit = 0.05 * self.f0_nominal
        self.f0_current = np.clip(self.f0_current, 
                                 self.f0_nominal - limit, 
                                 self.f0_nominal + limit)

    def measure_amplitude(self, drive_frequency):
        true_amp = self._lorentzian_response(drive_frequency)
        measured_amp = true_amp + np.random.normal(0, self.noise_floor)
        return np.clip(measured_amp, 0.0, 1.2)

    def reset(self):
        self.f0_current = self.f0_nominal
        return self.f0_current