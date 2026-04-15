import matplotlib.pyplot as plt
import numpy as np
from qiskit_integration.qiskit_resonator_env import QiskitResonatorEnv

print("Initializing Qiskit Sweep...")
env = QiskitResonatorEnv(shots=4096)
env.f0_current = 500000.0 # Force resonance to center for the test

frequencies = np.linspace(480000, 520000, 100) # 480kHz to 520kHz
amplitudes = []

print("Running quantum frequency sweep (100 points)...")
for f in frequencies:
    amp = env._get_quantum_amplitude(f)
    amplitudes.append(amp)

plt.figure(figsize=(10, 6))
plt.plot(frequencies/1000, amplitudes, 'b-', label='Qiskit P(|1>)')
plt.axvline(x=500, color='r', linestyle='--', label='f0 (Resonance)')
plt.title("Quantum Resonator Verification: Rabi-Lorentzian Peak")
plt.xlabel("Frequency (kHz)")
plt.ylabel("Excitation Probability P(|1>)")
plt.grid(True)
plt.legend()
plt.savefig("qiskit_sweep_verification.png")
print("Verification complete. Check 'qiskit_sweep_verification.png'.")
plt.show()