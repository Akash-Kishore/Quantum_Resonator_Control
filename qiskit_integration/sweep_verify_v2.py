# sweep_verify_v2.py
import numpy as np, matplotlib.pyplot as plt
from qiskit_integration.qiskit_resonator_env_v2 import QiskitResonatorEnvV2

env = QiskitResonatorEnvV2(shots=2048)
env.f0_current = 500000.0

freqs = np.linspace(475000, 525000, 80)
amps  = []
for f in freqs:
    env.ema_amp = 0.0            # RESET between each point
    env.prev_amp = 0.0           # RESET between each point
    # Single raw measurement, no EMA carry-forward
    delta = f - env.f0_current
    theta = float(np.pi * env.omega_rabi /
                  np.sqrt(env.omega_rabi**2 + delta**2))
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(1,1)
    qc.rx(theta, 0)
    qc.measure(0, 0)
    r = env.simulator.run(qc, shots=2048).result()
    amps.append(r.get_counts().get('1',0)/2048)

plt.figure(figsize=(9,4))
plt.plot(freqs/1000, amps, 'b-', lw=2)
plt.axvline(500, color='r', ls='--', label='f0 = 500 kHz')
plt.axhline(0.5, color='grey', ls=':', label='P=0.5 (half-max)')
plt.xlabel('Frequency (kHz)')
plt.ylabel('P(|1>)')
plt.title('Qiskit V2 Lorentzian — should peak at 500 kHz, floor ~0.04')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('sweep_v2_verification.png', dpi=150)