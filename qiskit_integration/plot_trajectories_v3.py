import json
import os
import matplotlib.pyplot as plt
import numpy as np

# PATHS
DATA_DIR = r'C:\MiniProject_Sem4\data_logs\qiskit_aer_simulation_2'
TRAJ_PATH = os.path.join(DATA_DIR, 'qft_v3_trajectories.json')
SAVE_PATH = os.path.join(DATA_DIR, 'v4_qft_v3_trajectory.png')

def plot_v3_results():
    if not os.path.exists(TRAJ_PATH):
        print(f"Error: Could not find {TRAJ_PATH}")
        return

    with open(TRAJ_PATH, 'r') as f:
        trajectories = json.load(f)

    # Plot the first episode from the evaluation
    ep = trajectories[0]
    steps = np.arange(len(ep['f_probe']))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Top Panel: Frequency Tracking
    ax1.plot(steps, np.array(ep['f_qubit'])/1000, label='Qubit Frequency (Red)', color='red', linewidth=2)
    ax1.plot(steps, np.array(ep['f_probe'])/1000, label='Agent Probe (Blue)', color='blue', alpha=0.7)
    ax1.set_ylabel('Frequency (kHz)')
    ax1.set_title('V4_QFT_V3: Frequency Tracking Trajectory (Task 4)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom Panel: Amplitude (Excitation)
    ax2.plot(steps, ep['amp'], label='EMA Amplitude', color='green')
    ax2.axhline(y=0.90, color='black', linestyle='--', alpha=0.5, label='Near-Resonance (0.90)')
    ax2.set_ylabel('Amplitude (P|1>)')
    ax2.set_xlabel('Step Number')
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(SAVE_PATH)
    print(f"Trajectory plot saved to: {SAVE_PATH}")
    plt.show()

if __name__ == "__main__":
    plot_v3_results()