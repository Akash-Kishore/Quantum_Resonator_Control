import os
import time
import serial
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulation.resonator_model import QuantumResonatorSim

SERIAL_PORT = "COM3"
BAUD_RATE = 921600
FREQ_START = 475000
FREQ_END = 525000
FREQ_STEP = 500
SERIAL_TIMEOUT = 2.0
SETTLE_DELAY = 0.05

def run_sweep(dry_run=True):
    frequencies = []
    amplitudes = []
    
    if dry_run:
        sim = QuantumResonatorSim()
        for freq in np.arange(FREQ_START, FREQ_END + FREQ_STEP, FREQ_STEP):
            amplitude = sim.measure_amplitude(freq)
            frequencies.append(freq)
            amplitudes.append(amplitude)
        return frequencies, amplitudes
        
    if not dry_run:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT)
        print("Serial connection established")
        for freq in np.arange(FREQ_START, FREQ_END + FREQ_STEP, FREQ_STEP):
            ser.write(f"SET_FREQ {int(freq)}\n".encode('utf-8'))
            time.sleep(SETTLE_DELAY)
            ser.write("MEASURE\n".encode('utf-8'))
            line = ser.readline().decode('utf-8').strip()
            try:
                amplitude = float(line.split(' ')[1])
            except Exception:
                amplitude = 0.0
            frequencies.append(freq)
            amplitudes.append(amplitude)
        ser.close()
        return frequencies, amplitudes

def plot_sweep(frequencies, amplitudes, dry_run):
    frequencies = np.array(frequencies)
    amplitudes = np.array(amplitudes)
    
    peak_amplitude = np.max(amplitudes)
    peak_freq = frequencies[np.argmax(amplitudes)]
    
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies, amplitudes, color="blue", linewidth=1.5)
    plt.axvline(x=peak_freq, color="red", linestyle="--", label=f"Peak: {int(peak_freq)} Hz")
    
    plt.xlabel("Drive Frequency (Hz)")
    plt.ylabel("Measured Amplitude (V)")
    
    if dry_run:
        plt.title("Frequency Sweep — Simulation (Dry Run)")
    if not dry_run:
        plt.title("Frequency Sweep — Real Hardware")
        
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    
    os.makedirs("data_logs", exist_ok=True)
    
    if dry_run:
        plt.savefig("data_logs/frequency_sweep_dryrun.png")
        print("Dry run sweep plot saved to data_logs/frequency_sweep_dryrun.png")
    if not dry_run:
        plt.savefig("data_logs/frequency_sweep_hardware.png")
        print("Hardware sweep plot saved to data_logs/frequency_sweep_hardware.png")
        
    print(f"Peak amplitude: {peak_amplitude:.4f} at {int(peak_freq)} Hz")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware", action="store_true")
    args = parser.parse_args()
    
    dry_run = not args.hardware
    
    if dry_run:
        print("Running frequency sweep in dry run mode (simulation)...")
    else:
        print("Running frequency sweep on real hardware...")
        
    frequencies, amplitudes = run_sweep(dry_run=dry_run)
    plot_sweep(frequencies, amplitudes, dry_run=dry_run)