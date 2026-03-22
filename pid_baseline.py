import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulation.resonator_model_gaussian import QuantumResonatorSim as GaussianSim
from simulation.resonator_model import QuantumResonatorSim as OUSim

NUM_EPISODES = 1000
MAX_STEPS = 200
AMP_TARGET = 0.95
FREQ_MIN = 475e3
FREQ_MAX = 525e3
ACTION_SCALE = 3000  # Hz — matches V3/V4 max step


class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def step(self, error):
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return float(np.clip(output, -1.0, 1.0))


def run_pid_evaluation(resonator_factory, kp, ki, kd, num_episodes=NUM_EPISODES, label=""):
    pid = PIDController(kp, ki, kd)

    all_mae = []
    all_near_resonance = []
    all_low_amp = []

    for _ in range(num_episodes):
        resonator = resonator_factory()
        resonator.reset()
        current_freq = 500e3
        ema_amp = resonator.measure_amplitude(current_freq)
        pid.reset()

        ep_mae = []
        ep_near = []
        ep_low = []

        for _ in range(MAX_STEPS):
            resonator.step_drift()
            raw_amp = resonator.measure_amplitude(current_freq)
            ema_amp = 0.3 * raw_amp + 0.7 * ema_amp

            error = AMP_TARGET - ema_amp
            action = pid.step(error)
            freq_shift = action * ACTION_SCALE
            current_freq = float(np.clip(current_freq + freq_shift, FREQ_MIN, FREQ_MAX))

            ep_mae.append(abs(current_freq - resonator.f0_current))
            ep_near.append(1.0 if ema_amp >= 0.90 else 0.0)
            ep_low.append(1.0 if ema_amp < 0.70 else 0.0)

        all_mae.append(np.mean(ep_mae))
        all_near_resonance.append(np.mean(ep_near) * 100)
        all_low_amp.append(np.mean(ep_low) * 100)

    mae_arr = np.array(all_mae)
    near_arr = np.array(all_near_resonance)
    low_arr = np.array(all_low_amp)

    mae_mean = round(float(np.mean(mae_arr)), 1)
    mae_std  = round(float(np.std(mae_arr)), 1)
    near_mean = round(float(np.mean(near_arr)), 1)
    near_std  = round(float(np.std(near_arr)), 1)
    low_mean  = round(float(np.mean(low_arr)), 1)
    low_std   = round(float(np.std(low_arr)), 1)

    print("\n=== PID Baseline | " + label + " | " + str(num_episodes) + " episodes ===")
    print("Kp=" + str(kp) + ", Ki=" + str(ki) + ", Kd=" + str(kd))
    print("Mean MAE:         " + str(mae_mean) + " +/- " + str(mae_std) + " Hz")
    print("Amplitude > 0.90: " + str(near_mean) + "% +/- " + str(near_std) + "%")
    print("Amplitude < 0.70: " + str(low_mean) + "% +/- " + str(low_std) + "%")

    return {
        "label": label,
        "mae_mean": mae_mean,
        "mae_std": mae_std,
        "near_mean": near_mean,
        "near_std": near_std,
        "low_mean": low_mean,
        "low_std": low_std,
        "mae_arr": mae_arr,
    }


def tune_pid(resonator_factory, label):
    """
    Grid search over Kp, Ki, Kd on 100 episodes to find best MAE.
    Returns best (kp, ki, kd).
    """
    print("\nTuning PID for " + label + "...")
    best_mae = float('inf')
    best_params = (0.5, 0.0, 0.1)

    kp_values = [0.1, 0.3, 0.5, 0.8, 1.0]
    ki_values = [0.0, 0.001, 0.005, 0.01]
    kd_values = [0.0, 0.05, 0.1, 0.2]

    for kp in kp_values:
        for ki in ki_values:
            for kd in kd_values:
                pid = PIDController(kp, ki, kd)
                episode_maes = []

                for _ in range(100):
                    resonator = resonator_factory()
                    resonator.reset()
                    current_freq = 500e3
                    ema_amp = resonator.measure_amplitude(current_freq)
                    pid.reset()
                    ep_mae = []

                    for _ in range(MAX_STEPS):
                        resonator.step_drift()
                        raw_amp = resonator.measure_amplitude(current_freq)
                        ema_amp = 0.3 * raw_amp + 0.7 * ema_amp
                        error = AMP_TARGET - ema_amp
                        action = pid.step(error)
                        current_freq = float(np.clip(
                            current_freq + action * ACTION_SCALE, FREQ_MIN, FREQ_MAX))
                        ep_mae.append(abs(current_freq - resonator.f0_current))

                    episode_maes.append(np.mean(ep_mae))

                mean_mae = np.mean(episode_maes)
                if mean_mae < best_mae:
                    best_mae = mean_mae
                    best_params = (kp, ki, kd)

    print("Best params: Kp=" + str(best_params[0]) +
          ", Ki=" + str(best_params[1]) +
          ", Kd=" + str(best_params[2]) +
          " | MAE=" + str(round(best_mae, 1)) + " Hz")
    return best_params


if __name__ == "__main__":
    import json

    results = {}

    # ── Gaussian ──────────────────────────────────────────────────────────
    kp_g, ki_g, kd_g = tune_pid(lambda: GaussianSim(), "Gaussian")
    results["gaussian"] = run_pid_evaluation(
        lambda: GaussianSim(), kp_g, ki_g, kd_g, label="Gaussian"
    )

    # ── OU ────────────────────────────────────────────────────────────────
    kp_ou, ki_ou, kd_ou = tune_pid(lambda: OUSim(), "OU Noise")
    results["ou"] = run_pid_evaluation(
        lambda: OUSim(), kp_ou, ki_ou, kd_ou, label="OU Noise"
    )

    # ── Summary table — all values come from results dict, no hardcoding ──
    rg  = results["gaussian"]
    rou = results["ou"]

    # Authoritative RL numbers — from evaluate_agent.py runs, never change
    v3g_mae   = "3333 +/- 1932"
    v3g_near  = "64.5%"
    v3g_low   = "13.4%"
    v4g_mae   = "2205 +/- 518"
    v4g_near  = "84.8% +/- 7.6%"
    v4g_low   = "5.1% +/- 3.7%"
    v3ou_mae  = "1241 +/- 254"
    v3ou_near = "97.7%"
    v3ou_low  = "0.0%"
    v4ou_mae  = "1284 +/- 93"
    v4ou_near = "97.9% +/- 0.4%"
    v4ou_low  = "0.0%"

    pid_g_mae  = str(rg["mae_mean"])  + " +/- " + str(rg["mae_std"])
    pid_g_near = str(rg["near_mean"]) + "%"
    pid_g_low  = str(rg["low_mean"])  + "%"
    pid_ou_mae  = str(rou["mae_mean"])  + " +/- " + str(rou["mae_std"])
    pid_ou_near = str(rou["near_mean"]) + "%"
    pid_ou_low  = str(rou["low_mean"])  + "%"

    col1 = 22
    col2 = 22
    col3 = 18

    print("\n\n=== SUMMARY vs RL BASELINES ===")
    print("Model".ljust(col1) + "MAE (Hz)".ljust(col2) +
          "Amp >0.90".ljust(col3) + "Amp <0.70")
    print("-" * 75)
    print("V3 Gaussian".ljust(col1) + v3g_mae.ljust(col2) +
          v3g_near.ljust(col3) + v3g_low)
    print("V4 Gaussian".ljust(col1) + v4g_mae.ljust(col2) +
          v4g_near.ljust(col3) + v4g_low)
    print("PID Gaussian".ljust(col1) + pid_g_mae.ljust(col2) +
          pid_g_near.ljust(col3) + pid_g_low)
    print()
    print("V3 OU".ljust(col1) + v3ou_mae.ljust(col2) +
          v3ou_near.ljust(col3) + v3ou_low)
    print("V4 OU".ljust(col1) + v4ou_mae.ljust(col2) +
          v4ou_near.ljust(col3) + v4ou_low)
    print("PID OU".ljust(col1) + pid_ou_mae.ljust(col2) +
          pid_ou_near.ljust(col3) + pid_ou_low)

    # ── Save results ──────────────────────────────────────────────────────
    save_data = {
        k: {kk: vv for kk, vv in v.items() if kk != "mae_arr"}
        for k, v in results.items()
    }
    save_data["best_pid_gaussian"] = {"kp": kp_g,  "ki": ki_g,  "kd": kd_g}
    save_data["best_pid_ou"]       = {"kp": kp_ou, "ki": ki_ou, "kd": kd_ou}

    os.makedirs("data_logs", exist_ok=True)
    with open("data_logs/pid_baseline_results.json", "w") as f:
        json.dump(save_data, f, indent=4)
    print("\nResults saved to data_logs/pid_baseline_results.json")