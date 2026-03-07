import numpy as np
from stable_baselines3.common.env_checker import check_env
from rl_training.rl_environment import ResonatorEnv

def verify():
    env = ResonatorEnv()
    print("Initiating API compliance check...")
    # This will now pass smoothly because F-005 fixed the observation bounds
    check_env(env)
    print("Environment API check passed.\n")

    print("=== Gradient SNR Empirical Test ===")
    print(f"Noise Floor Sigma: {env.unwrapped.resonator.noise_floor}")
    
    offsets = [500, 1000, 2000, 5000]
    drift_step = 500 # 1 sigma drift step
    
    print(f"{'Offset (Hz)':<15} | {'Raw SNR':<10} | {'EMA SNR':<10}")
    print("-" * 42)
    
    for offset in offsets:
        # Set up environment at specific offset
        env.reset()
        f_base = 500e3
        env.unwrapped.resonator.f0_current = f_base
        env.unwrapped.current_freq = f_base + offset
        
        raw_amps = []
        ema_amps = []
        
        # Take 100 samples at current offset
        for _ in range(100):
            raw = env.unwrapped.resonator.measure_amplitude(env.unwrapped.current_freq)
            env.unwrapped.prev_ema_amp = env.unwrapped.ema_amp
            env.unwrapped.ema_amp = (env.unwrapped.ema_alpha * raw) + ((1.0 - env.unwrapped.ema_alpha) * env.unwrapped.prev_ema_amp)
            raw_amps.append(raw)
            ema_amps.append(env.unwrapped.ema_amp)
            
        # Shift frequency by one drift step to simulate gradient
        env.unwrapped.current_freq = f_base + offset + drift_step
        
        raw_amps_shifted = []
        ema_amps_shifted = []
        
        # Take 100 samples at shifted offset
        for _ in range(100):
            raw = env.unwrapped.resonator.measure_amplitude(env.unwrapped.current_freq)
            env.unwrapped.prev_ema_amp = env.unwrapped.ema_amp
            env.unwrapped.ema_amp = (env.unwrapped.ema_alpha * raw) + ((1.0 - env.unwrapped.ema_alpha) * env.unwrapped.prev_ema_amp)
            raw_amps_shifted.append(raw)
            ema_amps_shifted.append(env.unwrapped.ema_amp)
            
        # Calculate gradients and noise
        delta_raw = np.mean(raw_amps_shifted) - np.mean(raw_amps)
        noise_raw = np.std(raw_amps) * np.sqrt(2) # Noise of difference
        snr_raw = abs(delta_raw) / noise_raw
        
        delta_ema = np.mean(ema_amps_shifted) - np.mean(ema_amps)
        noise_ema = np.std(ema_amps) * np.sqrt(2)
        snr_ema = abs(delta_ema) / noise_ema
        
        print(f"+{offset:<14} | {snr_raw:<10.3f} | {snr_ema:<10.3f}")

if __name__ == "__main__":
    verify()