import sys
import torch
import gymnasium as gym
import stable_baselines3
import numpy
from rl_training.rl_environment import ResonatorEnv

def verify_gpu_pipeline():
    print("=== GPU Pipeline Diagnostic Check ===\n")
    
    # 1. Software Versions
    print("[Software Versions]")
    print(f"Python:            {sys.version.split(' ')[0]}")
    print(f"PyTorch:           {torch.__version__}")
    print(f"Stable Baselines3: {stable_baselines3.__version__}")
    print(f"Gymnasium:         {gym.__version__}")
    print(f"NumPy:             {numpy.__version__}\n")

    # 2. Hardware & CUDA Status
    print("[Hardware Acceleration]")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available:    {cuda_available}")
    
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        vram_alloc = torch.cuda.memory_allocated(0) / (1024**2)
        
        print(f"GPU Detected:      {gpu_name}")
        print(f"Total VRAM:        {vram_total:.2f} GB")
        print(f"Allocated VRAM:    {vram_alloc:.2f} MB")
    else:
        print("WARNING: CUDA is completely inactive. PyTorch cannot see your GPU.")
        print("Please review the CUDA installation steps in the upgrade guide.")
        sys.exit(1)

    # 3. RL Environment Health Check
    print("\n[RL Environment Integration]")
    try:
        env = ResonatorEnv()
        obs, info = env.reset()
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        print("Environment Load:  SUCCESS")
        print("Simulated Step:    SUCCESS")
    except Exception as e:
        print(f"Environment Load:  FAILED")
        print(f"Error Details: {e}")
        
    print("\n=====================================")
    if cuda_available:
        print("SYSTEM READY: You are clear to launch rl_training/train.py")

if __name__ == "__main__":
    verify_gpu_pipeline()