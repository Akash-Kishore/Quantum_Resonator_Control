print("Step 1: Importing NumPy...")
import numpy as np
print(f"-> Success. NumPy version: {np.__version__}")

print("\nStep 2: Importing PyTorch (This might take 10-20 seconds)...")
import torch
print(f"-> Success. Torch version: {torch.__version__}")

print("\nStep 3: Checking CUDA/GPU...")
cuda_available = torch.cuda.is_available()
if cuda_available:
    print(f"-> Success. GPU found: {torch.cuda.get_device_name(0)}")
else:
    print("-> WARNING: GPU not detected by Torch.")

print("\nStep 4: Importing Qiskit Aer (This often hangs if versions mismatch)...")
try:
    from qiskit_aer import AerSimulator
    print("-> Success. Qiskit Aer imported.")
except Exception as e:
    print(f"-> FAILED: {e}")

print("\nStep 5: Initializing Simulator...")
sim = AerSimulator()
print("-> Success. Simulator is ready.")

print("\nStep 6: Importing Stable Baselines 3...")
import stable_baselines3 as sb3
print(f"-> Success. SB3 version: {sb3.__version__}")

print("\n--- ALL TESTS PASSED ---")