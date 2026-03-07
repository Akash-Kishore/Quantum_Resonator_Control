import sys
print(f"Python Executable: {sys.executable}")
try:
    import numpy as np
    import scipy
    import pandas as pd
    import serial
    import torch
    import gymnasium as gym
    import stable_baselines3
    import PyQt5
    import dash
    import plotly
    print("All critical dependencies loaded successfully.")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Stable Baselines3 Version: {stable_baselines3.__version__}")
except ImportError as e:
    print(f"DEPENDENCY FAILURE: {e}")
    sys.exit(1)