#!/bin/bash
set -e
REPO="C:/Projects/MiniProject_Sem4"
cd "$REPO"
for ver in v4a v4b; do
  for seed in 1 2 3 4; do
    echo "=== Training $ver seed $seed ==="
    conda run -n quantum_control python -m rl_training.train --version $ver --seed $seed
  done
done
echo "All Gaussian trainings done."