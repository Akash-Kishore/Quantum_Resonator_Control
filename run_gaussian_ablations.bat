@echo off
cd /d C:\Projects\MiniProject_Sem4
for %%v in (v4a v4b) do (
  for %%s in (1 2 3 4) do (
    echo === Training %%v seed %%s ===
    conda run -n quantum_control python -m rl_training.train --version %%v --seed %%s
  )
)
echo All Gaussian trainings done.