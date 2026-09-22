import subprocess, sys, time, os

REPO = r'C:\Projects\MiniProject_Sem4'
PY = r'conda run -n quantum_control python'

seeds = [0,1,2,3,4]
ver = "confound_B"

for seed in seeds:
    cmd = f'{PY} -m rl_training.train --version {ver} --seed {seed}'
    print(f'\n=== Starting {ver} seed {seed} ===')
    start = time.time()
    res = subprocess.run(cmd, shell=True, cwd=REPO, timeout=18000)  # 5h per run max
    elapsed = time.time() - start
    print(f'--- {ver} seed {seed} finished in {elapsed/60:.1f} min, exit={res.returncode} ---')
    if res.returncode != 0:
        print(f'ERROR: {ver} seed {seed} failed')
        sys.exit(1)
print('All confound_B runs completed.')