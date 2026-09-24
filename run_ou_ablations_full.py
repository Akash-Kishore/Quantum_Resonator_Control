import subprocess, sys, time, os

REPO = r"C:\Projects\MiniProject_Sem4"
PY = r"conda run -n quantum_control python"

configs = [
    ("v4a", 1), ("v4a", 2), ("v4a", 3), ("v4a", 4),
    ("v4b", 1), ("v4b", 2), ("v4b", 3), ("v4b", 4),
]

for ver, seed in configs:
    cmd = f'{PY} -m rl_training.train --version {ver} --seed {seed}'
    print(f'\n=== Starting {ver} seed {seed} (OU) ===')
    start = time.time()
    res = subprocess.run(cmd, shell=True, cwd=REPO, timeout=5400)  # 90 min per run
    elapsed = time.time() - start
    print(f'--- {ver} seed {seed} finished in {elapsed/60:.1f} min, exit={res.returncode} ---')
    if res.returncode != 0:
        print(f'ERROR: {ver} seed {seed} failed')
        sys.exit(1)
print('All OU ablations done.')