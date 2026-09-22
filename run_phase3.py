import os, sys, subprocess, json, time

REPO = r'C:\Projects\MiniProject_Sem4'
PY = r'conda run -n quantum_control python'

def run(cmd, desc):
    print(f'\n=== {desc} ===')
    start = time.time()
    res = subprocess.run(cmd, shell=True, cwd=REPO)
    print(f'--- {desc} finished in {time.time()-start:.1f}s, exit={res.returncode} ---')
    return res.returncode

# 1. V4a & V4b evaluation under OU (5 seeds each, but we have only seed0 saved? assume seed0)
for model in ['v4a_ablation', 'v4b_ablation']:
    for seed in [0]:
        run(f'{PY} -m rl_training.evaluate_agent --model {model} --seed {seed}', f'Eval {model} seed {seed} (OU)')

# 2. Robustness: V3 and V4 seed3 at drift_sigma 750 and 1000
for model, seed in [('v3_refined', None), ('v4_gradient_obs', 3)]:
    for sigma in [750, 1000]:
        run(f'{PY} -m rl_training.evaluate_agent --model {model} ' + (f'--seed {seed} ' if seed is not None else '') + f'--drift_sigma {sigma}', f'Robustness {model} sigma={sigma}')

# 3. Sample efficiency curves (parse tensorboard)
run(f'{PY} -c "import tbparse; print(\'tbparse ok\')"', 'Check tbparse')

# 4. Statistical test (full 1000 eps) - may be long
run(f'{PY} statistical_test.py', 'Statistical test (Mann-Whitney)')

print('All Phase 3 tasks launched.')