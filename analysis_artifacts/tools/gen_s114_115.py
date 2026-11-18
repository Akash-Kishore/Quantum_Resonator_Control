import json
PY = "C:\\Users\\akash\\anaconda3\\envs\\quantum_control\\python.exe"
S = "C:\\Projects\\MiniProject_Sem4\\qstudy"
W = "C:\\Projects\\MiniProject_Sem4_QSTUDY_WORK"
jobs = []
prev = "s_tr_rl__ramsey_iq__DRcurr_s4"  # tail of the big chain

# Stage 11.5: tuning grid, seed 100, chained at the tail of the gpu queue
for lr in ("3e-4", "1e-4"):
    for ent in ("0.0", "0.003"):
        for gam in ("0.99", "0.995"):
            name = "tune__lr%s__ent%s__gam%s" % (lr.replace("-", ""), ent.replace(".", ""), gam.replace(".", ""))
            jid = "s115_train_%s" % name
            jobs.append({
                "id": jid, "lane": "gpu", "cwd": S,
                "cmd": '"%s" train_ppo.py --config configs\\%s.json --seed 100 --out runs\\%s\\seed_100 '
                       '> "%s\\results\\%s.log" 2>&1' % (PY, name, name, W, jid),
                "env": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                "depends_on": [prev],
                "expected_outputs": ["%s\\runs\\%s\\seed_100\\best_model.zip" % (S, name),
                                      "%s\\runs\\%s\\seed_100\\vec_normalize.pkl" % (S, name),
                                      "%s\\runs\\%s\\seed_100\\train_summary.json" % (S, name)],
                "stall_minutes": 45, "max_wall_hours": 24,
                "watch": ["%s\\runs\\%s\\seed_100\\progress.log" % (S, name)],
            })
            prev = jid

# Stage 11.4: Aer fine-tune of rl__ramsey_iq__DRcurr seeds 0-2 (aer lane)
for seed in (0, 1, 2):
    jobs.append({
        "id": "s114_aerft_s%d" % seed, "lane": "aer", "cwd": S,
        "cmd": '"%s" finetune_aer.py --config configs\\rl__ramsey_iq__DRcurr.json '
               '--src runs\\rl__ramsey_iq__DRcurr\\seed_%d --out runs\\rl__ramsey_iq__DRcurr_aerft\\seed_%d '
               '--steps 300000 --n_envs 4 > "%s\\results\\s114_aerft_s%d.log" 2>&1' % (PY, seed, seed, W, seed),
        "env": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        "depends_on": ["s_tr_rl__ramsey_iq__DRcurr_s%d" % seed],
        "expected_outputs": ["%s\\runs\\rl__ramsey_iq__DRcurr_aerft\\seed_%d\\best_model.zip" % (S, seed),
                              "%s\\runs\\rl__ramsey_iq__DRcurr_aerft\\seed_%d\\vec_normalize.pkl" % (S, seed),
                              "%s\\runs\\rl__ramsey_iq__DRcurr_aerft\\seed_%d\\train_summary.json" % (S, seed)],
        "stall_minutes": 120, "max_wall_hours": 12,
        "watch": ["%s\\runs\\rl__ramsey_iq__DRcurr_aerft\\seed_%d\\progress.log" % (S, seed)],
    })

with open(W + "\\jobs_json\\s114_115.json", "w") as f:
    json.dump(jobs, f, indent=1)
print("wrote %d jobs" % len(jobs))
