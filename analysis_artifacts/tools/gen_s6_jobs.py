import json
PY = "C:\\Users\\akash\\anaconda3\\envs\\quantum_control\\python.exe"
S = "C:\\Projects\\MiniProject_Sem4\\qstudy"
W = "C:\\Projects\\MiniProject_Sem4_QSTUDY_WORK"
jobs = []
VAL_CONDS = ["nom", "ofat_drift_sigma_1000", "ofat_shots_128", "ofat_t2_50"]

# (i) controller baselines on val seeds: NoControl, HillClimb(rabi), PID(dither) at nominal + val set
for meth, probe in (("ctrl:noctrl", "rabi"), ("ctrl:hill", "rabi"), ("ctrl:pid", "dither")):
    for cond in ["nom"] + VAL_CONDS[1:]:
        name = "%s__%s" % (meth.replace(":", "_"), cond)
        jobs.append({
            "id": "s6_eval_%s" % name,
            "lane": "cpu1",
            "cwd": S,
            "cmd": '"%s" evaluate.py --method %s --cfg configs\\eval__%s.json --cond %s --split val '
                   '--episodes 200 --seed_base 500000 --out results\\val\\%s.json --traces 5 '
                   '> "%s\\results\\%s.log" 2>&1' % (PY, meth, probe, cond, name, W, name),
            "env": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            "expected_outputs": ["%s\\results\\val\\%s.json" % (S, name)],
            "stall_minutes": 30,
        })

# (ii) RL training rl__rabi__plain and rl__dither__plain, seeds 0-4, chained on gpu
prev = "s6_train_smoke"
for cfgname in ("rl__rabi__plain", "rl__dither__plain"):
    for seed in range(5):
        jid = "s6_train_%s_s%d" % (cfgname, seed)
        jobs.append({
            "id": jid,
            "lane": "gpu",
            "cwd": S,
            "cmd": '"%s" train_ppo.py --config configs\\%s.json --seed %d --out runs\\%s\\seed_%d '
                   '> "%s\\results\\%s.log" 2>&1' % (PY, cfgname, seed, cfgname, seed, W, jid),
            "env": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            "depends_on": [prev],
            "expected_outputs": ["%s\\runs\\%s\\seed_%d\\best_model.zip" % (S, cfgname, seed),
                                  "%s\\runs\\%s\\seed_%d\\vec_normalize.pkl" % (S, cfgname, seed),
                                  "%s\\runs\\%s\\seed_%d\\train_summary.json" % (S, cfgname, seed)],
            "stall_minutes": 45, "max_wall_hours": 24,
            "watch": ["%s\\runs\\%s\\seed_%d\\progress.log" % (S, cfgname, seed)],
        })
        prev = jid

with open(W + "\\jobs_json\\s6_jobs_b.json", "w") as f:
    json.dump(jobs, f, indent=1)
print("wrote %d jobs" % len(jobs))
