import json
PY = "C:\\Users\\akash\\anaconda3\\envs\\quantum_control\\python.exe"
S = "C:\\Projects\\MiniProject_Sem4\\qstudy"
W = "C:\\Projects\\MiniProject_Sem4_QSTUDY_WORK"
jobs = []
prev = "s6_train_rl__dither__plain_s4"  # tail of the Stage 6 chain

smokes = [("rl__rabi__K4__smoke", 0), ("rl__ramsey_iq__gru8__smoke", 0), ("rl__dither__DRcurr__smoke", 0)]
for cfgname, seed in smokes:
    jid = "s10_smoke_%s" % cfgname
    jobs.append({
        "id": jid, "lane": "gpu", "cwd": S,
        "cmd": '"%s" train_ppo.py --config configs\\%s.json --seed %d --out runs\\%s\\seed_%d '
               '> "%s\\results\\%s.log" 2>&1' % (PY, cfgname, seed, cfgname, seed, W, jid),
        "env": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        "depends_on": [prev],
        "expected_outputs": ["%s\\runs\\%s\\seed_%d\\best_model.zip" % (S, cfgname, seed),
                              "%s\\runs\\%s\\seed_%d\\vec_normalize.pkl" % (S, cfgname, seed),
                              "%s\\runs\\%s\\seed_%d\\train_summary.json" % (S, cfgname, seed)],
        "stall_minutes": 30, "max_wall_hours": 2,
        "watch": ["%s\\runs\\%s\\seed_%d\\progress.log" % (S, cfgname, seed)],
    })
    prev = jid

# then the full training batches, all chained on gpu:
def train_job(cfgname, seed, prev):
    jid = "s_tr_%s_s%d" % (cfgname, seed)
    j = {
        "id": jid, "lane": "gpu", "cwd": S,
        "cmd": '"%s" train_ppo.py --config configs\\%s.json --seed %d --out runs\\%s\\seed_%d '
               '> "%s\\results\\%s.log" 2>&1' % (PY, cfgname, seed, cfgname, seed, W, jid),
        "env": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        "depends_on": [prev],
        "expected_outputs": ["%s\\runs\\%s\\seed_%d\\best_model.zip" % (S, cfgname, seed),
                              "%s\\runs\\%s\\seed_%d\\vec_normalize.pkl" % (S, cfgname, seed),
                              "%s\\runs\\%s\\seed_%d\\train_summary.json" % (S, cfgname, seed)],
        "stall_minutes": 45, "max_wall_hours": 24,
        "watch": ["%s\\runs\\%s\\seed_%d\\progress.log" % (S, cfgname, seed)],
    }
    return j, jid

BATCHES = []
BATCHES += [("rl__ramsey_iq__plain", s) for s in range(5)]                    # Stage 7
BATCHES += [("rl__%s__filter" % p, s) for p in ("rabi", "dither", "ramsey_iq") for s in range(5)]  # Stage 9
BATCHES += [("rl__%s__K%d" % (p, K), s) for p in ("rabi", "ramsey_iq") for K in (4, 8, 16) for s in range(5)]  # Stage 10
BATCHES += [("rl__%s__gru8" % p, s) for p in ("rabi", "ramsey_iq") for s in range(5)]
BATCHES += [("rl__%s__DR" % p, s) for p in ("rabi", "dither", "ramsey_iq") for s in range(5)]      # Stage 11
BATCHES += [("rl__%s__DRcurr" % p, s) for p in ("rabi", "dither", "ramsey_iq") for s in range(5)]

for cfgname, seed in BATCHES:
    j, prev = train_job(cfgname, seed, prev)
    jobs.append(j)

with open(W + "\\jobs_json\\s7_11_train.json", "w") as f:
    json.dump(jobs, f, indent=1)
print("wrote %d jobs (3 smokes + %d trainings)" % (len(jobs), len(BATCHES)))
