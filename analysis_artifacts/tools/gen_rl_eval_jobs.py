import json, os
PY = "C:\\Users\\akash\\anaconda3\\envs\\quantum_control\\python.exe"
S = "C:\\Projects\\MiniProject_Sem4\\qstudy"
W = "C:\\Projects\\MiniProject_Sem4_QSTUDY_WORK"

# eval cfg per config family (probe + any env cfg needed for obs shape)
EVAL_CFG = {
    "rl__rabi__plain": "eval__rabi.json",
    "rl__dither__plain": "eval__dither.json",
    "rl__ramsey_iq__plain": "eval__ramsey_iq.json",
    "rl__rabi__filter": None, "rl__dither__filter": None, "rl__ramsey_iq__filter": None,
    "rl__rabi__K4": None, "rl__rabi__K8": None, "rl__rabi__K16": None, "rl__rabi__gru8": None,
    "rl__ramsey_iq__K4": None, "rl__ramsey_iq__K8": None, "rl__ramsey_iq__K16": None,
    "rl__ramsey_iq__gru8": None,
    "rl__rabi__DR": None, "rl__dither__DR": None, "rl__ramsey_iq__DR": None,
    "rl__rabi__DRcurr": None, "rl__dither__DRcurr": None, "rl__ramsey_iq__DRcurr": None,
}
# write eval cfg files that mirror each config's env
for cfgname in EVAL_CFG:
    spec = json.load(open(os.path.join(S, "configs", cfgname + ".json"), encoding="utf-8"))
    env = dict(spec.get("env", {}))
    evalname = "evalcfg__%s.json" % cfgname
    with open(os.path.join(S, "configs", evalname), "w") as f:
        json.dump(env, f)
    EVAL_CFG[cfgname] = evalname

VAL_CONDS = ["nom", "ofat_drift_sigma_1000", "ofat_shots_128", "ofat_t2_50"]
jobs = []
for cfgname, evalfile in EVAL_CFG.items():
    for seed in range(5):
        tr_id = "s_tr_%s_s%d" % (cfgname, seed)
        if cfgname in ("rl__rabi__plain", "rl__dither__plain"):
            tr_id = "s6_train_%s_s%d" % (cfgname, seed)
        for cond in VAL_CONDS:
            name = "rlval__%s__s%d__%s" % (cfgname, seed, cond)
            model_dir = "runs\\%s\\seed_%d" % (cfgname, seed)
            jobs.append({
                "id": "s_rlval_%s" % name,
                "lane": "cpu1",
                "cwd": S,
                "cmd": '"%s" evaluate.py --method rl:%s --cfg configs\\%s --cond %s --split val '
                       '--episodes 200 --seed_base 500000 --out results\\val\\%s.json '
                       '> "%s\\results\\%s.log" 2>&1' % (PY, model_dir, evalfile, cond, name, W, name),
                "env": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                "depends_on": [tr_id],
                "expected_outputs": ["%s\\results\\val\\%s.json" % (S, name)],
                "stall_minutes": 30,
            })

with open(W + "\\jobs_json\\s_rl_evals.json", "w") as f:
    json.dump(jobs, f, indent=1)
print("wrote %d RL eval jobs" % len(jobs))
