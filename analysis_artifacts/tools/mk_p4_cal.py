import json, os
W = r"C:\Projects\MiniProject_Sem4_QSTUDY_WORK"
QJ = os.path.join(W, "jobs_json"); SM = os.path.join(W, "smoke")
PY = r"C:\Users\akash\anaconda3\envs\quantum_control\python.exe"
CWD = r"C:\Projects\MiniProject_Sem4\qstudy"
def job(jid, lane, method, cfg, st, probe, eps, outdir, wall, stall, engine=None, deps=None):
    od = os.path.join(SM, outdir)
    cmd = '"%s" tools\\eval_set.py --method %s --cfg %s --set %s --probe %s --split val --episodes %d --seed_base 500000 --outdir "%s" --tag p4cal' % (PY, method, cfg, st, probe, eps, od)
    if engine: cmd += " --engine " + engine
    cmd += ' > "%s" 2>&1' % os.path.join(W, "results", jid + ".log")
    return {"id": jid, "lane": lane, "cwd": CWD, "cmd": cmd, "depends_on": deps or [],
            "expected_outputs": [os.path.join(od, "p4cal__%s.DONE.json" % st)],
            "env": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, "max_retries": 1,
            "stall_minutes": stall, "max_wall_hours": wall, "watch": []}
RL = "rl:runs_smoke\\rl__dither__plain"; SMOKE_DEP = ["p3_smoke_rl__dither__plain"]
jobs = [
  job("p4_cal_rl_5",     "cpu1", RL, "configs\\evalcfg__rl__dither__plain.json", "ofat", "dither", 5,   "p4_cal_rl_5",     8, 90, deps=SMOKE_DEP),
  job("p4_cal_rl_300",   "cpu1", RL, "configs\\evalcfg__rl__dither__plain.json", "ofat", "dither", 300, "p4_cal_rl_300",   8, 90, deps=SMOKE_DEP),
  job("p4_cal_pid_5",    "cpu2", "ctrl:pid", "configs\\eval__dither.json",        "ofat", "dither", 5,   "p4_cal_pid_5",    8, 90),
  job("p4_cal_pid_300",  "cpu2", "ctrl:pid", "configs\\eval__dither.json",        "ofat", "dither", 300, "p4_cal_pid_300",  8, 90),
  job("p4_cal_bayes_5",  "cpu3", "ctrl:bayes:nominal", "configs\\eval__ramsey_iq.json", "grids", "ramsey_iq", 5,   "p4_cal_bayes_5",  10, 90),
  job("p4_cal_bayes_300","cpu3", "ctrl:bayes:nominal", "configs\\eval__ramsey_iq.json", "grids", "ramsey_iq", 300, "p4_cal_bayes_300",10, 90),
  job("p4_cal_aer_2",    "aer",  "ctrl:pid", "configs\\eval__dither.json",        "aer6", "dither", 2,   "p4_cal_aer_2",    2, 60, engine="aer"),
  job("p4_cal_aer_50",   "aer",  "ctrl:pid", "configs\\eval__dither.json",        "aer6", "dither", 50,  "p4_cal_aer_50",   6, 60, engine="aer"),
]
p = os.path.join(QJ, "p4_cal.json")
assert not os.path.exists(p), "exists"
json.dump(jobs, open(p, "w", encoding="utf-8"), indent=1)
print("p4_cal.json", len(jobs), "jobs; lanes:", ",".join(sorted(set(j["lane"] for j in jobs))))
