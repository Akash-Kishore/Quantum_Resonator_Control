"""P3.1 (CONTINUATION5): smoke training jobs - nine frozen configs at 20k steps.

Writes WORK\smoke_cfgs\p3_<config>.json (total_steps 20000, val_every 10000,
snapshots []; real frozen configs untouched) and WORK\jobs_json\p3_smoke_train.json
(9 gpu jobs, ids p3_smoke_<config>, seed 3, out STUDY\runs_smoke\<config>).
"""
import json, os

S = r"C:\Projects\MiniProject_Sem4\qstudy"
W = r"C:\Projects\MiniProject_Sem4_QSTUDY_WORK"
PY = r"C:\Users\akash\anaconda3\envs\quantum_control\python.exe"
NINE = ["rl__rabi__plain", "rl__rabi__gru8", "rl__dither__plain", "rl__dither__filter",
        "rl__dither__DR", "rl__dither__DRcurr", "rl__dither__full",
        "rl__ramsey_iq__plain", "rl__ramsey_iq__gru8"]

jobs = []
for name in NINE:
    spec = json.load(open(os.path.join(S, "configs", "frozen", name + ".json"), encoding="utf-8"))
    spec["total_steps"] = 20000
    spec["val_every"] = 10000
    spec["snapshots"] = []
    cfgp = os.path.join(W, "smoke_cfgs", "p3_" + name + ".json")
    with open(cfgp, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=1)
    jid = "p3_smoke_" + name
    out = os.path.join(S, "runs_smoke", name)
    jobs.append(dict(
        id=jid, lane="gpu", cwd=S,
        cmd='"%s" train_ppo.py --config "%s" --seed 3 --out "%s" > "%s" 2>&1'
            % (PY, cfgp, out, os.path.join(W, "results", jid + ".log")),
        depends_on=[],
        expected_outputs=[os.path.join(out, "best_model.zip"),
                          os.path.join(out, "vec_normalize.pkl"),
                          os.path.join(out, "train_summary.json")],
        env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        max_retries=1, stall_minutes=60, max_wall_hours=2,
        watch=[os.path.join(out, "progress.log")]))
with open(os.path.join(W, "jobs_json", "p3_smoke_train.json"), "w", encoding="utf-8") as f:
    json.dump(jobs, f, indent=1)
print("p3_smoke_train.json", len(jobs))
