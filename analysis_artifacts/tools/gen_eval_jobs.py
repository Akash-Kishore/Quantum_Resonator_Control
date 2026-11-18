import json
PY = "C:\\Users\\akash\\anaconda3\\envs\\quantum_control\\python.exe"
S = "C:\\Projects\\MiniProject_Sem4\\qstudy"
W = "C:\\Projects\\MiniProject_Sem4_QSTUDY_WORK"
jobs = []

# Stage 7.3: tau sweep {25,50,100} us for PID ramsey_iq on TUNE seeds (200 eps)
for tau_us in (25, 50, 100):
    name = "ctrl_pid__ramsey_tau%d__tune" % tau_us
    cfgfile = "eval__ramsey_tau%d.json" % tau_us
    with open(S + "\\configs\\" + cfgfile, "w") as f:
        json.dump({"probe": "ramsey_iq", "tau": tau_us * 1e-6}, f)
    jobs.append({
        "id": "s7_eval_%s" % name, "lane": "cpu2", "cwd": S,
        "cmd": '"%s" evaluate.py --method ctrl:pid --cfg configs\\%s --cond nom --split tune '
               '--episodes 200 --seed_base 100000 --out results\\val\\%s.json '
               '> "%s\\results\\%s.log" 2>&1' % (PY, cfgfile, name, W, name),
        "env": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        "expected_outputs": ["%s\\results\\val\\%s.json" % (S, name)],
        "stall_minutes": 30,
    })

# Stage 8.3: PID probes over all OFAT noise factors (val seeds, 200 eps)
OFAT_NOISE = {
    "shots": [64, 128, 256, 512, 1024, 4096, 16384],
    "readout": [0.0, 0.02, 0.05, 0.10, 0.20],
    "t2": [25e-6, 50e-6, 100e-6, 200e-6, 800e-6, None],   # None = inf
    "action_delay": [0, 1, 2, 3],
    "dac_bits": [6, 8, 10, 12, 16],
}
PROBES = (("hill", "rabi"), ("pid", "dither"), ("pid", "ramsey_iq"))


def fmt(v):
    if v is None:
        return "inf"
    return ("%g" % v)


for meth, probe in PROBES:
    for factor, vals in OFAT_NOISE.items():
        for v in vals:
            if factor == "t2" and probe != "ramsey_iq":
                continue  # T2 acts only through the Ramsey contrast (documented)
            vid = fmt(v) if factor != "t2" else (str(int(v * 1e6)) if v is not None else "inf")
            # match stress.py condition ids where possible; otherwise custom cfg file
            if factor == "t2" and v is not None:
                cond = "ofat_t2_%d" % int(v * 1e6)
            elif factor == "t2":
                cond = "ofat_t2_inf"
            else:
                cond = "ofat_%s_%s" % (factor, vid)
            name = "noise_%s__%s__%s" % (meth, probe, cond)
            cfgfile = "eval__%s.json" % probe
            if factor == "readout":
                with open(S + "\\configs\\eval__%s__ro.json" % probe, "w") as f:
                    json.dump({"probe": probe, "e01": v, "e10": v}, f)
                cfgfile = "eval__%s__ro.json" % probe
                cond = "nom"  # readout via cfg file (factor not in stress ids for values)
            jobs.append({
                "id": "s8_eval_%s" % name, "lane": "cpu3", "cwd": S,
                "cmd": '"%s" evaluate.py --method ctrl:%s --cfg configs\\%s --cond %s --split val '
                       '--episodes 200 --seed_base 500000 --out results\\val\\%s.json '
                       '> "%s\\results\\%s.log" 2>&1' % (PY, meth, cfgfile, cond, name, W, name),
                "env": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                "expected_outputs": ["%s\\results\\val\\%s.json" % (S, name)],
                "stall_minutes": 30,
            })

# Stage 9.4: Bayes nominal/oracle x 3 probes at nominal + val set (200 eps, val seeds)
VAL_CONDS = ["nom", "ofat_drift_sigma_1000", "ofat_shots_128", "ofat_t2_50"]
for assumed in ("nominal", "oracle"):
    for probe in ("rabi", "dither", "ramsey_iq"):
        for cond in VAL_CONDS:
            name = "bayes_%s__%s__%s" % (assumed, probe, cond)
            jobs.append({
                "id": "s9_eval_%s" % name, "lane": "cpu4", "cwd": S,
                "cmd": '"%s" evaluate.py --method ctrl:bayes:%s --cfg configs\\eval__%s.json --cond %s '
                       '--split val --episodes 200 --seed_base 500000 --out results\\val\\%s.json '
                       '> "%s\\results\\%s.log" 2>&1' % (PY, assumed, probe, cond, name, W, name),
                "env": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                "expected_outputs": ["%s\\results\\val\\%s.json" % (S, name)],
                "stall_minutes": 60,
            })

with open(W + "\\jobs_json\\s7_9_eval.json", "w") as f:
    json.dump(jobs, f, indent=1)
print("wrote %d eval jobs" % len(jobs))
