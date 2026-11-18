"""S3: gen_z2a_snap.py - clone the s15__E4 z2e eval jobs onto staged snapshot dirs.
Same cells (nominal LG/LB/RG/RB 1000ep; ofat RB 300ep), same evalcfg, split sealed,
base 920000. model_dir = each staged snapshot dir (4 steps x 5 seeds). id prefix z2s_.
expected outputs extended by the step. Writes jobs_json\z2s_eval.json. Submits NOTHING.
"""
import json, os, sys, argparse

here = os.path.dirname(os.path.abspath(__file__))
W = os.path.dirname(here)
S = os.path.join(os.path.dirname(W), "MiniProject_Sem4", "qstudy")
PY = sys.executable
SEEDS = list(range(200, 205))
STEPS = [50000, 200000, 500000, 1000000]
CPUS = ["cpu1", "cpu2", "cpu3", "cpu4"]
SEALED = 920000
ENVS = ["LG", "LB", "RG", "RB"]

ap = argparse.ArgumentParser()
ap.add_argument("--work", default=W)
ap.add_argument("--study", default=S)
ap.add_argument("--py", default=PY)
a = ap.parse_args()
W, S, PY = a.work, a.study, a.py

def log(jid): return os.path.join(W, "results", jid + ".log")
def marker(tag, st): return os.path.join(S, "results", "sealed", tag + "__" + st + ".DONE.json")
def evalcfg(e): return "eval__rabi.json" if e == "RB" else "eval__rabi__%s.json" % e

def eval_job(jid, lane, method, cfgf, st, eps, tag):
    cmd = '"%s" tools%seval_set.py --method %s --cfg %s --set %s --probe rabi --split sealed --episodes %d --seed_base %d --outdir %s --tag %s > "%s" 2>&1' % (
        PY, os.sep, method, os.path.join("configs", cfgf), st, eps, SEALED,
        os.path.join("results", "sealed"), tag, log(jid))
    return dict(id=jid, lane=lane, cwd=S, cmd=cmd, depends_on=[],
                expected_outputs=[marker(tag, st)],
                env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                max_retries=1, stall_minutes=240, max_wall_hours=48, watch=[])

jobs = []
i = 0
for step in STEPS:
    for seed in SEEDS:
        method = "rl:" + os.path.join("runs_snap", "s15__E4_s%d" % step, "seed_%d" % seed)
        for e in ENVS:
            tag = "z2s__s15__E4_s%d__%s__s%d" % (seed, e, step)
            jid = "z2s_nominal_s15__E4_s%d__%s__s%d" % (seed, e, step)
            jobs.append(eval_job(jid, CPUS[i % 4], method, evalcfg(e), "nominal", 1000, tag))
            i += 1
        tag = "z2s__s15__E4_s%d__RB__s%d" % (seed, step)
        jid = "z2s_ofat_s15__E4_s%d__RB__s%d" % (seed, step)
        jobs.append(eval_job(jid, CPUS[i % 4], method, evalcfg("RB"), "ofat", 300, tag))
        i += 1

os.makedirs(os.path.join(W, "jobs_json"), exist_ok=True)
out = os.path.join(W, "jobs_json", "z2s_eval.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(jobs, f, indent=1)
print("z2s_eval.json %d jobs" % len(jobs))
print("steps=%s seeds=%s cells/seed/step=5 (nominal LG LB RG RB + ofat RB)" % (STEPS, SEEDS))
print("GEN-Z2A-SNAP-DONE (nothing submitted)")
