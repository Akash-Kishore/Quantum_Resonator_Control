#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gen_z2h.py - S15 TIER B evals for E2 and E3: writes the sealed evaluation job files.
Arms: E2 = env LB (lorentz+binom), E3 = env RG (rabi+gauss), seeds 200-204.
5 cells per model: nominal LG, LB, RG, RB (1000 eps) + OFAT on RB (300 eps).
Sealed split base 920000. Id prefix z2h_. No depends_on between eval jobs; each depends only on its training job.
Usage: python gen_z2h.py [--study STUDY] [--work WORK] [--py PYTHON]"""
import os, sys, json, argparse

ap = argparse.ArgumentParser()
here = os.path.dirname(os.path.abspath(__file__))
ap.add_argument("--work", default=os.path.dirname(here))
ap.add_argument("--study", default=None, help="folder qstudy (default: <work>\\..\\MiniProject_Sem4\\qstudy)")
ap.add_argument("--py", default=r"C:\Users\akash\anaconda3\envs\quantum_control\python.exe")
a = ap.parse_args()
W = a.work
S = a.study or os.path.join(os.path.dirname(W), "MiniProject_Sem4", "qstudy")
PY = a.py
SEEDS = list(range(200, 205))
CPUS = ["cpu1", "cpu2", "cpu3", "cpu4"]
SEALED = 920000

# Evaluation environments (same as E1/E4 in gen_z2a.py)
ENVS = {
    "LG": "eval__rabi__LG.json",
    "LB": "eval__rabi__LB.json",
    "RG": "eval__rabi__RG.json",
    "RB": "eval__rabi.json",      # quantum surrogate (default engine)
}

def log(jid):
    return os.path.join(W, "results", jid + ".log")

def marker(tag, st):
    return os.path.join(S, "results", "sealed", tag + "__" + st + ".DONE.json")

def eval_job(jid, lane, method, cfgf, st, eps, tag, dep):
    cmd = '"%s" tools\\eval_set.py --method %s --cfg %s --set %s --probe rabi --split sealed --episodes %d --seed_base %d --outdir %s --tag %s' % (
        PY, method, os.path.join("configs", cfgf), st, eps, SEALED, os.path.join("results", "sealed"), tag)
    cmd += ' > "%s" 2>&1' % log(jid)
    return dict(id=jid, lane=lane, cwd=S, cmd=cmd, depends_on=[dep], expected_outputs=[marker(tag, st)],
                env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, max_retries=1, stall_minutes=240, max_wall_hours=48, watch=[])

ev = []
i = 0
arms = ["s15__E2", "s15__E3"]
for s in SEEDS:
    for name in arms:
        train_jid = "z2g_%d_%s" % (s, name)
        method = "rl:" + os.path.join("runs", name, "seed_%d" % s)
        # Nominal evaluations on all 4 environments
        for e, cfgf in ENVS.items():
            tag = "z2h__%s_s%d__%s" % (name, s, e)
            jid = "z2h_nominal_%s_s%d_%s" % (name, s, e)
            ev.append(eval_job(jid, CPUS[i % 4], method, cfgf, "nominal", 1000, tag, train_jid))
            i += 1
        # OFAT on RB (quantum surrogate)
        tag = "z2h__%s_s%d__RB" % (name, s)
        jid = "z2h_ofat_%s_s%d_RB" % (name, s)
        ev.append(eval_job(jid, CPUS[i % 4], method, ENVS["RB"], "ofat", 300, tag, train_jid))
        i += 1

os.makedirs(os.path.join(W, "jobs_json"), exist_ok=True)
with open(os.path.join(W, "jobs_json", "z2h_eval.json"), "w") as f:
    json.dump(ev, f, indent=1)
print("z2h_eval.json %d" % len(ev))
print("GEN-Z2H-DONE total=%d (nothing submitted)" % len(ev))