#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gen_z2g.py - S15 TIER B arms E2 and E3: writes the training job files only (no evals).
Arms (PREREG2 S15, seeds 200-204): E2 = env LB (lorentz+binom), E3 = env RG (rabi+gauss).
Same frozen hyperparameters as E4, 2M steps, snapshots at 50k/200k/500k/1M, probe rabi.
Job ids: z2g_<seed>_<name> (gpu lane, no depends_on).
Usage: python gen_z2g.py [--study STUDY] [--work WORK] [--py PYTHON]"""
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

def cfg_path(*p):
    return os.path.join(S, "configs", *p)

def write_cfg(path, spec):
    text = json.dumps(spec, indent=1)
    if os.path.exists(path):
        if json.load(open(path, encoding="utf-8")) != spec:
            sys.exit("REFUSED: %s exists with different content (never overwrite a frozen config)" % path)
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

# Load frozen PPO hyperparameters from E4 (identical to E1)
frozen = json.load(open(cfg_path("frozen", "s15__E4.json"), encoding="utf-8"))
PPO = dict(frozen["ppo"])

# ------------------------------------------------------------------ configs (write to configs_tierb)
train_cfg = {}
train_cfg["s15__E2"] = dict(env=dict(probe="rabi", engine="hybrid", lineshape="lorentz", noise="binom"),
                            ppo=PPO, total_steps=2000000, val_every=100000, snapshots=[50000, 200000, 500000, 1000000])
train_cfg["s15__E3"] = dict(env=dict(probe="rabi", engine="hybrid", lineshape="rabi", noise="gauss"),
                            ppo=PPO, total_steps=2000000, val_every=100000, snapshots=[50000, 200000, 500000, 1000000])

# Write configs under configs_tierb (not frozen, since these are new Tier B configs)
tierb_cfg_dir = os.path.join(S, "configs_tierb")
for name, spec in train_cfg.items():
    path = os.path.join(tierb_cfg_dir, name + ".json")
    write_cfg(path, spec)

# Verify run directories do not exist
runs_dir = os.path.join(S, "runs")
for name in ("s15__E2", "s15__E3"):
    for s in SEEDS:
        d = os.path.join(runs_dir, name, "seed_%d" % s)
        if os.path.exists(d):
            sys.exit("RUN DIR EXISTS (refusing to overwrite): %s" % d)

# ------------------------------------------------------------------ jobs
train = []

def log(jid):
    return os.path.join(W, "results", jid + ".log")

def train_job(jid, name, s):
    outdir = os.path.join(S, "runs", name, "seed_%d" % s)
    return dict(id=jid, lane="gpu", cwd=S,
                cmd='"%s" train_ppo.py --config %s --seed %d --out "%s" > "%s" 2>&1' % (
                    PY, os.path.join("configs_tierb", name + ".json"), s, outdir, log(jid)),
                depends_on=[],
                expected_outputs=[os.path.join(outdir, "best_model.zip"),
                                  os.path.join(outdir, "vec_normalize.pkl"),
                                  os.path.join(outdir, "train_summary.json")],
                env={}, max_retries=1, stall_minutes=90, max_wall_hours=24,
                watch=[os.path.join(outdir, "progress.log")])

for s in SEEDS:
    for name in ("s15__E2", "s15__E3"):
        jid = "z2g_%d_%s" % (s, name)
        train.append(train_job(jid, name, s))

os.makedirs(os.path.join(W, "jobs_json"), exist_ok=True)
with open(os.path.join(W, "jobs_json", "z2g_train.json"), "w") as f:
    json.dump(train, f, indent=1)
print("z2g_train.json %d" % len(train))
print("GEN-Z2G-DONE total=%d (nothing submitted)" % len(train))