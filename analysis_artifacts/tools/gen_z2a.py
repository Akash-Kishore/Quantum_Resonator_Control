#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gen_z2a.py - S15 TIER A (transfer study, minimum set): writes the training/evaluation configs and the job files. Submits NOTHING.
Arms (PREREG2 S15, seeds 200-204): E1 = classical env LG (lorentz+gauss), E4 = quantum env RB (the Part-1 surrogate); fine-tune E1 -> RB:
FT-B (lr = frozen/10, observation statistics frozen) at 50k/200k/500k/1M steps and FT-A (frozen hyper-parameters) at 200k steps.
Every policy is evaluated at nominal on the four environments LG, LB, RG, RB (1000 episodes), OFAT on RB (300), Aer6 for seeds 200-202 (100); sealed base 920000.
Baselines (noctrl, hill, bayes:nominal, bayes:oracle, oracle) on the same cells. Tier B/C (E2, E3, E5, E6, SAC/TD3, other FT-A budgets) are NOT generated here.
Job ids: z2a_<seed>_<name> (E arms), z2b_<seed>_<name> (fine-tunes; they sort after the E arms on the gpu lane, so E1 is finished first), z2e_* (evals), z2v_* (baselines), z2x_* (Aer).
Usage: python gen_z2a.py [--study STUDY] [--work WORK] [--py PYTHON]   (defaults: derived from this file's location and the running interpreter)"""
import os, sys, json, argparse

ap = argparse.ArgumentParser()
here = os.path.dirname(os.path.abspath(__file__))
ap.add_argument("--work", default=os.path.dirname(here))
ap.add_argument("--study", default=None, help="folder qstudy (default: <work>\\..\\MiniProject_Sem4\\qstudy)")
ap.add_argument("--py", default=sys.executable)
a = ap.parse_args()
W = a.work
S = a.study or os.path.join(os.path.dirname(W), "MiniProject_Sem4", "qstudy")
PY = a.py
SEEDS = list(range(200, 205))
CPUS = ["cpu1", "cpu2", "cpu3", "cpu4"]
FT_B = [50000, 200000, 500000, 1000000]
FT_A = [200000]
ENVS = {"LG": dict(engine="hybrid", lineshape="lorentz", noise="gauss"),
        "LB": dict(engine="hybrid", lineshape="lorentz", noise="binom"),
        "RG": dict(engine="hybrid", lineshape="rabi", noise="gauss"),
        "RB": {}}                                   # RB = the quantum surrogate of Part 1 (default engine)
BASE = ["ctrl:noctrl", "ctrl:hill", "ctrl:bayes:nominal", "ctrl:bayes:oracle", "ctrl:oracle"]
SEALED = 920000

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

frozen = json.load(open(cfg_path("frozen", "rl__rabi__plain.json"), encoding="utf-8"))
PPO = dict(frozen["ppo"])                           # the FROZEN hyper-parameters, unchanged
# ------------------------------------------------------------------ configs
train_cfg = {}
train_cfg["s15__E1"] = dict(env=dict(probe="rabi", **ENVS["LG"]), ppo=PPO, total_steps=2000000, val_every=100000, snapshots=[50000, 200000, 500000, 1000000])
train_cfg["s15__E4"] = dict(env=dict(probe="rabi"), ppo=PPO, total_steps=2000000, val_every=100000, snapshots=[50000, 200000, 500000, 1000000])
ft_init = os.path.join("runs", "s15__E1", "seed_{seed}")
for b in FT_B:
    ppo = dict(PPO)
    ppo["learning_rate"] = PPO["learning_rate"] / 10.0
    train_cfg["s15__FTB_%d" % b] = dict(env=dict(probe="rabi"), ppo=ppo, total_steps=b, val_every=max(10000, min(100000, b // 4)),
                                         init_from=ft_init, freeze_obs_norm=True)
for b in FT_A:
    train_cfg["s15__FTA_%d" % b] = dict(env=dict(probe="rabi"), ppo=dict(PPO), total_steps=b, val_every=max(10000, min(100000, b // 4)),
                                         init_from=ft_init, freeze_obs_norm=False)
for name, spec in train_cfg.items():
    write_cfg(cfg_path("frozen", name + ".json"), spec)
for e, extra in ENVS.items():                        # evaluation environments (same file layout as eval__rabi.json)
    if e != "RB":
        write_cfg(cfg_path("eval__rabi__%s.json" % e), dict(probe="rabi", **extra))

def evalcfg(e):
    return "eval__rabi.json" if e == "RB" else "eval__rabi__%s.json" % e

# ------------------------------------------------------------------ jobs
train, ev, base, aer = [], [], [], []
i = 0

def log(jid):
    return os.path.join(W, "results", jid + ".log")

def marker(tag, st):
    return os.path.join(S, "results", "sealed", tag + "__" + st + ".DONE.json")

def train_job(jid, name, s):
    outdir = os.path.join(S, "runs", name, "seed_%d" % s)
    return dict(id=jid, lane="gpu", cwd=S,
                cmd='"%s" train_ppo.py --config %s --seed %d --out "%s" > "%s" 2>&1' % (PY, os.path.join("configs", "frozen", name + ".json"), s, outdir, log(jid)),
                depends_on=[],
                expected_outputs=[os.path.join(outdir, "best_model.zip"), os.path.join(outdir, "vec_normalize.pkl"), os.path.join(outdir, "train_summary.json")],
                env={}, max_retries=1, stall_minutes=90, max_wall_hours=24, watch=[os.path.join(outdir, "progress.log")])

def eval_job(jid, lane, method, cfgf, st, eps, tag, dep, engine=None):
    cmd = '"%s" tools%seval_set.py --method %s --cfg %s --set %s --probe rabi --split sealed --episodes %d --seed_base %d --outdir %s --tag %s' % (
        PY, os.sep, method, os.path.join("configs", cfgf), st, eps, SEALED, os.path.join("results", "sealed"), tag)
    if engine:
        cmd += " --engine " + engine
    cmd += ' > "%s" 2>&1' % log(jid)
    return dict(id=jid, lane=lane, cwd=S, cmd=cmd, depends_on=dep, expected_outputs=[marker(tag, st)],
                env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, max_retries=1, stall_minutes=240, max_wall_hours=48, watch=[])

arms = ["s15__E1", "s15__E4"] + ["s15__FTB_%d" % b for b in FT_B] + ["s15__FTA_%d" % b for b in FT_A]
for s in SEEDS:
    for name in arms:
        jid = ("z2a_%d_%s" if name in ("s15__E1", "s15__E4") else "z2b_%d_%s") % (s, name)
        train.append(train_job(jid, name, s))
        method = "rl:" + os.path.join("runs", name, "seed_%d" % s)
        for e in ENVS:
            tag = "z2__%s_s%d__%s" % (name, s, e)
            ev.append(eval_job("z2e_nominal_%s_s%d_%s" % (name, s, e), CPUS[i % 4], method, evalcfg(e), "nominal", 1000, tag, [jid]))
            i += 1
        tag = "z2__%s_s%d__RB" % (name, s)
        ev.append(eval_job("z2e_ofat_%s_s%d_RB" % (name, s), CPUS[i % 4], method, evalcfg("RB"), "ofat", 300, tag, [jid]))
        i += 1
        if s <= 202:
            aer.append(eval_job("z2x_%s_s%d" % (name, s), "aer", method, evalcfg("RB"), "aer6", 100, "z2aer__%s_s%d" % (name, s), [jid], engine="aer"))
for m in BASE:
    mm = m.replace(":", "_")
    for e in ENVS:
        base.append(eval_job("z2v_nominal_%s_%s" % (mm, e), CPUS[i % 4], m, evalcfg(e), "nominal", 1000, "z2__base_%s__%s" % (mm, e), []))
        i += 1
    base.append(eval_job("z2v_ofat_%s_RB" % mm, CPUS[i % 4], m, evalcfg("RB"), "ofat", 300, "z2__base_%s__RB" % mm, []))
    i += 1

os.makedirs(os.path.join(W, "jobs_json"), exist_ok=True)
counts = {}
for fn, jobs in (("z2a_train.json", train), ("z2a_eval.json", ev), ("z2a_base.json", base), ("z2a_aer.json", aer)):
    with open(os.path.join(W, "jobs_json", fn), "w") as f:
        json.dump(jobs, f, indent=1)
    counts[fn] = len(jobs)
    print("%-16s %d" % (fn, len(jobs)))
with open(os.path.join(W, "jobs_json", "z2a_counts.json"), "w") as f:
    json.dump(counts, f, indent=1)
print("GEN-Z2A-DONE total=%d (nothing submitted)" % sum(counts.values()))
