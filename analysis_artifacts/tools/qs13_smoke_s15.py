#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""qs13_smoke_s15.py - small smoke tests for the S15 Tier A arms (nothing sealed, nothing under STUDY\\results).
  gen    writes 4 tiny configs (WORK\\smoke_cfgs) and the job file WORK\\jobs_json\\p12_smoke.json (6 jobs):
         E1 smoke training (20000 steps, gpu), fine-tune smoke (3000 steps, cpu1, init from the E1 smoke model, lr/10, frozen statistics),
         evaluation of the E1 smoke model on LG and on RB (val split, 5 episodes), Bayes-nominal and oracle on LG (val, 5 episodes).
  check  after the jobs are DONE: markers exist, results contain no NaN, prints the numbers.
Usage: python qs13_smoke_s15.py gen|check [--study S] [--work W] [--py PY]"""
import os, sys, json, argparse, glob, math

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["gen", "check"])
here = os.path.dirname(os.path.abspath(__file__))
ap.add_argument("--work", default=os.path.dirname(here))
ap.add_argument("--study", default=None)
ap.add_argument("--py", default=sys.executable)
a = ap.parse_args()
W = a.work
S = a.study or os.path.join(os.path.dirname(W), "MiniProject_Sem4", "qstudy")
PY = a.py
OUT = os.path.join(W, "smoke_s15")
CF = os.path.join(W, "smoke_cfgs")
RUNS = os.path.join(S, "runs_smoke")

def load(p):
    return json.load(open(p, encoding="utf-8"))

if a.cmd == "gen":
    os.makedirs(CF, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    e1 = load(os.path.join(S, "configs", "frozen", "s15__E1.json"))
    ft = load(os.path.join(S, "configs", "frozen", "s15__FTB_50000.json"))
    e1.update(total_steps=20000, val_every=10000, snapshots=[])
    ft.update(total_steps=3000, val_every=1000, init_from=os.path.join(RUNS, "s15__E1", "seed_{seed}"))
    paths = {"E1": os.path.join(CF, "s15__E1__smoke.json"), "FT": os.path.join(CF, "s15__FTB__smoke.json")}
    json.dump(e1, open(paths["E1"], "w"), indent=1)
    json.dump(ft, open(paths["FT"], "w"), indent=1)

    def job(jid, lane, cmd, dep, outs, watch=()):
        return dict(id=jid, lane=lane, cwd=S, cmd=cmd + ' > "%s" 2>&1' % os.path.join(W, "results", jid + ".log"), depends_on=dep, expected_outputs=outs,
                    env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"} if lane != "gpu" else {}, max_retries=0, stall_minutes=60, max_wall_hours=2, watch=list(watch))
    d_e1 = os.path.join(RUNS, "s15__E1", "seed_0")
    d_ft = os.path.join(RUNS, "s15__FTB", "seed_0")
    three = lambda d: [os.path.join(d, "best_model.zip"), os.path.join(d, "vec_normalize.pkl"), os.path.join(d, "train_summary.json")]
    jobs = [job("p12_train_E1", "gpu", '"%s" train_ppo.py --config "%s" --seed 0 --out "%s"' % (PY, paths["E1"], d_e1), [], three(d_e1)),
            job("p12_train_FT", "cpu1", '"%s" train_ppo.py --config "%s" --seed 0 --out "%s"' % (PY, paths["FT"], d_ft), ["p12_train_E1"], three(d_ft))]
    def ev(jid, method, cfgf, tag, dep):
        cmd = '"%s" tools%seval_set.py --method %s --cfg %s --set nominal --probe rabi --split val --episodes 5 --seed_base 500000 --outdir "%s" --tag %s' % (
            PY, os.sep, method, os.path.join("configs", cfgf), OUT, tag)
        return job(jid, "cpu2", cmd, dep, [os.path.join(OUT, tag + "__nominal.DONE.json")])
    m = "rl:" + d_e1
    jobs += [ev("p12_eval_E1_on_LG", m, "eval__rabi__LG.json", "p12E1LG", ["p12_train_E1"]),
             ev("p12_eval_E1_on_RB", m, "eval__rabi.json", "p12E1RB", ["p12_train_E1"]),
             ev("p12_eval_bayes_LG", "ctrl:bayes:nominal", "eval__rabi__LG.json", "p12bayesLG", []),
             ev("p12_eval_oracle_LG", "ctrl:oracle", "eval__rabi__LG.json", "p12oracleLG", [])]
    os.makedirs(os.path.join(W, "jobs_json"), exist_ok=True)
    json.dump(jobs, open(os.path.join(W, "jobs_json", "p12_smoke.json"), "w"), indent=1)
    print("p12_smoke.json: %d jobs; configs in %s" % (len(jobs), CF))
    print("QS13-GEN-DONE")
else:
    bad = 0
    for tag in ("p12E1LG", "p12E1RB", "p12bayesLG", "p12oracleLG"):
        mk = os.path.join(OUT, tag + "__nominal.DONE.json")
        if not os.path.exists(mk):
            print("MISSING marker " + tag)
            bad += 1
            continue
        for f in load(mk).get("files", []):
            p = f["path"] if os.path.isabs(f["path"]) else os.path.join(OUT, os.path.basename(f["path"].replace("\\", "/")))
            if not os.path.exists(p):
                p = os.path.join(OUT, os.path.basename(f["path"].replace("\\", "/")))
            txt = open(p, encoding="utf-8").read()
            if "NaN" in txt or "Infinity" in txt:
                print("NaN/Inf in " + p)
                bad += 1
            d = json.loads(txt)
            lock = d.get("lock_w1000") or d.get("metrics", {}).get("lock_w1000") if isinstance(d, dict) else None
            print("%-12s %-24s lock_w1000=%s" % (tag, os.path.basename(p)[:24], lock))
    for d in ("s15__E1", "s15__FTB"):
        ok = all(os.path.exists(os.path.join(RUNS, d, "seed_0", n)) for n in ("best_model.zip", "vec_normalize.pkl", "train_summary.json"))
        print("training outputs %-9s %s" % (d, "OK" if ok else "MISSING"))
        bad += 0 if ok else 1
    print("SMOKE %s" % ("PASS" if not bad else "FAIL (%d problems)" % bad))
