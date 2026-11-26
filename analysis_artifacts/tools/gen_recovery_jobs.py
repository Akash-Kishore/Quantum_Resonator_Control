"""Recovery batch for A7/A8/A9 (Part-2 boot session, 2026-10-13).
Submits: (a) retrain jobs on the gpu lane with the s0z_ prefix (priority: they unblock the
Stage 11.5/11.8 selection) — dither_plain seed 3, 10 DRcurr retrains (curriculum fix A8),
8 tuning retrains; (b) eval jobs — 40 DR evals (models exist), 40 DRcurr evals chained to
their own retrain (O1-compliant), 4 dither_plain seed_3 evals chained to its retrain,
8 ramsey_iq gru8 eval gap-fills chained to their pending trainings."""
import json
import os
import sys

w = sys.argv[1]
py = sys.argv[2]
s = r"C:\Projects\MiniProject_Sem4\qstudy"
sys.path.insert(0, os.path.join(w, "tools"))
import jobtools

train_jobs = []
eval_jobs = []
VAL_CONDS = ["nom", "ofat_drift_sigma_1000", "ofat_shots_128", "ofat_t2_50"]
EVAL_CFG = {"rabi": "eval__rabi.json", "dither": "eval__dither.json", "ramsey_iq": "eval__ramsey_iq.json"}


def add_train(jid, cfg, seed, out, extra_exp=()):
    outdir = os.path.join(s, "runs", out, "seed_%d" % seed)
    cmd = '"%s" train_ppo.py --config configs\\%s --seed %d --out "%s"' % (py, cfg, seed, outdir)
    exp = [os.path.join(outdir, "best_model.zip"), os.path.join(outdir, "train_summary.json")]
    train_jobs.append(dict(id=jid, lane="gpu", cwd=s, cmd=cmd, depends_on=[],
                           expected_outputs=exp + list(extra_exp), env={}, max_retries=2,
                           stall_minutes=60, max_wall_hours=24,
                           watch=[os.path.join(outdir, "progress.log")]))


def add_eval(jid, model_dir, probe, cond, depends=()):
    name = jid.replace("s0z_eval_", "")
    outp = os.path.join(s, "results", "val", "rlval__%s__%s.json" % (name, cond))
    cmd = ('"%s" evaluate.py --method rl:"%s" --cfg configs\\%s --cond %s --split val '
           '--episodes 200 --seed_base 500000 --out "%s" --traces 5') % (py, model_dir, EVAL_CFG[probe], cond, outp)
    eval_jobs.append(dict(id=jid, lane="cpu3", cwd=s, cmd=cmd, depends_on=list(depends),
                          expected_outputs=[outp], env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                          max_retries=2, stall_minutes=60, max_wall_hours=6))


# (a) retrains
add_train("s0z_retrain_dither_plain_s3", "rl__dither__plain.json", 3, "rl__dither__plain")
for k in range(5):
    add_train("s0z_retrain_rabi_DRcurr_s%d" % k, "rl__rabi__DRcurr.json", k, "rl__rabi__DRcurr")
    add_train("s0z_retrain_dither_DRcurr_s%d" % k, "rl__dither__DRcurr.json", k, "rl__dither__DRcurr")
for cfg in ("tune__lr3e4__ent00__gam099", "tune__lr3e4__ent00__gam0995", "tune__lr3e4__ent0003__gam099",
            "tune__lr3e4__ent0003__gam0995", "tune__lr1e4__ent00__gam099", "tune__lr1e4__ent00__gam0995",
            "tune__lr1e4__ent0003__gam099", "tune__lr1e4__ent0003__gam0995"):
    add_train("s0z_retrain_%s" % cfg, cfg + ".json", 100, cfg)

# (b) evals
# DR evals (models exist, no deps)
for probe in ("rabi", "dither"):
    for k in range(5):
        for cond in VAL_CONDS:
            add_eval("s0z_eval_rl__%s__DR_s%d__%s" % (probe, k, cond),
                     os.path.join(s, "runs", "rl__%s__DR" % probe, "seed_%d" % k), probe, cond)
# DRcurr evals (chained to their own retrain; O1: evaluation jobs may depend on their own training only)
for probe in ("rabi", "dither"):
    for k in range(5):
        dep = ["s0z_retrain_%s_DRcurr_s%d" % (probe, k)]
        for cond in VAL_CONDS:
            add_eval("s0z_eval_rl__%s__DRcurr_s%d__%s" % (probe, k, cond),
                    os.path.join(s, "runs", "rl__%s__DRcurr" % probe, "seed_%d" % k), probe, cond, depends=dep)
# dither_plain seed_3 evals (chained to its retrain)
for cond in VAL_CONDS:
    add_eval("s0z_eval_rl__dither__plain_s3__%s" % cond,
             os.path.join(s, "runs", "rl__dither__plain", "seed_3"), "dither", cond,
             depends=["s0z_retrain_dither_plain_s3"])
# ramsey_iq gru8 eval gap (8 missing: seeds 3,4 x 4 conds; chained to their pending trainings)
for k in (3, 4):
    dep = ["s_tr_rl__ramsey_iq__gru8_s%d" % k]
    for cond in VAL_CONDS:
        add_eval("s0z_eval_rl__ramsey_iq__gru8_s%d__%s" % (k, cond),
                 os.path.join(s, "runs", "rl__ramsey_iq__gru8", "seed_%d" % k), "ramsey_iq", cond, depends=dep)

with open(os.path.join(w, "jobs_json", "recovery_train.json"), "w") as f:
    json.dump(train_jobs, f, indent=1)
with open(os.path.join(w, "jobs_json", "recovery_eval.json"), "w") as f:
    json.dump(eval_jobs, f, indent=1)
print("wrote", len(train_jobs), "retrain jobs and", len(eval_jobs), "eval jobs")
