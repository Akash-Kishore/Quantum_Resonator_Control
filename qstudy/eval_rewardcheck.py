"""qstudy eval_rewardcheck — Part-2 S17.8: reward-mismatch driver.

Best probe, 3 seeds each (420-422): R1 = base reward; R2 = threshold lock reward
(+1 if |e|<=1000 Hz else 0, minus 0.01|a| — envs.py reward_kind="threshold").
Both reward variants are trained and then evaluated on ALL metrics (lock widths, rms,
CVaR, lock-loss); the analysis reports cases where the TRAINING reward and the
SCIENTIFIC metrics disagree.

Modes:
  --emit-jobs FILE --config NAME : write R1/R2 training jobs (seeds 420-422) and
      evaluation jobs (nom + the 4-cell VAL_SET) for both variants.
  --analyze FILE --results DIR : per (variant, metric) means + disagreement flags.
  --smoke : tiny direct run verifying the threshold-reward env path end-to-end.
No submission happens here (pre-freeze); jobs are emitted as JSON.
"""
import argparse, glob, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SEEDS = (420, 421, 422)
PY = sys.executable


def evalcfg_for(config_name, tag):
    spec = json.load(open(os.path.join(HERE, "configs", config_name + ".json"), encoding="utf-8"))
    env = dict(spec.get("env", {}))
    out = {k: env[k] for k in ("probe", "stack", "filter_features") if k in env}
    name = "evalcfg__%s__%s.json" % (config_name, tag)
    path = os.path.join(HERE, "configs", name)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f)
    return "configs\\%s" % name


def emit_jobs(out_file, config, episodes, seed_base, split):
    jobs = []
    spec = json.load(open(os.path.join(HERE, "configs", config + ".json"), encoding="utf-8"))
    probe = spec.get("env", {}).get("probe", "ramsey_iq")
    # R2 config = same training config with reward_kind=threshold in env
    r2name = "%s__R2" % config
    r2spec = json.loads(json.dumps(spec))
    r2spec.setdefault("env", {})["reward_kind"] = "threshold"
    r2path = os.path.join(HERE, "configs", r2name + ".json")
    if not os.path.exists(r2path):
        with open(r2path, "w", encoding="utf-8") as f:
            json.dump(r2spec, f, indent=1)
    for variant, cname in (("R1", config), ("R2", r2name)):
        for s in SEEDS:
            jid = "s178_tr_%s_%s_s%d" % (variant, cname, s)
            jobs.append(dict(
                id=jid, lane="gpu", cwd=HERE,
                cmd='"%s" train_ppo.py --config configs\\%s.json --seed %d --out runs\\%s\\seed_%d '
                    '> "..\\..\\MiniProject_Sem4_QSTUDY_WORK\\results\\%s.log" 2>&1' % (PY, cname, s, cname, s, jid),
                depends_on=[], expected_outputs=[os.path.join(HERE, "runs", cname, "seed_%d" % s, "best_model.zip")],
                env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, stall_minutes=120))
    # evals on nom + VAL_SET cells
    import stress
    ecfg = evalcfg_for(config, "R12")
    for variant, cname in (("R1", config), ("R2", r2name)):
        for cond in ["nom"] + list(stress.VAL_SET):
            for s in SEEDS:
                res = r"results\rewardcheck\%s_s%d__%s.json" % (cname, s, cond)
                jobs.append(dict(
                    id="s178_eval_%s_s%d_%s" % (variant, s, cond), lane="cpu1", cwd=HERE,
                    cmd='"%s" evaluate.py --method rl:runs\\%s\\seed_%d --cfg %s --cond %s --split %s '
                        '--episodes %d --seed_base %d --out %s '
                        '> "..\\..\\MiniProject_Sem4_QSTUDY_WORK\\results\\s178_%s_%s_%d.log" 2>&1' %
                        (PY, cname, s, ecfg, cond, split, episodes, seed_base, res, variant, cond, s),
                    depends_on=["s178_tr_%s_%s_s%d" % (variant, cname, s)], expected_outputs=[res],
                    env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, stall_minutes=90))
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=1)
    print("emitted %d jobs -> %s" % (len(jobs), out_file))
    return 0


def analyze(results_dir, out_file):
    import collections
    agg = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        nm = os.path.basename(p)[:-5]
        variant = "R2" if "__R2" in nm else "R1"
        agg[variant].append(d)
    out_rows = []
    for variant, ds in sorted(agg.items()):
        keys = ["lock_w1000_mean", "lock_w500_mean", "rms_mean", "cvar95_rms", "lock_loss_prob",
                "p95_rms", "mean_reward_mean"]
        row = dict(variant=variant, n=len(ds))
        for k in keys:
            vals = [d[k] for d in ds if d.get(k) is not None]
            row[k] = float(np.mean(vals)) if vals else None
        out_rows.append(row)
    # disagreement: the variant with the higher TRAINING reward must not lose on the
    # scientific metrics; flag every metric where it does.
    disagreements = []
    if len(out_rows) == 2:
        r1, r2 = out_rows
        better = r2 if (r2.get("mean_reward_mean") or 0) > (r1.get("mean_reward_mean") or 0) else r1
        worse = r1 if better is r2 else r2
        for k in ("lock_w1000_mean", "lock_w500_mean"):
            if better.get(k) is not None and worse.get(k) is not None and better[k] < worse[k]:
                disagreements.append(dict(metric=k, better_reward_variant=better["variant"],
                                         better=better[k], worse=worse[k]))
        for k in ("rms_mean", "cvar95_rms", "lock_loss_prob", "p95_rms"):
            if better.get(k) is not None and worse.get(k) is not None and better[k] > worse[k]:
                disagreements.append(dict(metric=k, better_reward_variant=better["variant"],
                                         better=better[k], worse=worse[k]))
    out = dict(status="ok", rows=out_rows, disagreements=disagreements)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("analyze: %d variants, %d disagreements -> %s" % (len(out_rows), len(disagreements), out_file))
    return 0


def smoke(episodes, seed_base):
    from envs import BatchEnv, DEFAULT_CFG
    cfg = dict(DEFAULT_CFG)
    cfg.update(probe="ramsey_iq", reward_kind="threshold", horizon=30, burn_in=5)
    env = BatchEnv(cfg, episodes, seed_base)
    env.reset()
    rs = []
    for t in range(env.T):
        a = np.zeros((episodes, 1))
        obs, rew, done, info = env.step(a)
        rs.append(float(np.mean(rew)))
    print("SMOKE R2 threshold-reward mean over run: %.3f (finite=%s)" %
          (float(np.mean(rs)), bool(np.isfinite(rs).all())))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-jobs", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--analyze", default=None)
    ap.add_argument("--results", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed_base", type=int, default=500000)
    ap.add_argument("--split", default="val", choices=["tune", "val", "sealed"])
    a = ap.parse_args()
    if a.emit_jobs:
        return emit_jobs(a.emit_jobs, a.config, a.episodes, a.seed_base, a.split)
    if a.analyze:
        return analyze(a.results, a.analyze)
    if a.smoke:
        return smoke(a.episodes, a.seed_base)
    print("nothing to do: pass --emit-jobs/--analyze/--smoke")
    return 1


if __name__ == "__main__":
    sys.exit(main())
