"""qstudy eval_generalisation — Part-2 S17.7: generalisation-matrix driver.

DR-restricted training (drift kinds gaussian and ou ONLY; all other DR ranges as DR;
best probe; seeds 410-414) versus the full-DR models of Part 1, evaluated on:
  - held-out dynamics: onef, telegraph, sinusoid;
  - held-out parameters: the DR-OUT cells of O3 (stress.dr_support == "OUT");
  - observation shifts: shots 16384 and 64 with readout 0.10;
  - combined shifts: OUT sigma AND OUT readout (drift_sigma 4000 + readout 0.10).
Each shift is reported separately.

Modes:
  --emit-jobs FILE --config NAME --dr-model NAME : write DR-restricted training jobs
      (seeds 410-414) and evaluation jobs for both the DR-restricted and the full-DR
      model (the full-DR runs must already exist under runs\\<dr-model>\\seed_*).
  --analyze FILE --results DIR : lock_w1000/rms table per (model, shift).
  --smoke : tiny direct run verifying the DR-restricted env config path.
No submission happens here (pre-freeze); jobs are emitted as JSON.
"""
import argparse, glob, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SEEDS = tuple(range(410, 415))
PY = sys.executable


def shifts(probe):
    """(shift_id, cond_json) pairs; OFAT OUT cells are filtered by applicability."""
    import stress
    out = []
    for kind in ("onef", "telegraph", "sinusoid"):
        out.append(("dyn_%s" % kind, json.dumps({"drift": kind})))
    for cid in stress.list_conds("ofat"):
        if stress.dr_support(cid) != "OUT":
            continue
        if not stress.applies_to(cid, probe):
            continue
        out.append(("param_%s" % cid, cid))
    for shots in (16384, 64):
        out.append(("obs_shots%d_ro010" % shots,
                    json.dumps({"shots": int(shots), "e01": 0.10, "e10": 0.10})))
    out.append(("comb_sigma4000_ro010",
                json.dumps({"drift_sigma": 4000.0, "e01": 0.10, "e10": 0.10})))
    return out


def evalcfg_for(config_name, tag):
    """Plain eval cfg (probe/stack/filter_features) for a training config; tag in the
    file name keeps DR-restricted variants separate."""
    spec = json.load(open(os.path.join(HERE, "configs", config_name + ".json"), encoding="utf-8"))
    env = dict(spec.get("env", {}))
    out = {k: env[k] for k in ("probe", "stack", "filter_features") if k in env}
    name = "evalcfg__%s__%s.json" % (config_name, tag)
    path = os.path.join(HERE, "configs", name)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f)
    return "configs\\%s" % name


def emit_jobs(out_file, config, dr_model, episodes, seed_base, split):
    """config: DR-restricted training config NAME (a config json with the full DR env
    plus kinds [gaussian,ou] is WRITTEN here as configs\\<config>__drrestricted.json).
    dr_model: the full-DR Part-1 config name whose runs already exist."""
    jobs = []
    spec = json.load(open(os.path.join(HERE, "configs", config + ".json"), encoding="utf-8"))
    probe = spec.get("env", {}).get("probe", "ramsey_iq")
    # 1. DR-restricted training config: same as `config` but dr.kinds = [gaussian, ou]
    rname = "%s__drrestricted" % config
    rspec = json.loads(json.dumps(spec))
    rspec.setdefault("env", {}).setdefault("dr", {})
    rspec["env"]["dr"] = dict(rspec["env"].get("dr") or {})
    rspec["env"]["dr"]["kinds"] = ["gaussian", "ou"]
    rpath = os.path.join(HERE, "configs", rname + ".json")
    if not os.path.exists(rpath):
        with open(rpath, "w", encoding="utf-8") as f:
            json.dump(rspec, f, indent=1)
    # 2. trainings seeds 410-414 (gpu lane, no gpu deps)
    for s in SEEDS:
        jid = "s177_tr_%s_s%d" % (rname, s)
        jobs.append(dict(
            id=jid, lane="gpu", cwd=HERE,
            cmd='"%s" train_ppo.py --config configs\\%s.json --seed %d --out runs\\%s\\seed_%d '
                '> "..\\..\\MiniProject_Sem4_QSTUDY_WORK\\results\\%s.log" 2>&1' % (PY, rname, s, rname, s, jid),
            depends_on=[], expected_outputs=[os.path.join(HERE, "runs", rname, "seed_%d" % s, "best_model.zip")],
            env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, stall_minutes=120))
    # 3. evals: both models on every shift
    ecfg_r = evalcfg_for(rname, "eval")
    ecfg_f = evalcfg_for(dr_model, "eval")
    for shift_id, cond in shifts(probe):
        res_r = r"results\generalisation\drrestricted_%s__%s.json" % (rname, shift_id)
        jobs.append(dict(
            id="s177_eval_drrestricted_%s" % shift_id, lane="cpu1", cwd=HERE,
            cmd='"%s" evaluate.py --method rl:runs\\%s\\seed_%d --cfg %s --cond %s --split %s '
                '--episodes %d --seed_base %d --out %s '
                '> "..\\..\\MiniProject_Sem4_QSTUDY_WORK\\results\\s177_%s.log" 2>&1' %
                (PY, rname, SEEDS[0], ecfg_r, cond, split, episodes, seed_base, res_r, shift_id),
            depends_on=["s177_tr_%s_s%d" % (rname, SEEDS[0])], expected_outputs=[res_r],
            env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, stall_minutes=90))
        res_f = r"results\generalisation\fullDR_%s__%s.json" % (dr_model, shift_id)
        jobs.append(dict(
            id="s177_eval_fullDR_%s" % shift_id, lane="cpu1", cwd=HERE,
            cmd='"%s" evaluate.py --method rl:runs\\%s\\seed_%d --cfg %s --cond %s --split %s '
                '--episodes %d --seed_base %d --out %s '
                '> "..\\..\\MiniProject_Sem4_QSTUDY_WORK\\results\\s177f_%s.log" 2>&1' %
                (PY, dr_model, 0, ecfg_f, cond, split, episodes, seed_base, res_f, shift_id),
            depends_on=[], expected_outputs=[res_f],
            env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, stall_minutes=90))
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=1)
    print("emitted %d jobs -> %s" % (len(jobs), out_file))
    return 0


def analyze(results_dir, out_file):
    rows = []
    for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        rows.append(dict(name=os.path.basename(p)[:-5], method=d.get("method"),
                         lock_w1000=d.get("lock_w1000_mean"), rms=d.get("rms_mean"),
                         episodes=d.get("episodes")))
    out = dict(status="ok", rows=rows)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("analyze: %d rows -> %s" % (len(rows), out_file))
    return 0


def smoke(episodes, seed_base):
    from envs import BatchEnv, DEFAULT_CFG
    cfg = dict(DEFAULT_CFG)
    cfg.update(probe="ramsey_iq", horizon=40, burn_in=10,
               dr=dict(scale=1.0, kinds=["gaussian", "ou"]))
    env = BatchEnv(cfg, episodes, seed_base)
    env.reset()
    kinds = set()
    for d in env.drifts:
        kinds.add(d.kind)
        self_ok = d.kind in ("gaussian", "ou")
        if not self_ok:
            print("SMOKE FAIL: non-restricted kind sampled: %s" % d.kind)
            return 1
    print("SMOKE DR-restricted kinds sampled: %s" % sorted(kinds))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-jobs", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--dr-model", default=None)
    ap.add_argument("--analyze", default=None)
    ap.add_argument("--results", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed_base", type=int, default=500000)
    ap.add_argument("--split", default="val", choices=["tune", "val", "sealed"])
    a = ap.parse_args()
    if a.emit_jobs:
        return emit_jobs(a.emit_jobs, a.config, a.dr_model, a.episodes, a.seed_base, a.split)
    if a.analyze:
        return analyze(a.results, a.analyze)
    if a.smoke:
        return smoke(a.episodes, a.seed_base)
    print("nothing to do: pass --emit-jobs/--analyze/--smoke")
    return 1


if __name__ == "__main__":
    sys.exit(main())
