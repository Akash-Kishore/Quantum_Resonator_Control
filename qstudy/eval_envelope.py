"""qstudy eval_envelope — Part-2 S17.6: drift-speed x delay envelope driver.

Grid: sinusoid amplitude A in {500,1000,2000,4000} Hz, period P in {8,16,32,64,128}
steps (random phase; the sinusoid drift kind carries the small OU jitter), action
delay d in {0,1,2,3}; probes ramsey_iq and dither. Methods: the frozen candidates
C1-C6 (config names; seeds 100-102), pid, bayes:nominal, bayes:oracle.

Modes (no submission happens here; jobs are emitted as JSON for later submit):
  --emit-jobs FILE --configs cfg1,cfg2,... : write training jobs (C1-C6 x seeds 100-102)
      and evaluation jobs per (method, cell); result paths under results\\envelope\\.
  --analyze FILE --results DIR : read the envelope result JSONs, find per method the
      boundary where lock_w1000 falls below 0.90, and compare with the slew-rate bound
      2*pi*A/P versus action_scale (3000 Hz/step) and the delay-induced error
      A*2*pi*d/P. Writes the boundary table to FILE.
  --smoke : tiny direct run of one cell (pid, few episodes) to verify wiring.
"""
import argparse, glob, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

AMPS = (500, 1000, 2000, 4000)
PERIODS = (8, 16, 32, 64, 128)
DELAYS = (0, 1, 2, 3)
PROBES = ("ramsey_iq", "dither")
SEEDS = (100, 101, 102)
BOUNDARY = 0.90
ACTION_SCALE = 3000.0
PY = sys.executable


def cell_id(A, P, d):
    return "envl_A%d_P%d_d%d" % (A, P, d)


def cell_cond(A, P, d):
    return json.dumps(dict(drift="sinusoid", drift_params=dict(amp=float(A), period_range=[float(P), float(P)]),
                           action_delay=int(d)))


def evalcfg_for(config_name):
    """Write (idempotently) a plain eval cfg for a training config: only obs-relevant
    env keys (probe, stack, filter_features). Returns the configs-relative file name."""
    spec = json.load(open(os.path.join(HERE, "configs", config_name + ".json"), encoding="utf-8"))
    env = dict(spec.get("env", {}))
    out = {k: env[k] for k in ("probe", "stack", "filter_features") if k in env}
    name = "evalcfg__%s.json" % config_name
    path = os.path.join(HERE, "configs", name)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f)
    return "configs\\%s" % name


def emit_jobs(configs, out_file, episodes, seed_base, split):
    jobs = []
    results_dir = r"results\envelope"
    # trainings: one per (config, seed); NEVER gpu->gpu deps (O1); no deps at all
    for cfg_name in configs:
        for s in SEEDS:
            jid = "s176_tr_%s_s%d" % (cfg_name, s)
            jobs.append(dict(
                id=jid, lane="gpu", cwd=HERE,
                cmd='"%s" train_ppo.py --config configs\\%s.json --seed %d --out runs\\%s\\seed_%d '
                    '> "..\\..\\MiniProject_Sem4_QSTUDY_WORK\\results\\%s.log" 2>&1' % (PY, cfg_name, s, cfg_name, s, jid),
                depends_on=[], expected_outputs=[os.path.join(HERE, "runs", cfg_name, "seed_%d" % s, "best_model.zip")],
                env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, stall_minutes=120))
    # evals: pid/bayes/bayes:oracle per cell; RL per (config, seed) per cell
    for probe in PROBES:
        for A in AMPS:
            for P in PERIODS:
                for d in DELAYS:
                    cid = cell_id(A, P, d)
                    cond = cell_cond(A, P, d)
                    base_cfg = "configs\\eval__%s.json" % probe
                    for method in ("ctrl:pid", "ctrl:bayes:nominal", "ctrl:bayes:oracle"):
                        short = method.replace("ctrl:", "").replace(":", "_")
                        res = "%s\\%s__%s.json" % (results_dir, short, cid)
                        jobs.append(dict(
                            id="s176_eval_%s__%s" % (short, cid), lane="cpu1", cwd=HERE,
                            cmd='"%s" evaluate.py --method %s --cfg %s --cond %s --split %s '
                                '--episodes %d --seed_base %d --out %s '
                                '> "..\\..\\MiniProject_Sem4_QSTUDY_WORK\\results\\%s.log" 2>&1' %
                                (PY, method, base_cfg, cond, split, episodes, seed_base, res, cid),
                            depends_on=[], expected_outputs=[res],
                            env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, stall_minutes=90))
                    for cfg_name in configs:
                        ecfg = evalcfg_for(cfg_name)
                        for s in SEEDS:
                            res = "%s\\rl_%s_s%d__%s.json" % (results_dir, cfg_name, s, cid)
                            jobs.append(dict(
                                id="s176_eval_rl_%s_s%d__%s" % (cfg_name, s, cid), lane="cpu2", cwd=HERE,
                                cmd='"%s" evaluate.py --method rl:%s --cfg %s --cond %s --split %s '
                                    '--episodes %d --seed_base %d --out %s '
                                    '> "..\\..\\MiniProject_Sem4_QSTUDY_WORK\\results\\%s.log" 2>&1' %
                                    (PY, r"runs\%s\seed_%d" % (cfg_name, s), ecfg, cond, split,
                                     episodes, seed_base, res, cid),
                                depends_on=["s176_tr_%s_s%d" % (cfg_name, s)], expected_outputs=[res],
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
                         lock_w1000=d.get("lock_w1000_mean"), episodes=d.get("episodes")))
    # per method+probe: for each (P, d) find the smallest A whose mean lock_w1000 < 0.90
    by = {}
    for r in rows:
        nm = r["name"]
        parts = nm.replace(".json", "").split("__")
        key = parts[0]                      # method/config id
        cell = parts[-1]                    # envl_A.._P.._d..
        A = int(cell.split("_")[1][1:]); P = int(cell.split("_")[2][1:]); d = int(cell.split("_")[3][1:])
        by.setdefault(key, {}).setdefault((P, d), []).append((A, r["lock_w1000"]))
    table = []
    for key, cells in sorted(by.items()):
        for (P, d), pts in sorted(cells.items()):
            ok = sorted([(A, v) for A, v in pts if v is not None and v >= BOUNDARY])
            bad = sorted([(A, v) for A, v in pts if v is not None and v < BOUNDARY])
            boundary = bad[0][0] if bad else None      # smallest failing A
            slew_A = ACTION_SCALE * P / (2 * np.pi)    # A at which 2*pi*A/P equals the action scale
            delay_err = boundary * 2 * np.pi * d / P if boundary else None
            table.append(dict(method=key, P=P, d=d, boundary_A=boundary,
                              slew_bound_A=float(slew_A), delay_induced_err_Hz=delay_err,
                              n_cells=len(pts)))
    out = dict(status="ok", boundary=BOUNDARY, action_scale=ACTION_SCALE, rows=rows, table=table)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("analyze: %d result rows, %d boundary rows -> %s" % (len(rows), len(table), out_file))
    return 0


def smoke(episodes, seed_base):
    from envs import BatchEnv, DEFAULT_CFG
    from evaluate import make_controller
    cfg = dict(DEFAULT_CFG)
    cfg.update(probe="ramsey_iq", drift="sinusoid",
               drift_params=dict(amp=2000.0, period_range=[16.0, 16.0]), action_delay=1,
               horizon=40, burn_in=10)
    env = BatchEnv(cfg, episodes, seed_base)
    env.reset()
    ctrl = make_controller("ctrl:pid", cfg, env)
    e_tr = []
    for t in range(env.T):
        a = ctrl.act(None).reshape(-1, 1)
        obs, rew, done, info = env.step(a)
        e_tr.append(info["e"].copy())
    m = env.metrics(np.asarray(e_tr))
    print("SMOKE envelope cell A=2000 P=16 d=1 pid lock_w1000=%.3f" % float(np.mean(m["lock_w1000"])))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-jobs", default=None)
    ap.add_argument("--configs", default="")
    ap.add_argument("--analyze", default=None)
    ap.add_argument("--results", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed_base", type=int, default=500000)
    ap.add_argument("--split", default="val", choices=["tune", "val", "sealed"])
    a = ap.parse_args()
    if a.emit_jobs:
        return emit_jobs([c for c in a.configs.split(",") if c], a.emit_jobs, a.episodes, a.seed_base, a.split)
    if a.analyze:
        return analyze(a.results, a.analyze)
    if a.smoke:
        return smoke(a.episodes, a.seed_base)
    print("nothing to do: pass --emit-jobs/--analyze/--smoke")
    return 1


if __name__ == "__main__":
    sys.exit(main())
