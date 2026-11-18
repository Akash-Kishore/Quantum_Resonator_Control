"""qstudy eval_changepoint — Part-2 S17.2 driver: regime-switching environment evaluation
with the CUSUM detector on/off (bayes:nominal and RL-hybrid filter features), metrics per jump.

Sealed runs use split=sealed, seed_base 960000 (gated on PREREG2.sha256 exactly like
evaluate.py). Smoke: small --episodes run with the bayes controller on cpu lanes.
"""
import argparse, json, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import stress
from envs import BatchEnv, DEFAULT_CFG, mean_reward
from evaluate import cvar95, first_lock_time, git_head, load_rl, sha256_file, versions, METRIC_KEYS
from changepoint import BayesCusum, jump_metrics, REGIME_DEFAULTS, REGIME_HELDOUT


def sealed_gate(split, seed_base):
    if split != "sealed":
        return True
    base_prereg = {900000: ("PREREG.md", "PREREG.sha256"),
                   920000: ("PREREG2.md", "PREREG2.sha256"),
                   950000: ("PREREG2.md", "PREREG2.sha256"),
                   960000: ("PREREG2.md", "PREREG2.sha256")}.get(seed_base)
    if base_prereg is None:
        return False
    sha_path = os.path.join(HERE, base_prereg[1])
    if not os.path.exists(sha_path):
        return False
    want = open(sha_path, encoding="utf-8").read().strip().split()[0]
    return want == sha256_file(os.path.join(HERE, base_prereg[0]))


def regime_cfg(family, base_cfg):
    c = dict(DEFAULT_CFG)
    c.update(base_cfg or {})
    c["drift"] = "regime"
    c["drift_params"] = dict(REGIME_HELDOUT if family == "heldout" else REGIME_DEFAULTS)
    return c


def run(cfg, B, seed_base, method, model_dir, detector_on):
    env = BatchEnv(cfg, B, seed_base)
    obs = env.reset()
    T = env.T
    e_tr = np.zeros((T, B)); alarm_tr = np.zeros((T, B), bool); shots_tr = np.zeros((T, B))
    r_tr = np.zeros((T, B)); a_tr = np.zeros((T, B))
    rl = ctrl = None
    norm = None
    if method.startswith("rl:"):
        rl, norm = load_rl(model_dir)
        obs = norm(obs)
        if not cfg.get("filter_features"):
            raise SystemExit("rl hybrid requires filter_features=True in cfg")
        if bool(cfg.get("cusum")) != detector_on:
            raise SystemExit("cfg cusum flag must match --detector")
    else:
        if method != "ctrl:bayes:nominal":
            raise SystemExit("eval_changepoint supports ctrl:bayes:nominal and rl:<dir>")
        if detector_on:
            ctrl = BayesCusum(cfg, assumed="nominal", env_view=env)
        else:
            from controllers import BayesTracker
            ctrl = BayesTracker(cfg, assumed="nominal", env_view=env)
    t0 = time.time()
    for t in range(T):
        if rl is not None:
            a, _ = rl.predict(obs, deterministic=True)
            a = np.asarray(a, dtype=float).reshape(-1, 1)
        else:
            a = ctrl.act(None).reshape(-1, 1)
            if detector_on and t > 0:
                alarm_tr[t - 1] = np.asarray(ctrl.last_alarm, dtype=bool)
        obs, rew, done, info = env.step(a)
        e_tr[t] = info["e"]
        shots_tr[t] = info["shots_used"]
        r_tr[t] = rew
        a_tr[t] = a.reshape(-1)
        if rl is not None:
            flt = getattr(env, "_flt", None)
            if flt is not None and getattr(flt, "last_alarm", None) is not None:
                alarm_tr[t] = np.asarray(flt.last_alarm, dtype=bool)
            obs = norm(obs)
        if time.time() - t0 > 60.0:
            print("PROGRESS t=%d/%d" % (t + 1, T), flush=True)
            t0 = time.time()
    if ctrl is not None and detector_on:
        ctrl.act(None)  # process the final measurement so the last alarm is recorded
        alarm_tr[T - 1] = np.asarray(ctrl.last_alarm, dtype=bool)
    jump_events = [[(tj - 1, size) for tj, size in d.jump_log] for d in env.drifts]
    jm = jump_metrics(e_tr, alarm_tr, shots_tr, jump_events,
                      burn_in=int(cfg.get("burn_in", 20)))
    m = env.metrics(e_tr)
    m["mean_reward"] = mean_reward(r_tr, int(cfg.get("burn_in", 20)))
    m["cvar95_rms"] = cvar95(m["rms"])
    m["cvar95_lockmiss"] = cvar95(1.0 - m["lock_w1000"])
    m["p95_rms"] = float(np.percentile(np.asarray(m["rms"], dtype=float), 95))
    m["lock_loss_prob"] = float(np.asarray(m["lock_lost"]).mean())
    m["first_lock_time"] = np.array([first_lock_time(e_tr[:, b], 1000.0) for b in range(B)], dtype=float)
    return m, jm, e_tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--cfg", default=None)
    ap.add_argument("--detector", required=True, choices=["on", "off"])
    ap.add_argument("--family", default="train", choices=["train", "heldout"])
    ap.add_argument("--split", required=True, choices=["tune", "val", "sealed"])
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--seed_base", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        print("SKIP (exists): %s" % a.out)
        return 0
    if not sealed_gate(a.split, a.seed_base):
        print("SEALED REFUSED: gate failed (split=%s seed_base=%d)" % (a.split, a.seed_base), flush=True)
        return 3
    base_cfg = {}
    if a.cfg:
        base_cfg = json.loads(a.cfg) if a.cfg.strip().startswith("{") else json.load(open(a.cfg, encoding="utf-8"))
    cfg = regime_cfg(a.family, base_cfg)
    model_dir = None
    if a.method.startswith("rl:"):
        model_dir = a.method[3:]
        if not os.path.isdir(model_dir):
            print("REFUSED: model_dir not found: %s" % model_dir, flush=True)
            return 3
    detector_on = a.detector == "on"
    m, jm, e_tr = run(cfg, a.episodes, a.seed_base, a.method, model_dir, detector_on)
    per_jump = jm.pop("per_jump")
    npz = a.out[:-5] + ".npz" if a.out.endswith(".json") else a.out + ".npz"
    arrays = {k: np.asarray(m[k]) for k in m if not k.startswith("_")}
    arrays["trace_e"] = e_tr
    arrays["jump_detected"] = np.asarray([r["detected"] for r in per_jump], dtype=float)
    np.savez_compressed(npz, **arrays)
    out = {
        "status": "ok", "method": a.method, "model_dir": model_dir,
        "detector": a.detector, "family": a.family, "split": a.split,
        "episodes": a.episodes, "seed_base": a.seed_base,
        "cond_id": "regime_" + a.family, "drift_params": cfg["drift_params"],
        "git_head": git_head(), "versions": versions(),
        "cmd": " ".join(sys.argv), "npz_path": npz, "npz_sha256": sha256_file(npz),
    }
    for k in METRIC_KEYS:
        arr = np.asarray(m[k], dtype=float)
        out[k + "_mean"] = float(arr.mean())
        out[k + "_sd"] = float(arr.std())
    out["mean_reward_mean"] = float(np.asarray(m["mean_reward"]).mean())
    out["cvar95_rms"] = float(m["cvar95_rms"])
    out["cvar95_lockmiss"] = float(m["cvar95_lockmiss"])
    out["p95_rms"] = float(m["p95_rms"])
    out["lock_loss_prob"] = float(m["lock_loss_prob"])
    flt = np.asarray(m["first_lock_time"], dtype=float)
    locked = flt[flt >= 0]
    out["first_lock_time_mean"] = float(locked.mean()) if locked.size else None
    out["first_lock_time_never_frac"] = float((flt < 0).mean())
    out.update(jm)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("DONE %s detector=%s family=%s episodes=%d det_rate=%s false/100=%s" %
          (a.method, a.detector, a.family, a.episodes, out["detection_rate"], out["false_alarms_per100"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
