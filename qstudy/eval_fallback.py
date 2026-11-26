"""qstudy eval_fallback — Part-2 S17.5 driver: fallback wrapper on/off comparison
in the regime env and the nominal env, for bayes and RL-hybrid. The CUSUM detector
is ON in both arms (it is the alarm source); the only difference between arms is the
fallback wrapper. Reports failure severity (CVaR95/p95 of per-episode rms) next to
average performance (lock_w1000 mean).
"""
import argparse, json, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from envs import BatchEnv, DEFAULT_CFG, mean_reward
from evaluate import cvar95, git_head, load_rl, sha256_file, versions, METRIC_KEYS
from changepoint import BayesCusum, REGIME_DEFAULTS
from fallback import FallbackController
from eval_changepoint import sealed_gate


def build_cfg(env_kind, base_cfg):
    c = dict(DEFAULT_CFG)
    c.update(base_cfg or {})
    if env_kind == "regime":
        c["drift"] = "regime"
        c["drift_params"] = dict(REGIME_DEFAULTS)
    return c


def run(cfg, B, seed_base, method, model_dir, fallback_on):
    env = BatchEnv(cfg, B, seed_base)
    obs = env.reset()
    T = env.T
    e_tr = np.zeros((T, B)); r_tr = np.zeros((T, B)); a_tr = np.zeros((T, B))
    act_scale_tr = np.zeros((T, B))
    rl = ctrl = None
    rl_wrap = None
    norm = None
    if method.startswith("rl:"):
        rl, norm = load_rl(model_dir)
        obs = norm(obs)
        if not cfg.get("filter_features") or not cfg.get("cusum"):
            raise SystemExit("rl hybrid requires filter_features=True and cusum=True in cfg")
        if fallback_on:
            rl_wrap = FallbackController(_RlInner(np.zeros((B, 1))),
                                         sigma_provider=lambda: env._flt.last_std,
                                         alarm_provider=lambda: env._flt.last_alarm, B=B)
    else:
        if method != "ctrl:bayes:nominal":
            raise SystemExit("eval_fallback supports ctrl:bayes:nominal and rl:<dir>")
        inner = BayesCusum(cfg, assumed="nominal", env_view=env)
        if fallback_on:
            ctrl = FallbackController(inner, sigma_provider=lambda: inner.last_std,
                                      alarm_provider=lambda: inner.last_alarm, B=B)
        else:
            ctrl = inner
    t0 = time.time()
    for t in range(T):
        if rl is not None:
            a_raw, _ = rl.predict(obs, deterministic=True)
            a_raw = np.asarray(a_raw, dtype=float).reshape(-1, 1)
            if fallback_on:
                rl_wrap.inner._a = a_raw
                a = rl_wrap.act(None).reshape(-1, 1)
                act_scale_tr[t] = np.where(rl_wrap.active, 0.5, 1.0)
            else:
                a = a_raw
        else:
            a = ctrl.act(None).reshape(-1, 1)
            if fallback_on:
                act_scale_tr[t] = np.where(ctrl.active, 0.5, 1.0)
        obs, rew, done, info = env.step(a)
        e_tr[t] = info["e"]
        r_tr[t] = rew
        a_tr[t] = a.reshape(-1)
        if rl is not None:
            obs = norm(obs)
        if time.time() - t0 > 60.0:
            print("PROGRESS t=%d/%d" % (t + 1, T), flush=True)
            t0 = time.time()
    m = env.metrics(e_tr)
    m["mean_reward"] = mean_reward(r_tr, int(cfg.get("burn_in", 20)))
    m["cvar95_rms"] = cvar95(m["rms"])
    m["cvar95_lockmiss"] = cvar95(1.0 - m["lock_w1000"])
    m["p95_rms"] = float(np.percentile(np.asarray(m["rms"], dtype=float), 95))
    m["lock_loss_prob"] = float(np.asarray(m["lock_lost"]).mean())
    m["active_frac"] = (act_scale_tr < 1.0).mean(axis=0) if fallback_on else np.zeros(B)
    return m, e_tr


class _RlInner:
    """Adapter: a mutable raw-action holder used as the wrapper's 'inner controller'
    for one step of an RL policy."""
    def __init__(self, a):
        self._a = a

    def act(self, obs_or_state=None):
        return self._a.reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--cfg", default=None)
    ap.add_argument("--fallback", required=True, choices=["on", "off"])
    ap.add_argument("--env", required=True, choices=["regime", "nominal"])
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
    cfg = build_cfg(a.env, base_cfg)
    model_dir = None
    if a.method.startswith("rl:"):
        model_dir = a.method[3:]
        if not os.path.isdir(model_dir):
            print("REFUSED: model_dir not found: %s" % model_dir, flush=True)
            return 3
    fallback_on = a.fallback == "on"
    m, e_tr = run(cfg, a.episodes, a.seed_base, a.method, model_dir, fallback_on)
    npz = a.out[:-5] + ".npz" if a.out.endswith(".json") else a.out + ".npz"
    arrays = {k: np.asarray(m[k]) for k in m if not k.startswith("_")}
    arrays["trace_e"] = e_tr
    np.savez_compressed(npz, **arrays)
    out = {
        "status": "ok", "method": a.method, "model_dir": model_dir,
        "fallback": a.fallback, "env": a.env, "split": a.split,
        "episodes": a.episodes, "seed_base": a.seed_base,
        "cond_id": "fallback_%s" % a.env,
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
    out["active_frac_mean"] = float(np.asarray(m["active_frac"]).mean())
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("DONE %s fallback=%s env=%s episodes=%d lock_w1000=%.3f cvar95_rms=%.1f active=%.2f" %
          (a.method, a.fallback, a.env, a.episodes, out["lock_w1000_mean"], out["cvar95_rms"],
           out["active_frac_mean"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
